"""
Асинхронный сервис FastAPI для гибридного RAG-поиска и работы с векторной базой знаний.
Архитектура: Asyncio + asyncpg + pgvector + Pydantic v2.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from schemas import DocumentCreate, DocumentResponse, RAGQueryRequest, RAGQueryResponse
from services import RAGService

# Глобальный пул соединений PostgreSQL
db_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Управление жизненным циклом приложения и пулом БД."""
    global db_pool
    # Инициализация пула соединений PostgreSQL
    db_pool = await asyncpg.create_pool(
        dsn="postgresql://user:password@localhost:5432/rag_db",
        min_size=5,
        max_size=20,
    )
    
    # Автоматическое включение расширения pgvector при старте
    async with db_pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}'::jsonb,
                embedding vector(1024),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        # Создание HNSW индекса для быстрого косинусного поиска
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
            ON document_chunks USING hnsw (embedding vector_cosine_ops);
        """)

    yield

    # Корректное закрытие пула при остановке сервера
    if db_pool:
        await db_pool.close()


app = FastAPI(
    title="Async FastAPI RAG Service",
    description="Производственный шаблон сервиса векторного поиска и RAG-генерации на базе pgvector и LiteLLM.",
    version="1.0.0",
    lifespan=lifespan,
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Эндпоинт проверки работоспособности сервиса и БД."""
    if not db_pool:
        raise HTTPException(status_code=500, detail="БД не инициализирована")
    return {"status": "ok", "database": "connected"}


@app.post("/api/v1/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def add_document(doc: DocumentCreate):
    """Индексация документа: генерация векторов и запись в pgvector."""
    if not db_pool:
        raise HTTPException(status_code=500, detail="БД недоступна")
    
    rag_service = RAGService(db_pool)
    result = await rag_service.index_document(doc.content, doc.metadata)
    return result


@app.post("/api/v1/rag/query", response_model=RAGQueryResponse)
async def query_rag(request: RAGQueryRequest):
    """Выполнение гибридного RAG-поиска и генерация ответа через LLM."""
    if not db_pool:
        raise HTTPException(status_code=500, detail="БД недоступна")
    
    rag_service = RAGService(db_pool)
    response = await rag_service.execute_rag_pipeline(
        query=request.query,
        top_k=request.top_k,
        temperature=request.temperature,
    )
    return response
