# 🚀 Async FastAPI + pgvector RAG Template

Производственный шаблон высокопроизводительного асинхронного сервиса на **FastAPI**, **PostgreSQL** и **pgvector** для построения RAG-систем (Retrieval-Augmented Generation) и векторизации знаний.

## 🏛️ Архитектура и технологии
- **FastAPI**: Асинхронный REST API веб-каркас.
- **PostgreSQL + pgvector**: Векторная база данных с HNSW-индексами для мгновенного косинусного поиска.
- **asyncpg**: Высокоскоростной асинхронный драйвер PostgreSQL на C.
- **Pydantic v2**: Строгая типизация и валидация DTO контрактов.
- **LiteLLM**: Оркестрация и каскадная маршрутизация вызовов ИИ-моделей.

## 🛠️ Запуск проекта

### 1. Поднятие PostgreSQL с pgvector
docker run -d --name rag_postgres -e POSTGRES_USER=user -e POSTGRES_PASSWORD=password -e POSTGRES_DB=rag_db -p 5432:5432 pgvector/pgvector:pg16

### 2. Установка зависимостей и запуск
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

## 📡 API Эндпоинты
- `GET /health` — Проверка статуса сервиса и подключения к БД.
- `POST /api/v1/documents` — Индексация документов и запись векторов в pgvector.
- `POST /api/v1/rag/query` — Выполнение семантического RAG-поиска и генерация ответа.
