"""
Pydantic-схемы (DTO) для валидации входящих и исходящих данных RAG-сервиса.
"""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    content: str = Field(..., min_length=1, description="Текст документа или фрагмента")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Метаданные (источник, автор, дата)")


class DocumentResponse(BaseModel):
    id: int
    content: str
    metadata: Dict[str, Any]
    status: str = "indexed"


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Вопрос пользователя на естественном языке")
    top_k: int = Field(default=5, ge=1, le=20, description="Количество релевантных контекстов для извлечения")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0, description="Температура генерации LLM")


class ContextChunk(BaseModel):
    id: int
    content: str
    similarity_score: float
    metadata: Dict[str, Any]


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[ContextChunk]
    model_used: str
    execution_time_ms: float
