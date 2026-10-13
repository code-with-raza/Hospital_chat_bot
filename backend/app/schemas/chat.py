from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    response: str
    type: str = "text"
    data: object | None = None
