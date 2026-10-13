from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from app.graph.graph import chat_graph
from app.schemas.chat import ChatRequest, ChatResponse

import logging
logger = logging.getLogger(__name__)

app = FastAPI(title="Hospital AI Chat Support", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5175",
    "http://127.0.0.1:5175",
    "http://localhost:5173",
    "http://127.0.0.1:5173",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": request.session_id}}

    try:
        previous_state = chat_graph.get_state(config).values
        history = list(previous_state.get("conversation_history", []))[-12:]
        result = chat_graph.invoke(
            {"user_message": request.message, "conversation_history": history},
            config=config,
        )

        # Store one user/assistant pair per completed request in the same
        # checkpoint. The history is scoped by LangGraph's session thread ID.
        chat_graph.update_state(
            config,
            {
                "conversation_history": [
                    *history,
                    {"role": "user", "content": request.message},
                    {"role": "assistant", "content": result.get("response", "")},
                ][-12:]
            },
        )

        logger.info("Chat request processed")

        return ChatResponse(
            session_id=request.session_id,
            response=result.get(
                "response",
                "I couldn't complete that request.",
            ),
            type=result.get("response_type", "text"),
            data=result.get("response_data"),
        )

    except Exception as error:
        # Exception messages can contain user supplied medical or patient data.
        logger.error(
            "Chat request failed (error_type=%s); returning the generic response.",
            type(error).__name__,
        )
        return ChatResponse(
            session_id=request.session_id,
            response=(
                "I'm having trouble accessing hospital records right now. "
                "Please try again shortly."
            ),
            type="error",
        )

