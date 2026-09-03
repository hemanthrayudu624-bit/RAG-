"""
LLMService
-----------
Wraps the Groq chat completion API. This is the only service that talks
to Groq — swapping providers later only means editing this file.
"""
from functools import lru_cache
from typing import List, Dict

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below. If the answer is not contained in the "
    "context, say you don't have enough information in the provided "
    "documents — do not make things up. Be concise and cite which "
    "part of the context you used when helpful."
)


class LLMService:
    def __init__(self):
        from groq import Groq

        if not settings.groq_api_key:
            logger.warning(
                "GROQ_API_KEY is not set. Requests to Groq will fail until "
                "you set it in your .env file."
            )
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def generate_answer(
        self,
        query: str,
        context_chunks: List[str],
        chat_history: List[Dict[str, str]] = None,
    ) -> str:
        """Call Groq's chat completion endpoint with retrieved context."""
        context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else (
            "No relevant context was found in the knowledge base."
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if chat_history:
            messages.extend(chat_history)

        user_message = (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above."
        )
        messages.append({"role": "user", "content": user_message})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
        )
        return completion.choices[0].message.content


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()
