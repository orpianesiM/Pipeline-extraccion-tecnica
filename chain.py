import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from schemas import ExtraccionTecnica

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extrae entidades técnicas del texto proporcionado. "
            "Devuelve únicamente información respaldada por el texto y completa "
            "todos los campos del esquema estructurado. "
            "Si no puedes identificar tecnologías, usa [\"No identificada\"]. "
            "Asigna la criticidad según el impacto descrito; si no hay evidencia "
            "suficiente, usa \"media\". El resumen debe ser breve y técnico.",
        ),
        ("human", "Analiza este texto:\n\n{text}"),
    ]
)

def _log_attempt(prompt_value: Any) -> Any:
    logger.info("Enviando solicitud de extracción al modelo.")
    return prompt_value


def _ensure_validated(result: Any) -> ExtraccionTecnica:
    if isinstance(result, dict) and {
        "raw",
        "parsed",
        "parsing_error",
    }.issubset(result):
        if result["parsing_error"] is not None:
            raise ValueError("No se pudo interpretar la salida estructurada.") from result[
                "parsing_error"
            ]

        raw_message = result["raw"]
        finish_reason = getattr(raw_message, "response_metadata", {}).get(
            "finish_reason"
        )
        if finish_reason == "length":
            logger.warning("La salida del modelo fue truncada por límite de tokens.")
            raise ValueError("La respuesta estructurada quedó incompleta.")

        result = result["parsed"]

    if isinstance(result, ExtraccionTecnica):
        return result
    if isinstance(result, dict):
        return ExtraccionTecnica.model_validate(result)
    raise TypeError(
        "El modelo no devolvió un objeto ExtraccionTecnica validado "
        f"(tipo recibido: {type(result).__name__})."
    )


def build_chain(
    structured_model: Runnable[Any, Any] | None = None,
) -> Runnable[Any, ExtraccionTecnica]:
    if structured_model is None:
        structured_model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        ).with_structured_output(ExtraccionTecnica, include_raw=True)

    chain = (
        prompt
        | RunnableLambda(_log_attempt)
        | structured_model
        | RunnableLambda(_ensure_validated)
    )
    return chain.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
        retry_if_exception_type=(
            OutputParserException,
            ValidationError,
            ValueError,
            TypeError,
        ),
    )


resilient_chain: Runnable[Any, ExtraccionTecnica] | None = None


async def process_text(text: str) -> ExtraccionTecnica:
    if not text or not text.strip():
        raise ValueError("El texto de entrada no puede estar vacío.")

    logger.info("Iniciando extracción técnica.")
    try:
        global resilient_chain
        if resilient_chain is None:
            resilient_chain = build_chain()
        result = await resilient_chain.ainvoke({"text": text.strip()})
        logger.info("Respuesta recibida y validada correctamente.")
        return result
    except Exception:
        logger.exception("Falló la extracción técnica después de los reintentos.")
        raise
