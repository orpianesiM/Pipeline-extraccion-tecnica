import asyncio

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import chain as pipeline
from schemas import ExtraccionTecnica, NivelCriticidad


class StubChain:
    def __init__(self, result: ExtraccionTecnica) -> None:
        self.result = result
        self.received_input: dict[str, str] | None = None

    async def ainvoke(self, input: dict[str, str]) -> ExtraccionTecnica:
        self.received_input = input
        return self.result


def test_schema_requires_technology_and_non_empty_summary() -> None:
    with pytest.raises(ValueError):
        ExtraccionTecnica(
            tecnologias=[],
            nivel_de_criticidad=NivelCriticidad.MEDIA,
            resumen_tecnico="Resumen",
        )

    with pytest.raises(ValueError):
        ExtraccionTecnica(
            tecnologias=["FastAPI"],
            nivel_de_criticidad=NivelCriticidad.MEDIA,
            resumen_tecnico="   ",
        )


def test_process_text_trims_input_and_returns_validated_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ExtraccionTecnica(
        tecnologias=["FastAPI"],
        nivel_de_criticidad=NivelCriticidad.ALTA,
        resumen_tecnico="La API falla al agotarse el pool de conexiones.",
    )
    stub = StubChain(expected)
    monkeypatch.setattr(pipeline, "resilient_chain", stub)

    result = asyncio.run(pipeline.process_text("  error de conexión  "))

    assert result is expected
    assert stub.received_input == {"text": "error de conexión"}


def test_process_text_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no puede estar vacío"):
        asyncio.run(pipeline.process_text("   "))


def test_chain_retries_after_invalid_model_response() -> None:
    expected = ExtraccionTecnica(
        tecnologias=["FastAPI"],
        nivel_de_criticidad=NivelCriticidad.MEDIA,
        resumen_tecnico="API web.",
    )
    attempts = 0

    def fail_once_then_return(_input: object) -> ExtraccionTecnica:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("Respuesta estructurada incompleta.")
        return expected

    test_chain = pipeline.build_chain(RunnableLambda(fail_once_then_return))
    result = asyncio.run(test_chain.ainvoke({"text": "API FastAPI"}))

    assert result is expected
    assert attempts == 2


def test_chain_retries_when_model_response_was_truncated() -> None:
    expected = ExtraccionTecnica(
        tecnologias=["FastAPI"],
        nivel_de_criticidad=NivelCriticidad.MEDIA,
        resumen_tecnico="API web.",
    )
    attempts = 0

    def truncate_once_then_return(_input: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {
                "raw": AIMessage(
                    content="",
                    response_metadata={"finish_reason": "length"},
                ),
                "parsed": expected,
                "parsing_error": None,
            }
        return {
            "raw": AIMessage(
                content="{}",
                response_metadata={"finish_reason": "stop"},
            ),
            "parsed": expected,
            "parsing_error": None,
        }

    test_chain = pipeline.build_chain(RunnableLambda(truncate_once_then_return))
    result = asyncio.run(test_chain.ainvoke({"text": "API FastAPI"}))

    assert result is expected
    assert attempts == 2
