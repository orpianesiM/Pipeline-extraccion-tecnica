# Pipeline de Extracción de Entidades Técnicas

Pipeline asíncrono que recibe un texto sin procesar y devuelve una extracción
validada con tecnologías, nivel de criticidad y resumen técnico. Usa LangChain
LCEL, `ChatOpenAI` y Pydantic.

## Requisitos

- Python 3.11 o superior
- Una clave de API de OpenAI

## Instalación y configuración

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Configura `OPENAI_API_KEY` en `.env`. Opcionalmente, cambia `OPENAI_MODEL` o
`LOG_LEVEL`.

## Ejecutar el ejemplo asíncrono

```powershell
python example_async.py
```

Salida orientativa:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API usa Redis y PostgreSQL; el pool agotado provoca errores 503 durante picos de tráfico."
}
```

## Uso desde Python

```python
from chain import process_text

result = await process_text("La API FastAPI almacena datos en PostgreSQL.")
print(result.model_dump(mode="json"))
```

`schemas.py` define el contrato Pydantic. `chain.py` compone el prompt modular,
el modelo con salida estructurada y una validación final mediante LCEL. La
cadena reintenta hasta tres veces errores de parseo o validación y registra el
inicio de cada intento, la validación y el fallo final.

Ejecuta las pruebas locales sin llamar al modelo:

```powershell
python -m pytest
```
