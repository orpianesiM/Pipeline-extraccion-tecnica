import asyncio
import json

from chain import process_text


async def main() -> None:
    text = (
        "La API FastAPI almacena sesiones en Redis y persiste órdenes en "
        "PostgreSQL. El pool de conexiones se agota durante picos de tráfico, "
        "causando errores 503 y afectando el checkout."
    )
    result = await process_text(text)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
