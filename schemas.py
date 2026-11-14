from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class NivelCriticidad(str, Enum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tecnologias: list[NonEmptyText] = Field(min_length=1)
    nivel_de_criticidad: NivelCriticidad
    resumen_tecnico: NonEmptyText
