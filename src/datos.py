"""Carga del dataset Nursery.

El archivo original (data/raw/Datos.txt) no tiene encabezado, así que acá
le ponemos los nombres de las columnas según data/raw/Descripcion.txt.
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "raw" / "Datos.txt"

COLUMNAS = [
    "parents",   # ocupación de los padres
    "has_nurs",  # guardería actual del niño
    "form",      # forma de la familia
    "children",  # cantidad de hijos
    "housing",   # condiciones de vivienda
    "finance",   # situación financiera
    "social",    # condiciones sociales
    "health",    # condiciones de salud
    "clase",     # variable objetivo (NURSERY)
]


def cargar_datos(ruta=RUTA_DATOS) -> pd.DataFrame:
    """Devuelve el dataset completo como DataFrame (todas las columnas como texto)."""
    return pd.read_csv(ruta, header=None, names=COLUMNAS, dtype=str, skip_blank_lines=True)
