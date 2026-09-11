"""
Helpers de I/O para reemplazar pd.read_excel/to_excel por Parquet en los
archivos 100% internos del pipeline de entrenamiento (ver docs/REFACTOR.md,
ítem g-6). Excel es ~10-700x más lento que Parquet para leer/escribir los
DataFrames de este proyecto (decenas de miles de filas).

Las rutas siguen escribiéndose con extensión ".xlsx" en el código que llama
(para no tener que renombrar todo); estas funciones las mapean a ".parquet"
por debajo. El índice se preserva/restaura automáticamente (no hace falta
`index_col=0` al leer).

⚠️ Usar SOLO para archivos que no toque nadie más que el propio entrenamiento.
NO usar para: `df_integrated.xlsx`, `df_teams.xlsx`, `df_map_players_fs_so.xlsx`
(los lee también `p6_deployment/main_next_matches.py` en producción),
`df_constructed_*.xlsx` / `df_etiquetas_*.xlsx` con sufijo de hiperparámetros
(los lee `TrainingDataLoader` en producción), ni `df_ite_*.xlsx` /
`df_iteration.xlsx` / `*_predicciones.xlsx` (se inspeccionan a mano en Excel
para elegir modelo). Esos quedan en `.xlsx` a propósito.
"""
import pandas as pd


def _parquet_path(path_xlsx: str) -> str:
    if path_xlsx.endswith(".xlsx"):
        return path_xlsx[:-len(".xlsx")] + ".parquet"
    return path_xlsx + ".parquet"


def read_df(path_xlsx: str, **kwargs) -> pd.DataFrame:
    """Reemplazo de `pd.read_excel(path, index_col=0)` para archivos migrados
    a Parquet. No hace falta `index_col`: el índice viaja con el archivo."""
    return pd.read_parquet(_parquet_path(path_xlsx), **kwargs)


def write_df(df: pd.DataFrame, path_xlsx: str, index: bool = True, **kwargs) -> None:
    """Reemplazo de `df.to_excel(path, index=...)` para archivos migrados a
    Parquet."""
    df.to_parquet(_parquet_path(path_xlsx), index=index, **kwargs)
