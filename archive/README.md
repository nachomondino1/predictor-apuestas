# Código archivado (fuera del pipeline activo)

Módulos que ya no se usan. No se mantienen ni se garantiza que funcionen.

| Archivo | Estado | Motivo | Fecha |
|---|---|---|---|
| `scraper_whoscored.py` | conservado aquí | Constructor incompatible con el `Crawler` actual (pasa `path` como `browser`), rutas absolutas de un repo viejo, `df_competencias.xlsx` con nombre viejo. Sin importadores. Posible fuente de respaldo a futuro. | 2026-09-10 |
| `p4_modeling/nn.py` | **borrado** (recuperable del historial git) | MLP con Keras. Solo alcanzable desde el `main.py::main` borrado; en `main_train_models.py` la red está comentada. Se quitó junto con tensorflow/keras/scikeras de `requirements.txt`. | 2026-09-10 |
