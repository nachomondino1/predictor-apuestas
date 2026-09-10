# Refactor — estado inicial, plan y registro de cambios

> Documento vivo. Acompaña a [`ARQUITECTURA.md`](./ARQUITECTURA.md) (que explica
> cómo funciona el código).
>
> Objetivo del refactor: **simplificar y optimizar** el código manteniendo la
> **robustez del scraping**. Trabajo **módulo por módulo**, con diffs revisables.
>
> Regla dura: **no** se eliminan ni modifican `sleep`/`SEC_WAIT_*` (delays),
> `options.add_argument(...)` (flags anti-bloqueo), headers ni `accept_cookies()`
> sin consultarlo antes.

---

## 1. Estado inicial del repo

Snapshot al commit base `6b5a83e8f` (rama `staging`), sesión iniciada 2026-09-10.

### 1.1 Tamaño y forma

- ~16.000 líneas de Python en 41 archivos versionados (`git ls-files '*.py'`).
- Archivos más grandes: `main.py` (1356), `p6_deployment/main_next_matches.py`
  (1260), `p3_data_preparation/construct_data.py` (1028),
  `p2_.../scraper_flashscore.py` (1017).
- Sin tests. Sin linter/formatter configurado en CI (aunque `black` está en
  `requirements.txt`).
- 3 archivos de dependencias: `requirements.txt` (~150 paquetes, incluye Django
  completo + TensorFlow + Streamlit), `requirements_updated.txt` (gitignored),
  `p6_deployment/requirements_mnm.txt` (~11, casi sin pinear).
- `venv/` presente en el working tree (ignorado, no versionado).

### 1.2 Cómo se ejecuta hoy

Ver [`ARQUITECTURA.md` §2–3](./ARQUITECTURA.md#2-los-dos-modos-de-ejecución).
Resumen:

- **CI:** solo `update_results.yml` (scrapeo de marcadores finales).
- **Manual/local:** entrenamiento (`main_train_models.py`), generación de
  predicciones (`collect_predictions.py` → `main_next_matches.py`), publicación
  (`publish_bets.py`), scrapeo histórico (`scraper_flashscore.py`,
  `scraper_sofifa.py`).

### 1.3 Hallazgos del análisis inicial

**Código muerto / roto:**

- `main.py :: main()` y su `if __name__ == "__main__"` — llamadas con firmas que
  ya no existen (`format_data` sin `df_match_odds`, `clean_data` con
  `df_teams_sofifa`, `construct_data` con `l_days`/`dif_con_against`,
  `treat_nan_values` inexistente, `DataPreparation(id_country, country)` sin
  `date`). Las **clases** de `main.py` sí se usan.
- `p2_.../scraper_whoscored.py` — `WhoScoredCrawler(headless, path)` llama
  `super().__init__(headless, path)` pero el constructor actual es
  `Crawler(headless, browser="Chrome")` → `path` entra como `browser`. Rutas
  absolutas de un repo viejo (`/Users/nachomondino/Documents/GitHub/predictor-apuestas/...`),
  `df_competencias.xlsx` (nombre viejo).
- `utils/set_up_logging_save.py` — no se importa; ruta `/home/runner/...`
  hardcodeada.
- `p4_modeling/utils_select_model/old/` — 4 archivos.
- `p4_modeling/nn.py` — solo alcanzable vía el `main.py::main` roto.
- `define_metrics.py` vs `define_metrics_v02.py` — dos versiones de
  `determine_metrics_by_model`.
- `.github/workflows/desuso/` — 6 workflows deshabilitados.

**Fragilidad del scraping:**

- ~116 XPaths acoplados a clases CSS de cada sitio, dispersos en 3 scrapers
  (`scraper_flashscore` 37, `scraper_sofifa` 31, `scraper_whoscored` 48). Fallo
  silencioso: `extract_tag` devuelve `None` → `NaN` → fila filtrada, solo
  `logger.warning` sueltos. No hay validación de esquema ruidosa post-scrape
  (aunque `p2_.../validate_data.py` tiene el germen: `validar_estructura` /
  `validar_datos`, huérfano).
- `extract_data` / `extract_missing_matches` / `extract_next_matches` en
  `scraper_flashscore.py` son ~90 % el mismo bucle `season → match` copiado 3
  veces.
- `inicialize_chrome_driver` — [resuelto en módulo 1]. Tenía: `try/except`
  anidado de 3 niveles, ruta personal hardcodeada, `raise ValueError` sin
  mensaje, tiers 2/3 tragándose la excepción real.

**Performance:**

- `construct_data.py` — `determine_mean_last_matches_*`, `h2h_by_date`,
  `determine_number_matches_last_days` hacen bucles por equipo × por partido con
  `.iterrows()` + slicing por máscara → O(n²), y se llaman dentro de un
  `itertools.product` anidado en 5 niveles en `main_train_models.py`. Es el cuello
  de botella del entrenamiento (el propio código imprime "Ritmo: X
  iteraciones/hora (ideal >60)").
- ~108 usos de `pd.concat([df, fila])` en loops en todo el repo.
- Todo el intercambio de datos en `.xlsx`.

**Sprawl de `prod`:**

- `format_data`, `clean_data`, `integrate_data`, `clean_post_integrate`,
  `clean_post_construct`, `scale_data` ramifican en
  `prod`/`fifa_not_released_yet`/`predict_missing`, y además
  `DataPreparationNew` reimplementa varios pasos. Difícil razonar sobre paridad
  train ↔ prod.

**Automatización / operación:**

- `main_next_matches.main()` tiene **11 llamadas a `input()`** → no automatizable.
- `create_action_update.py` genera un `.yml` por concatenación de strings.
- `update_results.yml` tiene ~25 crons.
- `publish_bets.py` coloca apuestas reales vía Selenium sin modo dry-run;
  `l_ids_to_sel` hardcodeado.

**Higiene:**

- ~20 rutas `/Users/nachomondino/...` hardcodeadas (dumps a `~/Desktop`, una
  lectura real en `format_data.py::__main__`, `sys.path.append` absoluto en
  `select_data.py`).
- `sys.path.append('.')` en ~20 archivos (falta empaquetado).
- `raise ValueError` sin mensaje (33×), `except:` desnudo (6×).
- `utils/set_up_logging.py` emite 4 logs de ejemplo en cada import.
- **Bugs concretos detectados:**
  - `format_data.py:256` — `df_match_odds_dtype.to_excel(...'df_match_player_dtype.xlsx')`
    pisa el archivo de dtypes de players.
  - `construct_data.py:655` — escribe `df_pre_constructed.xlsx` siempre, pero
    `df_preconstructed` queda indefinida si `with_historic=False` →
    `UnboundLocalError`.
  - `main_next_matches.py:997` — `return ValueError` (devuelve la clase).

---

## 2. Plan de cambios

Prioridades: **P1** = alto impacto / hacer primero · **P2** = valioso ·
**P3** = oportunista.
Riesgo y sensibilidad al scraping anotados por ítem.

### 2.1 `p2_data_understanding`

> ⚠️ **p2-1, p2-2, p2-8 CONGELADOS** (2026-09-10). La evaluación
> [`EVALUACION_VS_BET365.md`](./EVALUACION_VS_BET365.md) mostró que el modelo aún
> no le gana a bet365, así que no se justifica invertir en modernizar los
> scrapers todavía. Se retoman si/cuando el modelo muestre edge real.

| # | Prio | Cambio | Riesgo | Scraping sensible |
|---|---|---|---|---|
| p2-1 | ❄️ | Centralizar los XPaths de los 3 scrapers en un módulo de selectores único + validación ruidosa post-scrape (cablear `validate_data.py`). | Bajo (aditivo) | XPaths se **mueven**, no cambian |
| p2-2 | ❄️ | Deduplicar `extract_data` / `extract_missing_matches` / `extract_next_matches` en una función con flag `mode`. Acumular filas en lista → un `DataFrame`. | Medio, self-contained | No |
| p2-8 | ❄️ | Entrypoint único del scraper: dispatcher fino `python -m scraper <job>` (subcomandos `flashscore-history`, `sofifa-history`, `next-matches`, `results`, `missing`) que solo llama a las funciones que ya existen. Reemplaza el patrón "editá el `__main__`" + `sys.argv`/`ast.literal_eval`. Va junto con p2-2. | Bajo (aditivo) | No |
| p2-3 | P2 | `extract_data`: quitar `headless=False` hardcodeado y el `SEC_WAIT_MED = SEC_WAIT_MIN` mágico tras `n_season > 9` → parámetros explícitos (valores idénticos). | Bajo | Consultar: cambia *cómo* se setean las esperas |
| p2-4 | P2 | `check_if_season_already_extracted`: `os.path.exists` en vez de 3 lecturas Excel + `except:` desnudo. | Bajo | No |
| p2-5 | P2 | Mover `clean_id`, `extract_id_from_href`, `extract_name_from_href` a `parsers.py` + tests unitarios. | Nulo | No |
| p2-6 | P3 | Decidir destino de ~~`scraper_whoscored.py` → `archive/`~~ ✅ / `scraper_new_variables.py` / `validate_data.py` (pendiente). | Bajo | No |
| p2-7 | P3 | `concat_*.py` → funciones con parámetros. | Bajo | No |

### 2.2 `p3_data_preparation`

| # | Prio | Cambio | Riesgo | Scraping sensible |
|---|---|---|---|---|
| p3-1 | P1 | Vectorizar el hot path de `construct_data.py` (medias móviles / h2h / nº partidos) con `groupby` + rolling temporal / `merge_asof`. | Alto esfuerzo, alto payoff. Requiere golden-output test. | No |
| p3-2 | P1 | Domar el sprawl de `prod`: extraer core compartido, train/prod como wrappers finos. | Medio-alto. Test de paridad. | No |
| p3-3 | ~~P2~~ ✅ | Fix bug (era `main.py` format_data, no `format_data.py:256`): archivo de dtypes pisado. | Nulo | No |
| p3-4 | ~~P2~~ ✅ | Fix bug (era `main.py` construct_data): `df_preconstructed` indefinida si `with_historic=False`. | Bajo | No |
| p3-5 | ~~P2~~ ✅ | Quitado `clean_data.py` dump `~/Desktop/df_nan.xlsx` y `select_data.py` `sys.path` absoluto. | Bajo | No |
| p3-6 | P2 | `integrate_sofifa_to_flashscore`: blocking key + caché en el fuzzy matcher. | Medio. Test de calidad de mapeo. | No |
| p3-7 | P3 | `describe_data.scatter_plot`: cap de columnas / flag (originó los PNG de 29 MB). | Bajo | No |

### 2.3 `p4_modeling`

| # | Prio | Cambio | Riesgo | Scraping sensible |
|---|---|---|---|---|
| p4-1 | P1 | Consolidar `utils_select_model`: elegir `define_metrics_v02.py`, borrar la otra versión + `old/`. | Bajo (código muerto/dup) | No |
| p4-2 | P1 | ~~Decidir sobre `nn.py`: borrar + sacar `tensorflow`/`keras`/`scikeras` de requirements~~ ✅ | Bajo | No |
| p4-3 | ~~P2~~ ✅ | `asses_model.py` → `assess_model.py` + 8 importadores. | Bajo | No |
| p4-4 | P2 | Podar `asses_model.py`: mapear qué métricas usa de verdad `train_and_assess_models` / `main_select_model` y borrar el resto. | Medio | No |
| p4-5 | P2 | `build_model.py::space_params`: escalera if/elif → dict/config. | Bajo | No |
| p4-6 | P3 | Externalizar params de `betting_strategy` a config por país. | Bajo | No |

### 2.4 `p6_deployment`

| # | Prio | Cambio | Riesgo | Scraping sensible |
|---|---|---|---|---|
| p6-1 | P1 | Partir `main_next_matches.main` (~370 líneas) en etapas `run_missing/understanding/preparation/modeling` con I/O explícito. Reemplazar los 11 `input()` por flags. | Medio-alto, self-contained | No (usa `extract_next_matches` sin tocarlo) |
| p6-2 | P1 | Resolver la historia de workflows: ¿predicción diaria vuelve a CI o modo manual es intencional? Un workflow paramétrico único **o** documentar + borrar `create_action_update.py` + `desuso/`. | Bajo | No |
| p6-3 | P2 | `predict_models.py`: quitar escrituras a `~/Desktop/df_{country}.xlsx`. | Nulo | No |
| p6-4 | P2 | `collect_predictions.py`: cap/archivado de `historial_predicciones`. | Bajo | No |
| p6-5 | P2 | Normalizar contrato de retorno de `main_next_matches.main` (`return ValueError` → excepción o df vacío consistente). | Bajo | No |
| p6-6 | P3 | `publish_bets.py`: modo dry-run explícito; sacar `l_ids_to_sel` hardcodeado. | Bajo | Delays intactos |
| p6-7 | P3 | `update_results.yml`: consolidar crons con guardia interna. | Bajo | No toca el scraper |

### 2.5 Transversal / general

| # | Prio | Cambio | Riesgo |
|---|---|---|---|
| g-1 | P1 | Borrar código muerto: ~~`main.py::main` + `__main__`~~ ✅, ~~`set_up_logging_save.py`~~ ✅, ~~WhoScored → archive/~~ ✅, ~~`nn.py`~~ ✅. (`utils_select_model/old/` ya está en `.gitignore`, no está en el repo — nada que borrar.) | Bajo |
| g-2 | P1 | ~~Quitar los 4 `logger.x("Este es un mensaje…")` de `utils/set_up_logging.py`~~ ✅. | Nulo |
| g-3 | P2 | Empaquetado: `pyproject.toml` + `pip install -e .`, borrar todos los `sys.path.append`. | Medio |
| g-4 | P2 | Requirements: `requirements-scrape.txt` / `requirements-train.txt`, pinear `mnm`, dropear deps no usadas. | Bajo |
| g-5 | 🟡 | ~20 rutas `/Users/nachomondino/...`: **hechas las de código activo**; quedan las de `archive/` y 1 comentada. | Bajo |
| g-6 | P2 | Migrar intercambio de datos `.xlsx` → Parquet + acumulación por lista. | Medio (mucha superficie) |
| g-7 | P3 | `raise ValueError` sin mensaje, `except:` desnudo, typos en nombres públicos — oportunista por módulo. | Bajo |
| g-8 | P3 | `pytest` + un smoke test por fase. | Bajo |

### 2.6 Orden de ejecución sugerido

1. ~~**g-1, g-2**~~ ✅ — borrado de código muerto.
2. ~~**p4-2**~~ ✅ (nn.py) · **p4-1, p4-3** — consolidar `utils_select_model`, rename `asses_model`.
3. **Entender por qué el modelo no le gana a bet365** (nueva prioridad, ver
   `EVALUACION_VS_BET365.md`): validar que el backtest es out-of-sample, analizar
   el drift trimestral, revisar calibración y `betting_strategy.py`.
4. **p3-3, p3-4, p3-5** (quick wins / bugs) → **p3-2** (sprawl `prod`) →
   **p3-1** (vectorización de `construct_data`, con golden test) — habilita
   iterar modelos rápido.
5. **p6-1** → **p6-2** — partir `main_next_matches` + resolver workflows.
6. **g-3, g-4, g-6, g-8** — empaquetado, requirements, Parquet, tests.
7. **[congelado] p2-1 → p2-2 → p2-8** — modernización del scraping, solo si el
   modelo muestra edge real.

---

## 3. Registro de cambios

Formato: fecha · ítem del plan · qué se hizo · verificación · commit.

### 2026-09-10 — p3-3/p3-4/p3-5/g-5/p4-3: bugs, rutas personales y rename
- **Bugs (`main.py`):**
  - `format_data()` escribía `df_match_odds_dtype` al archivo de
    `df_match_player_dtype` (lo pisaba). → `df_match_odds_dtype.xlsx`.
  - `construct_data()`: `df_preconstructed.to_excel()` daba `UnboundLocalError`
    si `with_historic=False`. → guardado bajo `if with_historic`.
- **Rutas `/Users/nachomondino/…` en código activo (g-5):** `select_data.py`
  `sys.path` absoluto → `'.'`; quitados dumps de debug a `~/Desktop` en
  `clean_data.py` / `expected_result.py` / `define_metrics.py`; paths de
  `__main__` en `format_data.py` / `generate_test_design.py` → relativos;
  salidas de `assess_in_prod.py` / `evaluate_test_with_new_metrics.py` /
  `predict_models.py` → `data/…` en vez de `~/Desktop`. Queda solo 1 referencia
  comentada en `utils/expected_table.py`.
- **p4-3:** `asses_model.py` → `assess_model.py` (typo) + 8 importadores
  actualizados. De paso: comentario roto `# Fallaba el import de mainimport
  pandas as pd` corregido en 11 archivos.
- **Verificación:** `py_compile` en todos los tocados; `import assess_model` OK.
  Sin cambios de comportamiento en flujos activos.

### 2026-09-10 — Evaluación modelo vs bet365 (cambia prioridades)
- Nuevo: `analysis/evaluate_vs_bet365.py` — parsea el dump MySQL de producción
  (`data/backup_predictor_apuestas.sql`, no versionado) y calcula acierto 1X2,
  calibración (log-loss/Brier) y ROI del modelo vs bet365. Reproducible; escribe
  la parte de cifras de `docs/EVALUACION_VS_BET365.md` (preserva el TL;DR escrito
  a mano por encima de un marcador).
- **Resultado:** sobre 2079 partidos resueltos (jul-2024 → sep-2025) el modelo
  **no le gana a bet365**: acierto 49.0 % vs 53.7 %, probabilidades peor
  calibradas, ROI de la estrategia ≈ 0 %. Detalle en
  [`EVALUACION_VS_BET365.md`](./EVALUACION_VS_BET365.md).
- **Impacto en el plan:** p2-1 / p2-2 / p2-8 (modernización del scraping) quedan
  **congelados** hasta entender por qué el modelo va detrás del bookie. El foco
  pasa a calidad de modelo / calibración / drift.
- Sin cambios de comportamiento en el código del pipeline.

### 2026-09-10 — Documentación inicial
- Creados `docs/ARQUITECTURA.md` y `docs/REFACTOR.md` (este archivo).
- Sin cambios de comportamiento.

### 2026-09-10 — p2-6 / p4-2: archivado de WhoScored + borrado de `nn.py`
- `p2_data_understanding/collect_initial_data/scraper_whoscored.py` →
  `archive/scraper_whoscored.py` (con `archive/README.md` explicando el motivo).
- `p4_modeling/nn.py` **borrado** (`git rm`). Solo lo usaba el `main.py::main`
  removido; recuperable del historial.
- `requirements.txt`: quitados 13 paquetes del stack de TensorFlow
  (`tensorflow`, `keras`, `scikeras`, `tensorboard`, `tensorboard-data-server`,
  `astunparse`, `flatbuffers`, `gast`, `google-pasta`, `libclang`, `ml-dtypes`,
  `namex`, `opt_einsum`). 131 → 118 paquetes. Ningún otro archivo importa
  tensorflow/keras (verificado con `git grep`).
- **Verificación:** `git grep` confirma que nada más importa `nn` /
  `TrainNeuralNetwork` / `tensorflow` / `keras`.

### 2026-09-10 — g-1 / g-2: borrado de código muerto (parcial)
- `main.py`: eliminados `crear_variables()`, `main()` y el bloque
  `if __name__ == "__main__"` (firmas desactualizadas, nadie lo llamaba). El
  módulo queda como biblioteca de clases; se añadió una nota al pie apuntando a
  `main_train_models.py` y `p6_deployment/main_next_matches.py`. Imports sin
  podar (algunos los usan los métodos de clase; limpieza de imports queda para
  g-3).
- Eliminado `utils/set_up_logging_save.py` (sin referencias; ruta
  `/home/runner/...` hardcodeada).
- `utils/set_up_logging.py`: quitadas las 4 líneas `logger.x("Este es un
  mensaje…")` que se ejecutaban en cada import.
- **Pendiente de g-1:** borrar `p4_modeling/utils_select_model/old/` y decidir
  destino de `scraper_whoscored.py` + `p4_modeling/nn.py` (requiere tu OK).

**Verificación:** `py_compile` de `main.py`, `main_train_models.py`,
`main_next_matches.py`, `concat_mapeos.py` OK. `import main` + acceso a las 3
clases OK en `venv/`. El logger ya no imprime las 4 líneas de ejemplo.

### 2026-09-10 — Módulo 1: limpieza de `inicialize_chrome_driver`
Archivo: `p2_data_understanding/collect_initial_data/web_scraping_selenium.py`
(+ `.env.example`).

- `try/except` anidado de 3 niveles → lista ordenada de 3 estrategias
  (`_driver_latest_version`, `_driver_local_chrome_version`,
  `_driver_from_executable`) + un bucle que usa la primera que tenga éxito.
  **Mismo orden de fallback.**
- Ruta personal hardcodeada `/Users/nachomondino/Documents/chromedriver` →
  env var `CHROMEDRIVER_PATH` (documentada en `.env.example`). Si no está
  definida, esa 3ª estrategia se saltea.
- `raise ValueError` (lanzaba la clase, sin mensaje) →
  `raise RuntimeError("No se pudo inicializar el ChromeDriver por ninguna vía")`.
- Las estrategias 2 y 3 ya no se tragan la excepción real: cada fallo se loguea
  con su `e`.
- Docstring corregida (arg `path` inexistente eliminado).
- Añadidos `import os` + `from dotenv import load_dotenv` + `load_dotenv()` al
  módulo para poder leer `CHROMEDRIVER_PATH` al usar el scraper directamente.

**Intacto (sin cambios):** todo `options.add_argument(...)`, `--headless`,
`--remote-debugging-port=9222`, `enable-automation`, `get_chrome_version()`, y la
firma pública `inicialize_chrome_driver(headless)`.

**Verificación:** `python -m py_compile` OK. Smoke test con Chrome headless real
(en `venv/`): la estrategia 1 crea el driver, `driver.get` + `extract_tag` +
`click_boton` funcionan igual que antes.

**Pendiente (follow-ups anotados, requieren OK):**
- `--remote-debugging-port=9222` fijo → colisiona con 2 instancias; cambiar a `=0`.
- Typo `inicialize_chrome_driver` → `initialize_...` (1 caller interno).
