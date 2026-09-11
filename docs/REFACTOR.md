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
  `p2_data_understanding/scraper_flashscore.py` (1017).
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
- `p2_data_understanding/scraper_whoscored.py` — `WhoScoredCrawler(headless, path)` llama
  `super().__init__(headless, path)` pero el constructor actual es
  `Crawler(headless, browser="Chrome")` → `path` entra como `browser`. Rutas
  absolutas de un repo viejo (`/Users/nachomondino/Documents/GitHub/predictor-apuestas/...`),
  `df_competencias.xlsx` (nombre viejo).
- `utils/set_up_logging_save.py` — no se importa; ruta `/home/runner/...`
  hardcodeada.
- `p4_modeling/model_selection/old/` — 4 archivos.
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

### 2.1 `p2_data_understanding` (hoy `src/predictor/data_understanding/`)

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

### 2.2 `p3_data_preparation` (hoy `src/predictor/data_preparation/`)

| # | Prio | Cambio | Riesgo | Scraping sensible |
|---|---|---|---|---|
| p3-9 | ~~P1~~ ✅ | **Vectorizar `integrate_player_data_in_match`** (`integrate_sofifa_to_flashscore.py`). Medido con el profiling del smoke: **85% del tiempo de entrenamiento**, no `construct_data`. **45.7 min → 0.85 seg** sobre los 15.886 partidos de england (~3200x). Ver detalle en el registro de cambios. | Bajo (test bit-a-bit contra la versión vieja) | No |
| p3-1 | P3 (bajó de prioridad) | Vectorizar `construct_data.py` (medias móviles / h2h / nº partidos). Medido: solo ~3% del tiempo total (1.7 min de 53.5 min) — mucho menos urgente de lo que se pensaba antes de medir. | Alto esfuerzo, payoff medio. Requiere golden-output test. | No |
| p3-2 | P1 | Domar el sprawl de `prod`: extraer core compartido, train/prod como wrappers finos. | Medio-alto. Test de paridad. | No |
| p3-3 | ~~P2~~ ✅ | Fix bug (era `main.py` format_data, no `format_data.py:256`): archivo de dtypes pisado. | Nulo | No |
| p3-4 | ~~P2~~ ✅ | Fix bug (era `main.py` construct_data): `df_preconstructed` indefinida si `with_historic=False`. | Bajo | No |
| p3-5 | ~~P2~~ ✅ | Quitado `clean_data.py` dump `~/Desktop/df_nan.xlsx` y `select_data.py` `sys.path` absoluto. | Bajo | No |
| p3-6 | P2 | `integrate_sofifa_to_flashscore`: blocking key + caché en el fuzzy matcher. | Medio. Test de calidad de mapeo. | No |
| p3-7 | ~~P1 (subió de prioridad)~~ ✅ | `describe_data.scatter_plot`: cap a 12 cols / 2000 filas, crea el dir, try/except. Pasó de "nice to have" a **bloqueante**: colgaba `comprehensive_search(verbose>=1)` con el `df_match` real (121 cols). También arreglé el bug de `stages.py::describe_data()` que pasaba `df_match` a los 4 llamados (por eso los 4 PNG borrados pesaban exactamente igual). | Bajo | No |

### 2.3 `p4_modeling` (hoy `src/predictor/modeling/`)

| # | Prio | Cambio | Riesgo | Scraping sensible |
|---|---|---|---|---|
| p4-7 | **P1 (endorsed por iter4)** | **`betting_strategy.py` → "sin ea".** La iter4 concluye: reducir la bs a stake plano, sacar `apply_strategy_by_result`, `define_hiperparameters(strategy=…)` y la selección de estrategia por test (duplicaba overfitting). Dejar solo las 2 salvedades que iter4 mantiene: `stake=0` en local (`result_to_bet==1`) y `stake=0` si `player_emergency_fill==1`. | Bajo | No |
| p4-1 | ~~P2~~ ✅ | **Decidido:** se borra el estudio retrospectivo de correlación por país (`define_metrics.py`, `define_metrics_v02.py`, `evaluate_test_with_new_metrics.py`). `main_select_model.py` ya elige el modelo a deployar por ROI real de test (+ `expected_error`) directamente — no hace falta la correlación. Se conserva `roi_in_time.py` (gráfico de ROI en el tiempo, para detectar cuándo un modelo se degrada y hay que reentrenar) y `assess_in_prod.py` (re-evalúa candidatos contra partidos reales recientes antes de elegir, ya integrado en `main_select_model.py` vía `assess=True`). | Bajo | No |
| p4-2 | P1 | ~~Decidir sobre `nn.py`: borrar + sacar `tensorflow`/`keras`/`scikeras` de requirements~~ ✅ | Bajo | No |
| p4-3 | ~~P2~~ ✅ | `asses_model.py` → `assess_model.py` + 8 importadores. | Bajo | No |
| p4-4 | P2 | Podar `assess_model.py`: mapear qué métricas usa de verdad `train_and_assess_models` / `main_select_model` y borrar el resto. Se hace **junto con p4-7** (al simplificar la bs caen métricas de ROI/estrategia). | Medio | No |
| p4-5 | P2 | `build_model.py::space_params`: escalera if/elif → dict/config. | Bajo | No |
| p4-6 | ~~P3~~ | ~~Externalizar params de `betting_strategy` a config por país~~ — obsoleto: p4-7 elimina esos params. | — | No |

### 2.4 `p6_deployment` (hoy `src/predictor/deployment/`)

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
| g-1 | P1 | Borrar código muerto: ~~`main.py::main` + `__main__`~~ ✅, ~~`set_up_logging_save.py`~~ ✅, ~~WhoScored → archive/~~ ✅, ~~`nn.py`~~ ✅. (`model_selection/old/` ya está en `.gitignore`, no está en el repo — nada que borrar.) | Bajo |
| g-2 | P1 | ~~Quitar los 4 `logger.x("Este es un mensaje…")` de `utils/set_up_logging.py`~~ ✅. | Nulo |
| g-3 | ~~P2~~ ✅ | `pyproject.toml` + `pip install -e .`; `sys.path.append('.')` borrado de 33 archivos; CI usa `PYTHONPATH=.`. | Medio |
| g-4 | P2 | Requirements: `requirements-scrape.txt` / `requirements-train.txt`, pinear `mnm`, dropear deps no usadas. | Bajo |
| g-5 | 🟡 | ~20 rutas `/Users/nachomondino/...`: **hechas las de código activo**; quedan las de `archive/` y 1 comentada. | Bajo |
| g-6 | ~~P1~~ ✅ | Migrar intercambio de datos `.xlsx` → Parquet, alcance "desde `clean_data` en adelante" (`utils/io.py`). Medido: **8.1 → 6.8 min** en el smoke de england. | Medio (mucha superficie) — mitigado con scope acotado + grep exhaustivo |
| g-7 | P3 | `raise ValueError` sin mensaje, `except:` desnudo, typos en nombres públicos — oportunista por módulo. | Bajo |
| g-8 | 🟡 | `pytest` + un smoke test por fase. **Hecho el esqueleto** (`tests/test_imports.py`, `tests/test_parsers.py`, 29 tests). Falta cobertura por fase. | Bajo |
| g-9 | ~~P1~~ ✅ | Historial de entrenamientos (`utils/training_log.py` → `data/_training_log.xlsx`): cada corrida de `comprehensive_search` (smoke o real) queda anotada con commit, país, duración, métricas de test y carpeta de modelos, para comparar corridas entre sí. | Nulo |
| g-10 | ~~P1~~ ✅ | Estructura de carpetas Nivel 1: todo el código bajo `src/predictor/{data_understanding,data_preparation,modeling,deployment,utils}` + `stages.py`. Pedido explícito del usuario ("una sola carpeta para scripts .py"). Ver §2.7. | Medio (mitigado: codemod anclado a imports, no toca `data/`) |

### 2.7 Estructura de archivos

**Diagnóstico:** el split por fases CRISP-DM (`p2→p3→p4→p6`) es defendible pero se
desvía de buenas prácticas: `main.py` mal nombrado, orquestación en 4 "mains" a 3
profundidades, `p6_deployment/` como cajón de sastre, prefijos `pN_` verbosos,
`utils/` grab-bag + un 2º `utils/` en p3, ~15 scripts sueltos como código de
librería, sin `src/` ni `tests/`.

**Dos frenos a renombrar los paquetes `pN_`** (contexto de las iteraciones, ver
ARQUITECTURA §1bis) — **resueltos en `g-10` (2026-09-11)** sin caer en ninguno
de los dos:
1. Los prefijos `pN_` mapean 1:1 con las fases CRISP-DM y con los 4 documentos de
   iteración → **se preservó la semántica** (`data_understanding`,
   `data_preparation`, `modeling`, `deployment`, no nombres genéricos), documentada
   en ARQUITECTURA §1.
2. El árbol `data/` (39 GB) espeja esos nombres, hardcodeado en cientos de
   f-strings → **no se tocó**: las rutas de datos son strings independientes
   del nombre de la carpeta de código, así que renombrar el código no obliga a
   tocar `data/`.

| Nivel | Qué | Riesgo | Estado |
|---|---|---|---|
| **0** | Renombrar `main.py`→`stages.py`; `scripts/` con los glue; limpiar `utils/`; esqueleto `tests/`. | Bajo | ✅ 2026-09-10 |
| **1** | Renombrar paquetes `pN_` por codemod (~68 imports) a `src/predictor/{data_understanding,data_preparation,modeling,deployment,utils}` + `stages.py`. Sin tocar `data/`. Commit dedicado + `pytest` + smoke end-to-end. | Medio | ✅ 2026-09-11 (`g-10`) |
| **2** | `[project.scripts]` + disolver `stages.DataPreparation` en clases por fase. Va con p3-2 y p6-1. | Alto | pendiente |

### 2.6 Orden de ejecución sugerido

Foco actual: **simplificar / lean**. El primer entrenamiento de prueba (england)
**ya se corrió y funcionó** (§2.8) — confirma que el pipeline de train sobrevivió
al refactor.

**Hecho** ✅: módulo 1 (chromedriver), g-1, g-2, g-3 (packaging), g-5 (parcial),
p2-6, p3-3/4/5, p3-7 (scatter_plot), p3-9 (vectorizo `integrate_data`), g-6
(Parquet), g-9 (historial de entrenamientos), p4-2 (nn+TF), p4-3 (rename),
estructura Nivel 0 + aplanado de carpetas, limpieza de peso (repo 52→16 GB,
`data/` 39→6 GB), smoke de entrenamiento england (53.5 → 8.1 → 6.8 min).

**Próximo:**
1. **p4-7 + p4-4** — `betting_strategy.py` → "sin ea" + podar `assess_model.py`.
2. **g-4** — separar requirements + pinear `mnm`.
3. **p6-1** — partir `main_next_matches.main` + sacar los 11 `input()`.
4. **p3-1** (vectorizar `construct_data`) → **p3-2** (sprawl `prod`). Con
   `integrate_data` y el I/O resueltos, `construct_data` (1.7 min, ahora la
   mayor parte del total) vuelve a ser el próximo cuello de botella real.
5. **p6-2** (workflows), **g-8** (más tests), **p4-1** (`define_metrics`,
   cuando digas cuál usás).
6. **[pausa]** comparación vs bet365 · **[pausa]** p2-1/2/8 (scrapers) · **[P3]**
   estructura Nivel 1.

### 2.8 Primer entrenamiento de prueba (england) — ✅ hecho, 2026-09-10

`scripts/smoke_train.py` corrió `comprehensive_search` para england con grid
mínimo (1 iteración, 1 modelo). **53.5 min, sin errores.** Detalle y métricas en
el registro de cambios más abajo. Bloqueador encontrado y resuelto en el camino:
`describe_data()` colgaba con `verbose>=1` (p3-7).

Para un entrenamiento real (grid completo de `define_params_space`, los 3
modelos) contá **horas por país** hasta que se resuelva p3-1.
---

## 3. Registro de cambios

Formato: fecha · ítem del plan · qué se hizo · verificación · commit.

### 2026-09-11 — Primer entrenamiento real (england) + p4-1: simplifico selección de modelo

**`scripts/real_train.py`** (nuevo): corre `comprehensive_search` con el grid
completo de `define_params_space` (128 combinaciones de datos × 3 modelos =
384 filas) para 1 país, en vez del grid mínimo de smoke. Resultado va a
`data/{country}/p4_modeling/{date}/df_iteration.xlsx`.

**Corrida en england:** 123.7 min, sin errores, `pytest` 34/34. Mejor combo
por f1_score: `LogisticRegression` (f1=55.6, accuracy=57.8, pero roi=-58.4).
Mejor por ROI real: `XGBClassifier` (roi=74.4, f1=54.4) — confirma en la
práctica lo que ya se sospechaba: la mejor métrica de test no siempre es la
que mejor rinde en plata. Promedio de f1_score por modelo: XGBoost 42.3 >
RandomForest 41.5 > LogisticRegression 41.3 (bastante parejos). Se anotó solo
en `data/_training_log.xlsx` vía el hook de `g-9`.

**p4-1 (decisión del usuario):** el propósito histórico de
`define_metrics.py`/`v02` (estudio retrospectivo de correlación por país
entre métricas de test y ROI de producción) era elegir qué modelo deployar
cada fin de semana y detectar cuándo reentrenar. Reviendo `main_select_model.py`
encontré que **ya no hace falta**: ese script elige el modelo a deployar
directo por ROI real de test (+ `expected_error`) — el ROI de test *es* la
métrica de negocio, correlacionarla contra sí misma no aporta. Se borran
`define_metrics.py`, `define_metrics_v02.py`, `define_metrics_info.txt`,
`evaluate_test_with_new_metrics.py` (grep exhaustivo: sin referencias fuera
de sí mismos). Se conservan intactos `main_select_model.py`, `assess_in_prod.py`
(re-evalúa candidatos contra partidos reales recientes antes de elegir) y
`roi_in_time.py` (gráfico de ROI en el tiempo — ya cubre "detectar cuándo un
modelo se degrada y reentrenar" sin agregar nada nuevo). `pytest` 34/34.

### 2026-09-11 — g-6: Excel → Parquet + g-9: historial de entrenamientos

**g-6.** Con `integrate_data` resuelto (p3-9), el I/O de Excel pasó a ser el
cuello de botella (~78% del tiempo). Reemplacé `pd.read_excel`/`to_excel` por
`pd.read_parquet`/`to_parquet` (`utils/io.py`, helper `read_df`/`write_df` que
mapea `algo.xlsx` → `algo.parquet`) en los archivos **100% internos del
pipeline de training, desde `clean_data` en adelante**: `stages.py`
(`clean_data`, `describe_integrate_data`, `clean_post_integrate`,
`construct_data`, `tag_string_data_to_integer`, `clean_post_construct`,
`select_data`, `clean_post_select`, `generate_test_design`),
`main_train_models.py` (lectura de los sofifa cleaned globales) y
`p3_data_preparation/concat_mapeos.py`.

**Alcance acotado a propósito** (decisión conjunta, no toqué esto):
- Quedan en `.xlsx` los datos crudos pre-`clean_data` (`format_data.py`) porque
  tienen columnas de tipo mixto a propósito (señal para el type-sniffing) —
  verifiqué con un chequeo real que desde `clean_data` en adelante **no hay
  columnas de tipo mixto**, lo que hace segura la conversión.
- Quedan en `.xlsx` las tablas maestras (`df_competencias`, `df_countries`,
  `df_best_models`) y las salidas finales (`predicciones`,
  `historial_predicciones`).
- Quedan en `.xlsx` los archivos que también toca producción
  (`p6_deployment/main_next_matches.py`): `df_integrated.xlsx`,
  `df_teams.xlsx`, `df_map_players_fs_so.xlsx`, y la familia con sufijo
  `df_constructed_*`/`df_etiquetas_*`/`df_ite_*`/`df_iteration`/
  `*_predicciones` (usada por `TrainingDataLoader` en producción o inspeccionada
  a mano en Excel según la iter4).
- **Sin doble escritura**: se reemplaza `.xlsx` por `.parquet`, no se mantienen
  los dos formatos.

**Verificación:** grep exhaustivo de cada patrón de archivo convertido en todo
el repo para confirmar que no quedó ningún lector/escritor fuera de lo tocado
(2 falsos positivos revisados a mano y descartados). Validación funcional
sobre datos reales cacheados (round-trip idéntico). `pytest` 34/34. Smoke
end-to-end (england): encontré y corregí un archivo compartido entre corridas
(`df_player_sofifa_cleaned`/`df_player_fifa_sofifa_cleaned`, mantenido por
`concat_mapeos.py`, no regenerado en cada training) que aún estaba en `.xlsx`
— migración puntual + borrado del `.xlsx` viejo.

**Resultado confirmado con `scripts/smoke_train.py`:** **8.1 → 6.8 min**
(~16%). Menos dramático que p3-9 porque el smoke solo hace I/O una vez por
etapa, pero el ahorro es proporcionalmente mayor en un grid con muchas
iteraciones (cada combo reescribe varios `.xlsx`).

**g-9 (pedido nuevo — "Objetivo 2: anotar entrenamientos"):** cada corrida de
`comprehensive_search`, sea smoke o real, ahora se anota en
`data/_training_log.xlsx` (`utils/training_log.py::log_run`, llamado
incondicionalmente al final de `comprehensive_search`, antes del `return`):
commit de git, país, fecha de iteración, cantidad de combos, modelos
probados, duración, carpeta de modelos guardados (los `.pkl` ya se guardaban;
esto no cambió), y un resumen de `f1_score`/`test_accuracy`/`roi` si están
disponibles. Queda en `.xlsx` (no Parquet) a propósito, para poder abrirlo a
mano. Se agregó `!data/_training_log.xlsx` a las excepciones del `.gitignore`
y se completaron a mano 2 filas históricas (baseline pre-p3-9 y post-p3-9
pre-g-6) para tener el historial completo desde el primer smoke.

**Verificación:** `pytest` 34/34, `py_compile` OK. Confirmado en vivo: la
corrida de smoke lanzada ya con el hook activo agregó su fila automáticamente
a `data/_training_log.xlsx` sin intervención manual.

### 2026-09-11 — p3-9: vectorizo `integrate_player_data_in_match` (85% → segundos)

Perfilé el smoke de entrenamiento (los `print(f"... en X minutos")` que ya tenía
cada fase) y medí, sobre england (15.886 partidos): `format_data` 0.0 min,
`clean_data` 0.1 min, **`integrate_data` 45.7 min (85% del total)**,
`construct_data` 1.7 min, resto ~0. La sospecha original (que `construct_data`
era el cuello de botella, siguiendo la queja de iter3) **era incorrecta** para
este dataset — quedó corregida en el plan (§2.6/§2.8 y aquí).

**Diagnóstico:** `integrate_player_data_in_match` (en
`integrate_sofifa_to_flashscore.py`) recorre 3 titularidades × 2 condiciones ×
15.886 partidos × hasta 11 columnas de jugador (~1M iteraciones), y en **cada
una** recastea (`.astype(str)`/`.astype(int)`) y escanea linealmente las
columnas completas de `df_map_fs_so` (26.742 filas) y `df_player_fifa_sofifa`
(144.589 filas) en vez de usar un índice — un lookup lineal repetido ~1M veces
sobre tablas de decenas/cientos de miles de filas.

**Reescritura:** arma los lookups (dict `id_player_fs→id_player_so`, dict
`id_player_so→height`, `MultiIndex (id_player, fifa_year)→stats`) **una sola
vez**, y resuelve cada columna de jugador con un join vectorizado
(`.map()`/`.reindex()`/`groupby`) sobre todos los partidos a la vez, en vez de
partido por partido. Mismo pipeline de fallback al FIFA anterior, mismos
umbrales `n_reg_min`, misma propagación de NaN en las métricas (pandas
`sum()`/`mean()` ignoran NaN por default; se "envenenan" a mano los grupos con
algún NaN para replicar el `sum()` de Python puro que usaba la versión vieja).

**Validación** (antes de tocar el módulo real): benchmark con los inputs
cacheados del smoke de ayer (`data/england/p3_data_preparation/2026-09-10/...`)
— justo el patrón de "guardar los df de cada etapa" que ya usás. Comparé
bit-a-bit contra la versión vieja en 3 muestras (300 secuenciales, 800 al azar,
3000 al azar) — **idéntico** en las 3. Agregué
`tests/test_integrate_sofifa_to_flashscore.py` (5 tests sintéticos: match
directo, fallback a FIFA anterior, NaN que propaga en una sola métrica, jugador
sin mapeo, partido por debajo de `n_reg_min`). `pytest` 34/34.

**Resultado:** `integrate_player_data_in_match` sobre los 15.886 partidos de
england: **45.7 min → 0.85 seg (~3200x)**.

**Confirmado end-to-end** con `scripts/smoke_train.py` (mismo grid mínimo):
**el entrenamiento completo pasó de 53.5 min a 8.1 min** (log: `Integracion de
datos en 0.0 minutos`). Nuevo desglose medido:

| Fase | Antes | Ahora |
|---|---|---|
| Carga inicial + I/O de `.xlsx` (format/clean, leer+escribir varios archivos de varios MB) | ~4 min | **~6.3 min (78% del nuevo total)** |
| `integrate_data` | 45.7 min | 0.0 min |
| `construct_data` | 1.7 min | 1.7 min (**21% del nuevo total**, sin tocar) |
| resto (select/modeling) | ~0 | ~0 |
| **Total** | **53.5 min** | **8.1 min** |

El cuello de botella cambió de nuevo: con `integrate` resuelto, ahora domina la
**lectura/escritura de Excel** (`p3-9` no tocó eso). Sube de prioridad **g-6**
(migrar el intercambio de datos a Parquet) — potencialmente el siguiente salto
grande. `construct_data` (`p3-1`) queda en un distante segundo lugar (21%, sin
subir de prioridad todavía).

### 2026-09-10 — Smoke de entrenamiento england + p3-7
- **`scripts/smoke_train.py`** (nuevo, reutilizable): corre `comprehensive_search`
  para 1 país con grid mínimo (1 iteración, 1 modelo). Primer intento con
  `verbose=1` se colgó — diagnostiqué que `describe_data()` hace `sns.pairplot`
  sobre el `df_match` completo (121 columnas) → subí `p3-7` de "nice to have" a
  bloqueante y lo arreglé (cap 12 cols / 2000 filas, crea el dir, try/except; de
  paso arreglé el bug de que los 4 `scatter_plot` de `describe_data()` pasaban
  siempre `df_match` en vez del df correcto).
- **Resultado del smoke (`verbose=0`, england, id_country=48):** ✅ corrió
  end-to-end sin errores. 53.5 min. `df_match` 34.697×121 → filtro fecha →
  5.212 filas → selección de features sobre 1.272 → test set de 99.
  `LogisticRegression` (GridSearchCV): `test_accuracy≈42%`, `f1_score
  train 44.8%→test 40.2%` (marcó posible overfitting; 1 combo arbitrario, no es
  tuning real). Confirma que **el pipeline de train funciona post-refactor**
  (renames, packaging, aplanado de carpetas, limpieza de `data/`).
- **Costo:** ~20 min de data prep una sola vez (`integrate_data` iteró 15.886
  partidos a ~15/s) + `construct_data` (el O(n²) de `p3-1`). El grid completo
  real (128 iter × 3 modelos) proyecta **~5-6 h por país** → confirma que
  **p3-1 (vectorizar `construct_data`) es la mejora de mayor impacto** para
  poder iterar modelos.
- **Verificación:** `pytest` 29/29, `py_compile` OK, test funcional sintético de
  `scatter_plot` (wide+grande: 5.1s vs colgado antes; <2 cols numéricas: skip
  sin crash).

### 2026-09-11 — Aplanado de carpetas (estructura, dentro de fase)
Sin renombrar los paquetes de fase `pN_` (mapean con CRISP-DM y con `data/`).
- `p2_data_understanding/collect_initial_data/*` → sube a `p2_data_understanding/`.
- `p6_deployment/automatize_predict/*` y `.../dispatch_event/*` → suben a
  `p6_deployment/`.
- `p4_modeling/utils_select_model/` → renombrado `p4_modeling/model_selection/`;
  borrado su `old/` (4 scripts, untracked). Borrado `utils/prueba.py` (untracked).
- Codemod de imports (13 `.py`) + `update_results.yml` (sparse-checkout +
  rutas de invocación) + `tests/`. `pip install -e .` refrescado.
- **−4 niveles de carpeta.** Estructura de código: `p2_data_understanding/`,
  `p3_data_preparation/`, `p4_modeling/` (+`model_selection/`), `p6_deployment/`,
  `utils/`, `scripts/`, `tests/`, `analysis/`, `docs/`.
- **Verificación:** `pytest` 29/29, `py_compile` 40/40, imports OK desde
  `cwd=/tmp`.
- ⚠️ El cambio de `update_results.yml` está en `claude-test`; el CI de `prod` usa
  la estructura vieja hasta que se mergee. **No cherry-pickear el yml sin el
  código** (van juntos).

### 2026-09-10 — Limpieza de peso (archivos y `data/`)
- **Tracked borrados** (recuperables del historial): `archive/`,
  `dispatch_event/config_webhook.py`, `scraper_new_variables.py`,
  `validate_data.py`, `update_predictions/` (`create_action_update.py` +
  `schedules.xlsx`). `.gitignore`: `.pytest_cache/`.
- **Local (no tracked):** borrado `__pycache__` del proyecto, 350 `.DS_Store`,
  `.github/workflows/desuso/`.
- **`data/` 39 GB → 6 GB** (−32 GB): borradas todas las carpetas `old/` /
  `data/_old/` / `data/_metrics/` y los `iteration_date` `< 2025-08-26` en
  `p2/old_updated`, `p3_data_preparation`, `p4_modeling`. Se conserva `2025-08-26`
  (el modelo activo según `df_best_models.xlsx`) y posteriores (08-29, 08-30,
  09-19, 09-21). `pytest` 29/29 tras la limpieza.
- **Nota:** `df_best_models.xlsx` apunta a `2025-08-26` para `usa` y `argentina`
  pero esas carpetas **nunca existieron** (inconsistencia pre-existente, no la
  causó esta limpieza). El pipeline de esos 2 países no está armado para ese
  modelo.
- **`venv/` recreado** (`python3.12` + `requirements.txt` sin TensorFlow): 2.1 GB
  → 1.0 GB, `pip` ya no está roto. Borrado `p6_deployment/venv_mnm/` (178 MB).
  `pytest` 29/29, imports OK desde cwd ajeno.
- **Total repo: ~52 GB → ~16 GB** (`.git` 8.9 + `venv` 1.0 + `data` 6.3 + código).
- **Pendiente `.git` (8.9 GB):** la bloat (commits viejos de `data/`,
  `env-model/`, `django_project/`) es ancestro de `staging`/`prod` → NO se puede
  reducir el `.git` local sin reescribir esas ramas (force-push coordinado). Si
  `staging`/`prod` no se tocan, el `.git` se queda en 8.9 GB.

### 2026-09-10 — Contexto de las 4 iteraciones CRISP-DM (docs externos)
Leídos los 4 `.docx` de `~/Downloads/doc_tiptopia/` (ene-2023 → jun-2025). Impacto
en el plan (sin cambios de código):
- Nuevo **p4-7 (P1, endorsed)**: la iter4 concluye que hay que reducir
  `betting_strategy.py` a "sin ea". `p4-6` queda obsoleto.
- **Nivel 1 de estructura baja a P3**: los `pN_` mapean con las fases CRISP-DM y
  con los docs; el árbol `data/` los espeja.
- `p4-1` deja de ser "borrar código muerto": `main_select_model` /
  `predict_models` / `define_metrics` / `roi_in_time` son el workflow semi-manual
  intencional de selección de modelo.
- Rationale documentado en `ARQUITECTURA.md` §1bis (`id_match` como índice, árbol
  `data/`, `data/all` + `id_country=-1`, umbrales de NaN, `fill_na`).

### 2026-09-10 — Estructura Nivel 0 + g-8 (esqueleto de tests)
- `main.py` → `stages.py` (es biblioteca de clases, no entrypoint). 3
  importadores + `py-modules` de `pyproject` actualizados.
- Nuevo `scripts/` con los glue sin importadores: `concat_all_countries`,
  `concat_dfs`, `concat_flashscore_data`, `concat_sofifa_data` (ex p2),
  `expected_table` (ex utils), `train_expected_result` (ex
  `p3_data_preparation/utils/expected_result.py` — el dir vacío se eliminó).
- `git rm utils/caracteres_especiales.txt` (sin uso).
- Nuevo `tests/`: `test_imports.py` (26 módulos de librería) + `test_parsers.py`
  (`clean_id` / `extract_id_from_href` / `extract_name_from_href`). **29 passed.**
  `[tool.pytest.ini_options]` en `pyproject`. `pytest` instalado en el venv.
- **Verificación:** `pytest` 29/29, `py_compile` OK, `import stages` OK.



### 2026-09-10 — g-3: empaquetado, fin de `sys.path.append('.')`
- Nuevo `pyproject.toml` (setuptools, namespace packages, `dependencies = []` —
  las deps siguen en `requirements*.txt`). Instalación: `pip install -e . --no-deps`.
- Quitado `import sys` + `sys.path.append('.')` de **33 archivos**. Los 3 que
  usan `sys.argv` / `sys.exit` conservan `import sys`.
- `.github/workflows/update_results.yml`: `PYTHONPATH: .` en el job +
  `pyproject.toml` agregado al `sparse-checkout` (replica el efecto en CI sin
  depender del editable install).
- `.gitignore`: `*.egg-info/`, `build/`, `dist/`. README actualizado.
- **Nota lateral:** `p3_data_preparation/utils/expected_result.py` tiene código a
  nivel de módulo (lee un `.xlsx` en el import) → falla al importarlo fuera de la
  raíz. Pre-existente, nadie lo importa; queda para limpieza futura.
- **Verificación:** los 38 módulos importan desde `cwd=/tmp` con el editable
  install; import chain de CI OK con `PYTHONPATH`; `py_compile` OK en 42 `.py`.

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
- `p2_data_understanding/scraper_whoscored.py` →
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
- **Pendiente de g-1:** borrar `p4_modeling/model_selection/old/` y decidir
  destino de `scraper_whoscored.py` + `p4_modeling/nn.py` (requiere tu OK).

**Verificación:** `py_compile` de `main.py`, `main_train_models.py`,
`main_next_matches.py`, `concat_mapeos.py` OK. `import main` + acceso a las 3
clases OK en `venv/`. El logger ya no imprime las 4 líneas de ejemplo.

### 2026-09-10 — Módulo 1: limpieza de `inicialize_chrome_driver`
Archivo: `p2_data_understanding/web_scraping_selenium.py`
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
