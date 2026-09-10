# Arquitectura del proyecto

> Documento vivo. Describe **cómo funciona el código hoy** y su estructura.
> Se actualiza a medida que el refactor avanza. Para el plan y el registro de
> cambios ver [`REFACTOR.md`](./REFACTOR.md).
>
> Última actualización: 2026-09-10 · commit base `6b5a83e8f`

---

## 1. Qué es

Sistema de predicción de resultados de fútbol para apuestas. Pipeline de ciencia
de datos organizado por **fases estilo CRISP-DM**, cada una en su carpeta
numerada:

| Carpeta | Fase | Responsabilidad |
|---|---|---|
| `p2_data_understanding/` | Data understanding | Web scraping (Flashscore, Sofifa, WhoScored) y descripción de datos crudos |
| `p3_data_preparation/` | Data preparation | Formateo, limpieza, integración de fuentes, construcción de features, selección |
| `p4_modeling/` | Modeling | Diseño de test, entrenamiento, evaluación, estrategia de apuesta, selección de modelo |
| `p6_deployment/` | Deployment | Predicción de próximos partidos, scrapeo de resultados, publicación de apuestas |
| `utils/` | — | Logging, creación de directorios, helpers varios |
| `main.py` | — | **Biblioteca de clases** (`DataUnderstanding`, `DataPreparation`, `Modeling`) usada por los orquestadores |
| `main_train_models.py` | — | Orquestador de **entrenamiento** |
| `analysis/` | — | Análisis ad-hoc fuera del pipeline (p. ej. `evaluate_vs_bet365.py`) |
| `docs/` | — | Documentación viva (`ARQUITECTURA.md`, `REFACTOR.md`, `EVALUACION_VS_BET365.md`) |

> No hay carpeta `p1` ni `p5`; la numeración sigue CRISP-DM (p1 = business
> understanding, p5 = evaluation) aunque esas fases no tienen código propio.

---

## 2. Los dos modos de ejecución

El sistema tiene **dos flujos** que comparten las clases de `main.py` y los
módulos de `p3`/`p4`:

### 2.1 Entrenamiento (manual / local)

```
main_train_models.py  →  comprehensive_search()
```

- Recorre un espacio de hiperparámetros con `itertools.product` **anidado en 5
  niveles** (`clean_post_integrate` × `construct` × `select` × `clean_post_select`
  × `modeling`).
- Para cada combinación: prepara datos → genera test design → entrena una lista
  de modelos (`LogisticRegression`, `XGBClassifier`, `RandomForestClassifier`) →
  evalúa en test con métricas de ROI.
- Escribe una fila por modelo entrenado en `df_ite_train.xlsx` / `df_ite_test.xlsx`
  / `df_params_ite.xlsx` bajo `data/{country}/p4_modeling/{date}/`.
- La selección del "mejor modelo" se hace después con `p4_modeling/main_select_model.py`
  y `p4_modeling/utils_select_model/`.

### 2.2 Producción / predicción (manual / local hoy)

```
p6_deployment/automatize_predict/collect_predictions.py
   →  p6_deployment/main_next_matches.py :: main(d_run, id_country, ...)
```

- `collect_predictions.py` itera países, lee de `data/df_best_models.xlsx` la
  `iteration_date` del modelo elegido, llama a `main_next_matches.main()` y
  acumula en `data/predicciones.xlsx` + `data/historial_predicciones.xlsx`.
- `main_next_matches.main()` ejecuta, según los flags de `d_run`
  (`run_missing`, `data_unders`, `data_prep`, `modeling`):
  1. **MISSING DATA** — scrapea partidos ya jugados que aún no estaban en el
     dataset de entrenamiento y los integra.
  2. **DATA UNDERSTANDING** — scrapea los próximos partidos (`extract_next_matches`).
  3. **DATA PREPARATION** — replica el pipeline de `p3` sobre los próximos
     partidos usando los hiperparámetros y artefactos (`scaler`, `df_etiquetas`)
     guardados en el entrenamiento.
  4. **MODELING** — carga el `.pkl` del modelo, predice probabilidades, aplica
     la estrategia de apuesta (`betting_strategy`) y devuelve `df` con stakes.
- Contiene **11 llamadas a `input()`** → hoy requiere un humano en la terminal.

### 2.3 Actualización de resultados (CI)

```
.github/workflows/update_results.yml  (cron)
   →  p6_deployment/automatize_predict/update_results.py 1
   →  p6_deployment/automatize_predict/dispatch_event/dispatch_event.py
```

- **Único flujo que corre en GitHub Actions.** ~25 crons (fines de semana y
  noches).
- Scrapea de Flashscore el marcador final de los partidos de los últimos `n_days`
  que están en `historial_predicciones.xlsx`, calcula `result` / `acerte` /
  `bet_yield`, hace `git push` a la rama `prod` y dispara vía
  `repository_dispatch` el repo secundario (`mondineta/landing`, presumiblemente
  VPS + Django + MySQL).

---

## 3. Entry points (`if __name__ == "__main__"`)

| Archivo | Rol | ¿Se usa? |
|---|---|---|
| `main_train_models.py` | Entrenamiento (orquestador real) | **Sí**, manual |
| `p6_deployment/automatize_predict/collect_predictions.py` | Predicción multi-país | **Sí**, manual |
| `p6_deployment/main_next_matches.py` | Predicción de un país (también librería) | **Sí**, manual / debug |
| `p6_deployment/automatize_predict/update_results.py` | Scrapeo de resultados | **Sí**, CI |
| `p6_deployment/publish_bets.py` | Auto-bet en stakehunters.com vía Selenium | **Sí**, manual |
| `p6_deployment/predict_models.py` | Comparación de N modelos candidatos | Ocasional, manual |
| `p2_data_understanding/collect_initial_data/scraper_flashscore.py` | Scrapeo histórico Flashscore | **Sí**, manual (bootstrap) |
| `p2_data_understanding/collect_initial_data/scraper_sofifa.py` | Scrapeo histórico Sofifa | **Sí**, manual (bootstrap) |
| `main.py` | ~~`main()` + `__main__`~~ | Removidos en el refactor. Solo quedan las **clases** (biblioteca). |
| `archive/scraper_whoscored.py` | Scrapeo WhoScored | **NO** — roto/legacy, movido a `archive/` |
| `p2_.../scraper_new_variables.py`, `validate_data.py`, `concat_*.py` | Utilitarios one-shot | Esporádico, no cableados |
| El resto de `p3_*/`, `p4_*/` con `__main__` | Bloques de prueba / scripts sueltos | Esporádico |

---

## 4. Flujo de datos end-to-end

```
                    ┌─────────────── FUENTES (web) ───────────────┐
                    │  Flashscore            Sofifa               │
                    │  · partidos            · ratings jugadores  │
                    │  · stats               · valor de mercado   │
                    │  · odds (Bet365)       · datos de equipos    │
                    │  · alineaciones                              │
                    └──────────┬─────────────────┬────────────────┘
                               │ Selenium        │ Selenium
                               ▼                 ▼
   p2   scraper_flashscore.py           scraper_sofifa.py
        → df_match, df_match_player,     → df_player_sofifa (atemporal),
          df_match_odds                    df_player_fifa_sofifa (por FIFA/fecha),
                                           df_teams_sofifa
                               │                 │
                               └────────┬────────┘
                                        ▼
   p3   format_data     → tipos correctos, %s, goles, capacidad
        clean_data      → texto (lower/acentos/stopwords), outliers, filtro por fecha
        integrate_data  → fuzzy match jugadores Sofifa↔Flashscore (df_map_players_fs_so)
                          + join de ratings/valor a cada partido  →  df_integrated
        clean_post_integrate → drop columnas constantes/ruido, filtro filas
        construct_data  → variables derivadas (SOG2S, PPDA, ELO, cards, clean_sheet…)
                          + históricas: medias móviles por equipo en últimos N días,
                            h2h, nº de partidos recientes  →  df_constructed
        tag_string_data_to_integer → codifica categóricas (df_etiquetas)
        clean_post_construct → drop data-leakage + columnas sin construir
        select_data     → drop correlacionadas (thr_corr) + feature selection (thr_fs)
        clean_post_select → fillna + StandardScaler (scaler.pkl)
                                        ▼
   p4   generate_test_design → split train/val/test por índice temporal + balanceo
        build_model     → GridSearch/Bayes de hiperparámetros, scorers custom
        assess_model    → métricas básicas + calculate_metrics
        assess_model_with_roi + betting_strategy → stakes, ROI, Kelly
        → modelo.pkl + df_ite_train/test/params
        main_select_model + utils_select_model → elige "mejor modelo" por país
                                        ▼
   p6   main_next_matches.main() reusa p3+p4 sobre PRÓXIMOS partidos
        → data/{country}/p6_deployment/predicciones.xlsx
        collect_predictions.py → data/predicciones.xlsx + historial_predicciones.xlsx
        publish_bets.py → carga apuestas en stakehunters.com
        update_results.py → agrega goles reales + acerte + yield al historial
        dispatch_event.py → notifica al repo landing (VPS/DB)
```

---

## 5. Componentes clave por fase

### 5.1 `p2_data_understanding`

- **`collect_initial_data/web_scraping_selenium.py`** — clase base `Crawler`.
  Encapsula:
  - Init del WebDriver. `inicialize_chrome_driver()` intenta **3 estrategias en
    orden** (última versión vía `webdriver-manager` → versión del Chrome local →
    ejecutable en `CHROMEDRIVER_PATH`). También hay init de Firefox y Safari.
  - `extract_tag(xpath, ...)` / `extract_tags(xpath, ...)` — wrappers sobre
    `WebDriverWait` + `expected_conditions`, con reintentos ante
    `StaleElementReferenceException` y `xpath_alt` de fallback. Devuelven `None` /
    `[]` en caso de fallo (**no lanzan**).
  - `click_boton(tag)` — click normal con fallback a `execute_script`.
  - Helpers de cookies en shadow DOM / iframe, `login_website`, `select_option`.
  - **Config anti-bloqueo / esperas** (NO tocar sin consultar): flags en
    `options.add_argument(...)`, `SEC_WAIT_MIN/MED/MAX` por subclase, `sleep`s con
    jitter (`random.uniform`), `accept_cookies()`.
- **`scraper_flashscore.py`** — `FlashscoreCrawler(Crawler)` con ~37 XPaths.
  Métodos por bloque de la ficha del partido (`extract_match_information`,
  `extract_teams`, `extract_result`, `extract_stats`, `extract_odds`,
  `extract_lineups`, `extract_coaches`, `extract_bajas_pre_partido`…).
  Funciones a nivel módulo:
  - `extract_data(...)` — scrapeo histórico completo (todas las temporadas).
  - `extract_missing_matches(...)` — solo partidos faltantes.
  - `extract_next_matches(...)` — próximos partidos (alineaciones previstas, odds).
  - Los tres comparten un bucle `season → match` casi idéntico (candidato a
    deduplicar).
  - Puras: `clean_id`, `extract_id_from_href`, `extract_name_from_href` (regex).
- **`scraper_sofifa.py`** — `SofifaCrawler(Crawler)`. Recorre la paginación de
  FIFAs y fechas de actualización; `extract_players`, `extract_teams`.
- **`update_sofifa_data.py`** — actualización incremental de Sofifa (usado por
  `main_train_models`).
- **`describe_data.py`** — `getting_to_know_data`, `verificar_unicidad_registros`,
  `scatter_plot` (genera los PNG grandes de `images/`).
- **Legacy / one-shot:** `archive/scraper_whoscored.py` (constructor roto,
  movido a `archive/`), `scraper_new_variables.py`, `validate_data.py` (tiene un
  validador de esquema útil pero huérfano), `concat_*.py`.

### 5.2 `p3_data_preparation`

Todo se orquesta desde `main.py :: DataPreparation` (y su subclase
`DataPreparationNew` en `main_next_matches.py`).

- **`format_data.py`** — conversión de tipos (`convert_*_to_int/float`),
  `format_percentage_columns`, `rename_and_merge_columns` (unifica nombres de
  stats nuevas vs viejas), `convert_columns_to_int` (etiquetado categórico),
  `format_df_*` (verificación de esquema por dataframe), `map_teams`.
- **`clean_data.py`** — `TextPreparation` (lower, acentos, stopwords, stemming),
  `prepare_text_columns`, `clean_teams_names`, familia `delete_*_nan` /
  `fill_nan_values` / `fillna_with_mean_in_last_matches`, `replace_infinite`,
  `corregir_goals`.
- **`integrate_sofifa_to_flashscore.py`** — el matching de fuentes:
  - `create_df_teams/coaches/stadiums/player` (extrae entidades únicas de `df_match`).
  - `match_dataframes_by_str_column` + `calculate_coincidence` — fuzzy match por
    similitud de strings (O(n·m), sin blocking).
  - `map_players(...)` → `df_map_players_fs_so.xlsx` (persistido; en prod se
    **reutiliza** en vez de re-mapear).
  - `integrate_player_data_in_match(...)` → añade columnas de rating/valor/edad
    agregadas por equipo a cada partido.
- **`construct_data.py`** — features:
  - Derivadas puntuales: `determine_result/points`, `determine_expected_result`,
    `assign_elo_before_match`, `construct_sum_columns`, `construct_percentaje_column`,
    columnas `KGP`, `PPDA`, `cards`, `clean_sheet`, `defensive_efficiency`…
  - **Históricas (hot path de performance):** `h2h_by_date`,
    `determine_number_matches_last_days`, `determine_mean_last_matches_home_away`,
    `determine_mean_last_matches_difference`. Bucles Python por equipo × por
    partido con `.iterrows()` + slicing por máscara → **O(n²)**, y se llaman
    dentro del `product` de 5 niveles del entrenamiento.
  - `calculate_dif_col_players` — diferencias home−away de variables de jugadores.
- **`select_data.py`** — `delete_correlated_columns` (triangular superior de la
  matriz de correlación), clase `FeatureSelection` (ANOVA, RF, RFE, Lasso →
  `sum_and_normalize_importances`), `select_best_features`,
  `determine_country_competitions` (mapea `id_country` → listas de competiciones),
  `select_league_matches`.
- **`concat_mapeos.py`** — consolida los `df_map_players_fs_so` de todos los
  países en `data/data_preparation/` (para cuando el FIFA nuevo aún no salió).

### 5.3 `p4_modeling`

- **`generate_test_design.py`** — `balance_dataset` (under/over/SMOTE),
  `separate_train_val_and_test`, `n_rows_to_test`.
- **`build_model.py`** — `select_best_hiperparameters` (Grid o `BayesSearchCV`),
  `space_params` (escalera if/elif de 165 líneas con el espacio por modelo),
  scorers custom (`combined_f1_logloss`, `custom_refit`), `manual_cross_validation`.
- **`asses_model.py`** *(sic — typo por `assess`)* — ~15 funciones de métricas:
  `calculate_metrics`, `confusion_matrix`, `calculate_roi` / `calculate_reality_roi`
  / `determine_roi`, `calculate_yield`, `calculate_bookie_metrics`,
  `calculate_result_probabilities_by_bookmaker`, `calculate_gp_by_result`,
  `calculate_combined_metric`, `normalize_column`…
- **`betting_strategy.py`** — clase `BettingStrategy`: `determine_result_to_bet`
  (umbral de probabilidad), `determine_stake_to_bet` (relación lineal / Kelly),
  `stake_reduction` / `cap_stake` / `normalize_stake`, `determine_winning_bets`,
  `calculate_roi_in_combination`, `apply_strategy` / `apply_strategy_by_result`.
- ~~**`nn.py`** — `TrainNeuralNetwork` (MLP Keras)~~ — **borrado** en el refactor
  (junto con tensorflow/keras/scikeras de `requirements.txt`). Recuperable del
  historial git si se quiere volver a probar redes neuronales.
- **`main_select_model.py`** + **`utils_select_model/`** — a partir de
  `df_iteration.xlsx` filtran y rankean modelos:
  `filter_models_by_metric/distribution`, `define_metrics.py` **y**
  `define_metrics_v02.py` (dos versiones), `assess_in_prod.py`, `roi_in_time.py`,
  `evaluate_test_with_new_metrics.py`. Carpeta `old/` con 4 archivos muertos.

### 5.4 `p6_deployment`

- **`main_next_matches.py`** — el pipeline de producción. Clases:
  `DataUnderstandingNew`, `DataPreparationNew(DataPreparation)`,
  `TrainingDataLoader` (carga hiperparámetros/artefactos del entrenamiento),
  `MissingData` (lee/concatena datasets old + missing). Función `main()` de ~370
  líneas.
- **`automatize_predict/collect_predictions.py`** — loop de países + acumulación
  en los xlsx de `data/`.
- **`automatize_predict/update_results.py`** — scrapeo de marcadores finales +
  cálculo de acierto/yield.
- **`automatize_predict/dispatch_event/`** — `dispatch_event.py` (POST a la API de
  GitHub para `repository_dispatch`), `config_webhook.py`.
- **`automatize_predict/update_predictions/create_action_update.py`** — genera un
  workflow `.yml` por **concatenación de strings** a partir de `schedules.xlsx`.
- **`publish_bets.py`** — `StakeHunterCrawler(Crawler)`: login y carga de apuestas
  reales en stakehunters.com vía Selenium.
- **`predict_models.py`** — corre `main_next_matches.main()` para los N mejores
  modelos y cuenta coincidencias de predicción.

---

## 6. Persistencia

- **Formato:** todo en **`.xlsx`** (`openpyxl`). Lectura/escritura con
  `pd.read_excel` / `df.to_excel`. Acumulación frecuente con
  `pd.concat([df, fila], axis=0)` dentro de loops.
- **Layout:** `data/{country}/{fase}/...`, con subcarpetas por `iteration_date`
  en `p3`/`p4` y sufijos `data_seg` / `per_season` / `per_competition` para
  guardados intermedios "por seguridad".
- **Versionado (`.gitignore`):** `data/*` está ignorado **salvo** 5 archivos
  clave: `df_best_models.xlsx`, `df_countries.xlsx`, `df_competencies.xlsx`,
  `historial_predicciones.xlsx`, `predicciones.xlsx`. También se ignoran `venv/`,
  `.env`, `images/`, `*.csv`, `*.pkl`, `data_seg/`, `old/`, `desuso/`,
  `descarte/`.
- **`df_countries.xlsx` / `df_competencies.xlsx`** — tablas maestras: id ↔ nombre
  de país, y por competición su nombre en Flashscore/Sofifa, `is_cup`,
  `is_public`.
- **`data/backup_predictor_apuestas.sql`** (no versionado) — dump MySQL de la DB
  de producción (Django + tabla `historial_predicciones` con resultados y
  `acerte`/`bet_yield`). Es la fuente de verdad para la evaluación retrospectiva;
  el `historial_predicciones.xlsx` de la rama `staging` NO tiene resultados
  cargados (eso se escribe en `prod`).
- **Salida final:** `predicciones.xlsx` (index = `id_match`, importante para
  MySQL) → repo `landing` vía dispatch.

---

## 7. Automatización (GitHub Actions)

- **Activo:** `.github/workflows/update_results.yml` — cron, `sparse-checkout` de
  un subconjunto de carpetas de la rama **`prod`**, Python 3.12, instala
  `p6_deployment/requirements_mnm.txt`, corre `update_results.py`, hace `git push`
  a `prod` y dispatch.
- **Deshabilitados:** `.github/workflows/desuso/` — 6 workflows
  (`get_predictions.yml`, `update_predictions.yml`, `create_update_predictions.yml`,
  `test_*.yml`) que **antes** automatizaban la generación de predicciones.
- Ramas: `staging` (main del repo), `prod` (deploy), `claude-test` (trabajo
  actual).

---

## 8. Configuración

- **`.env`** (no versionado; template en `.env.example`):
  - `ENVIRONMENT` = `dev` | `prod` — cambia si los parámetros vienen hardcodeados
    o de `sys.argv`.
  - `BASE_DIR_LOCAL` — raíz local.
  - `GITHUB_TOKEN` — para el dispatch.
  - `USER_SH` / `PASS_SH` — credenciales de stakehunters.com (`publish_bets.py`).
  - `CHROMEDRIVER_PATH` — *(añadido en el refactor, módulo 1)* ruta al ejecutable
    de chromedriver como último fallback del init del driver.
- **Imports:** el repo se instala como paquete con `pip install -e . --no-deps`
  (`pyproject.toml`, namespace packages). Eso reemplaza el viejo
  `sys.path.append('.')` que había al inicio de ~33 archivos. Alternativa sin
  instalar: `PYTHONPATH=.` (lo que usa la GitHub Action).
- **`utils/set_up_logging.py`** — logger con formato de colores por nivel
  (CRITICAL en verde = "éxito"). Emite 4 líneas de ejemplo en cada import
  (pendiente de limpiar). Existe también `set_up_logging_save.py` (muerto).

---

## 9. Glosario de DataFrames

| Nombre | Unidad | Origen | Contenido |
|---|---|---|---|
| `df_match` | 1 fila = 1 partido (index `id_match`) | Flashscore | fecha, equipos, goles, stats, árbitro, cancha, `id_country`, `id_competition`, `season` |
| `df_match_player` | 1 fila = 1 partido | Flashscore | columnas `id_player_{titular/sub/miss}_{home/away}_{n}` y sus nombres |
| `df_match_odds` | 1 fila = 1 partido | Flashscore | `odds_home` / `odds_draw` / `odds_away` (Bet365) |
| `df_player_sofifa` | 1 fila = 1 jugador (index `id_player`) | Sofifa | datos **atemporales**: nombre, altura, pie hábil, nacionalidad |
| `df_player_fifa_sofifa` | 1 fila = jugador × FIFA × fecha | Sofifa | datos **temporales**: overall, potential, valor, sueldo, edad, reputación |
| `df_teams_sofifa` | 1 fila = 1 equipo | Sofifa | prestigio, estadio, rival clásico |
| `df_map_players_fs_so` | 1 fila = 1 jugador | integración | mapeo `id_player` Flashscore ↔ Sofifa |
| `df_integrated` | 1 fila = 1 partido | p3 | `df_match` + agregados de jugadores + ELO |
| `df_constructed` | 1 fila = 1 partido | p3 | `df_integrated` + features derivadas + históricas |
| `df_etiquetas` | 1 fila = valor categórico | p3 | mapeo string → int para variables categóricas |
| `df_ite_train` / `df_ite_test` / `df_params_ite` | 1 fila = 1 modelo entrenado | p4 | hiperparámetros + métricas por combinación |
| `predicciones.xlsx` / `historial_predicciones.xlsx` | 1 fila = 1 partido futuro/pasado | p6 | probas, `result_to_bet`, `stake_to_bet`, y (histórico) `goals_*`, `acerte`, `bet_yield` |
