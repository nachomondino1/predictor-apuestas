# Contexto del proyecto (leeme primero)

Predictor de resultados 1X2 de fútbol + estrategia de apuestas. Pipeline CRISP-DM
en 4 fases, 5 países (england, france, germany, italy, spain), entrenamiento
manual local y predicción semanal.

**Idioma del proyecto: español.** Código, docs, commits y conversación en español.

## Dónde está cada cosa

| Necesito… | Leer |
|---|---|
| **Dónde estamos hoy, qué sigue, qué decisiones están abiertas** | [`docs/ESTADO.md`](docs/ESTADO.md) ← **empezar acá** |
| Cómo correr un experimento y comparar corridas | [`docs/EXPERIMENTOS.md`](docs/EXPERIMENTOS.md) |
| Cómo está organizado el código y por qué | [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md) |
| Backlog del refactor + historial de cambios | [`docs/REFACTOR.md`](docs/REFACTOR.md) |
| Cuán lejos está el modelo de bet365 | [`docs/EVALUACION_VS_BET365.md`](docs/EVALUACION_VS_BET365.md) |
| Instalación / setup | [`README.md`](README.md) |

`ESTADO.md` es el único doc que describe **el presente**; `REFACTOR.md` es
historia (append-only) y `ARQUITECTURA.md` es estructura. Si algo del presente
cambia, se actualiza `ESTADO.md`.

## Reglas que no se negocian

1. **Scraping: no tocar sin consultar.** No romper la lógica actual ni eliminar
   tiempos de espera (delays), cabeceras (headers) o mecanismos anti-bloqueo sin
   preguntar primero. Son la diferencia entre tener datos y estar bloqueado.
2. **Código sin uso se borra.** Nada de `archive/`, `old/`, `_v2`, ni comentar y
   dejar. Si no se usa, `git rm`; el historial de git es el archivo.
3. **`p2-1`, `p2-2`, `p2-8` están congelados** (modernización de scrapers). Se
   retoman si/cuando el modelo muestre edge real sobre el bookie.
4. **Docs vivos.** Todo cambio de comportamiento se registra en `REFACTOR.md`
   (entrada con fecha) y, si cambia el presente, en `ESTADO.md`. No es opcional:
   es lo único que hace que la sesión siguiente no arranque de cero.
5. **Un cambio a la vez cuando se mide.** Ver `EXPERIMENTOS.md`: si se meten dos
   cambios juntos, la métrica deja de decir cuál sirvió.

## Cómo correr

```bash
source venv/bin/activate          # python3.12
pip install -e . --no-deps        # una vez: habilita `from predictor...`
```

```bash
pytest -q                                      # 56 tests, < 1 min
python scripts/smoke_train.py 48 2026-09-11    # smoke 1 país, grid mínimo (~5 min con caché)
python scripts/real_train.py 48 2026-09-11     # grid completo 128×3 (horas)
```

El **2º argumento (`iteration_date`) es obligatorio en la práctica**: con fecha
nueva el pipeline rearma el mapeo de sofifa y el formateo desde `.xlsx` crudos y
tarda **3-8 h por país**; con una fecha ya usada reusa esa caché y tarda minutos.
Para comparar corridas hay que pasar **siempre la misma fecha**.

Cada corrida se anota sola en `data/_shared/logs/_training_log.xlsx` (commit,
país, duración, métricas, carpeta de modelos).

## Convenciones de código

- Librería en `src/predictor/{data_understanding,data_preparation,modeling,deployment,utils}`
  + `stages.py`. Glue ejecutable en `scripts/`. Nada de código de librería en la raíz.
- Imports **absolutos** (`from predictor.utils.io import ...`). **Nunca**
  `sys.path.append`; el paquete está instalado con `pip install -e .`.
- Sin `__init__.py` (namespace packages).
- Toda aleatoriedad sale de `predictor.config.SEED`. Un modelo nuevo se instancia
  con `random_state=SEED`. No hardcodear `42`.
- I/O de datos intermedios por `predictor.utils.io` (Parquet). Los `.xlsx` quedan
  solo para crudos de scraping y archivos chicos para mirar a mano.
- `tests/` espeja `src/`; test de regresión obligatorio cuando se vectoriza o se
  reemplaza una implementación.

## Trampas conocidas

- **`verbose >= 1` en `comprehensive_search` cuelga la corrida**: dispara
  `describe_data()` → `sns.pairplot` sobre un df de 121 columnas. Usar `verbose=0`.
- **`data/` (~7 GB) no está en git** salvo 6 archivos. No se puede regenerar sin
  re-scrapear: **nunca** borrar o mover a ciegas.
- `comprehensive_search` usa `id_country` como **global del módulo**, no como
  parámetro (`mtm.id_country = 48` antes de llamar).
- El workflow `update_results.yml` hace sparse-checkout de `prod`, que **todavía
  tiene la estructura vieja de carpetas**. Hay que actualizarlo cuando `g-10`/`g-11`
  lleguen a `prod`.
- macOS trae bash 3.2: no hay `declare -A` en los scripts de shell.

## Ramas y sesiones

- Rama de trabajo: **`claude-test`**. Base para PRs: `staging`. Producción: `prod`.
- **Una sola sesión escribiendo el repo a la vez.** Los entrenamientos son largos
  y escriben en `data/{país}/...`: dos sesiones entrenando el mismo país con la
  misma fecha se pisan los archivos intermedios. Si hacen falta dos sesiones en
  paralelo, que una sea de solo lectura (análisis, docs) o que trabajen sobre
  países distintos.
- Antes de empezar algo largo, `git status` limpio y commit al terminar cada
  unidad de trabajo.
