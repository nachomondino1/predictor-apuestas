# Evaluación: modelo vs bet365

> Este bloque (hasta el marcador `<!-- === bloque autogenerado ... -->`) es a
> mano. Todo lo de abajo del marcador lo reescribe
> `python analysis/evaluate_vs_bet365.py` (necesita
> `data/backup_predictor_apuestas.sql`, que NO está versionado).
>
> Última corrida: 2026-09-10.

## TL;DR

Con los datos de producción a hoy (**2079 partidos resueltos**, jul-2024 →
sep-2025), **el modelo no le gana a bet365**:

| | modelo | bet365 |
|---|---|---|
| Acierto 1X2 | **49.0 %** | **53.7 %** (favorito por cuota) |
| log-loss (calibración, menor=mejor) | 1.03 | 0.96 |
| ROI estrategia real (~1900 apuestas) | **≈ 0 %** (+0.30 % / −0.35 % s/ `bet_yield`) | — |

- El acierto del modelo está **~5 puntos por debajo** del favorito del bookie, y
  sus probabilidades están **peor calibradas** que las cuotas de-vigadas.
- La estrategia de apuestas queda **en break-even**: ni gana plata ni cubre el
  margen del bookie (~5-6 % de overround).
- Único matiz a favor: a stake plano, seguir al modelo (−0.5 %) pierde bastante
  menos que seguir al favorito de bet365 (−4.3 %) → el modelo **sí** encuentra
  algo de *value* en no-favoritos, pero no alcanza para ROI positivo.
- El acierto **cae con el tiempo**: 52 % en 2024Q3 → 46 % en 2025Q3. Hay drift
  (features que envejecen, o el pipeline de prod difiere del de entrenamiento).

### Implicación para el refactor

**Todavía no se justifica invertir en modernizar los XPath de Flashscore/Sofifa.**
El cuello de botella es la calidad del modelo, no la recolección de datos. Antes
de tocar scrapers conviene:

1. Confirmar que este backtest es realmente *out-of-sample* (cada predicción se
   generó con el modelo vigente en su momento, sin ver ese partido).
2. Entender el drift trimestral (¿cambió algo en el pipeline de prod? ¿features
   que Flashscore dejó de medir?).
3. Revisar calibración y estrategia de stake (`betting_strategy.py`) — hoy
   destruye valor en las apuestas de doble-oportunidad (−26 u en 67 apuestas).

Ver plan completo en [`REFACTOR.md`](./REFACTOR.md).

### Caveats

- `bookmaker_result` sólo está cargado en ~750 de 2079 filas; el "favorito de
  bet365" se recalcula acá como la menor de las 3 cuotas.
- 49 partidos settled no tienen las 3 cuotas y quedan fuera de los cálculos que
  las necesitan.
- No se ajusta por límites de casa, cierre de línea ni disponibilidad real de la
  cuota al momento de apostar.

<!-- === bloque autogenerado: todo lo de abajo se reescribe === -->

## Cifras (autogeneradas)

- Fuente: `data/backup_predictor_apuestas.sql` (tabla `historial_predicciones`)
- Predicciones en el dump: **2199** · con resultado (settled): **2079** · con las 3 cuotas: **2030**
- Rango: **2024-07-03 → 2025-09-27**

## 1. Acierto 1X2 (menor cuota = favorito de bet365)

| | acierto |
|---|---|
| Modelo (`predicted_result`) | **49.0%** |
| bet365 (favorito por cuota) | **53.7%** |
| Apuesta real del sistema (`acerte`) | 47.9% |
| Baseline «siempre local» | 42.9% |

→ El modelo **no supera** a bet365 en acierto puro (49.0% vs 53.7%).

## 2. Calidad de las probabilidades (menor = mejor)

Sobre 2028 partidos, contra las probabilidades de bet365 de-vigadas (cuotas normalizadas).

| métrica | modelo | bet365 |
|---|---|---|
| log-loss | 1.0325 | 0.9610 |
| Brier | 0.6139 | 0.5707 |

→ Las probabilidades del modelo son **peores** que las cuotas de-vigadas.

## 3. ROI de la estrategia de apuesta del sistema

- Apuestas con `stake_to_bet > 0`: **1918** · total apostado: 8921 u
- P/L recomputado (`stake·(odd−1)` / `−stake`): **+26.8 u** → ROI **+0.30%**
- ROI según la columna `bet_yield` del sistema: **-0.35%**

## 4. ROI a stake plano (1 u), subset con cuotas

| estrategia | ROI |
|---|---|
| Seguir al modelo (`predicted_result`) | -0.50% |
| Seguir al favorito de bet365 | -4.25% |
| Apostar siempre local | -12.17% |

## 5. Acierto por trimestre

| q | n | acc_modelo | acc_bet365 |
| --- | --- | --- | --- |
| 2024Q3 | 323 | 0.523 | 0.542 |
| 2024Q4 | 519 | 0.501 | 0.539 |
| 2025Q1 | 546 | 0.485 | 0.524 |
| 2025Q2 | 391 | 0.473 | 0.552 |
| 2025Q3 | 251 | 0.458 | 0.534 |

## 6. Acierto por país

| country | n | acc_modelo | acc_bet365 |
| --- | --- | --- | --- |
| england | 436 | 0.475 | 0.537 |
| france | 356 | 0.500 | 0.551 |
| germany | 349 | 0.516 | 0.521 |
| italy | 432 | 0.491 | 0.530 |
| spain | 444 | 0.473 | 0.543 |
| usa | 13 | 0.538 | 0.692 |
