"""
Smoke test: los módulos de librería del pipeline importan sin error.

Protege contra regresiones de imports (rutas de paquete, `sys.path`,
dependencias faltantes). NO cubre los scripts de `scripts/` ni módulos con
efectos secundarios en el import (p. ej. `dispatch_event`).
"""
import importlib

import pytest

LIBRARY_MODULES = [
    "predictor.stages",
    "predictor.modeling.main_train_models",
    # scraping (data_understanding)
    "predictor.data_understanding.web_scraping_selenium",
    "predictor.data_understanding.scraper_flashscore",
    "predictor.data_understanding.scraper_sofifa",
    "predictor.data_understanding.update_sofifa_data",
    "predictor.data_understanding.describe_data",
    # preparation (data_preparation)
    "predictor.data_preparation.format_data",
    "predictor.data_preparation.clean_data",
    "predictor.data_preparation.integrate_sofifa_to_flashscore",
    "predictor.data_preparation.construct_data",
    "predictor.data_preparation.select_data",
    "predictor.data_preparation.concat_mapeos",
    # modeling
    "predictor.modeling.build_model",
    "predictor.modeling.assess_model",
    "predictor.modeling.betting_strategy",
    "predictor.modeling.generate_test_design",
    "predictor.modeling.main_select_model",
    "predictor.modeling.model_selection.assess_in_prod",
    # deployment
    "predictor.deployment.main_next_matches",
    "predictor.deployment.collect_predictions",
    "predictor.deployment.update_results",
    "predictor.deployment.predict_models",
    "predictor.deployment.publish_bets",
    # infra
    "predictor.utils.set_up_logging",
    "predictor.utils.directories",
]


@pytest.mark.parametrize("modname", LIBRARY_MODULES)
def test_module_imports(modname):
    importlib.import_module(modname)
