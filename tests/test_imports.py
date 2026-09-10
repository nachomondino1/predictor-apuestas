"""
Smoke test: los módulos de librería del pipeline importan sin error.

Protege contra regresiones de imports (rutas de paquete, `sys.path`,
dependencias faltantes). NO cubre los scripts de `scripts/` ni módulos con
efectos secundarios en el import (p. ej. `dispatch_event`).
"""
import importlib

import pytest

LIBRARY_MODULES = [
    "stages",
    "main_train_models",
    # scraping (p2)
    "p2_data_understanding.collect_initial_data.web_scraping_selenium",
    "p2_data_understanding.collect_initial_data.scraper_flashscore",
    "p2_data_understanding.collect_initial_data.scraper_sofifa",
    "p2_data_understanding.collect_initial_data.update_sofifa_data",
    "p2_data_understanding.describe_data",
    # preparation (p3)
    "p3_data_preparation.format_data",
    "p3_data_preparation.clean_data",
    "p3_data_preparation.integrate_sofifa_to_flashscore",
    "p3_data_preparation.construct_data",
    "p3_data_preparation.select_data",
    "p3_data_preparation.concat_mapeos",
    # modeling (p4)
    "p4_modeling.build_model",
    "p4_modeling.assess_model",
    "p4_modeling.betting_strategy",
    "p4_modeling.generate_test_design",
    "p4_modeling.main_select_model",
    "p4_modeling.utils_select_model.assess_in_prod",
    # deployment (p6)
    "p6_deployment.main_next_matches",
    "p6_deployment.automatize_predict.collect_predictions",
    "p6_deployment.automatize_predict.update_results",
    "p6_deployment.predict_models",
    "p6_deployment.publish_bets",
    # infra
    "utils.set_up_logging",
    "utils.directories",
]


@pytest.mark.parametrize("modname", LIBRARY_MODULES)
def test_module_imports(modname):
    importlib.import_module(modname)
