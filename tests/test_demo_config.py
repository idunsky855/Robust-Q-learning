from pathlib import Path

from ears_q_learning.config import load_config


def test_demo_config_fixes_selected_parameters_and_isolates_outputs() -> None:
    config = load_config(Path("configs/demo.yaml"))

    assert config.learning.discount_grid == (0.3,)
    assert config.learning.exploration_grid == (0.2,)
    assert config.learning.updates == 50_000
    assert config.learning.tuning_seeds == (201,)
    assert config.learning.final_seeds == (201,)
    assert config.data.economic_training_scenario == (0.15, 0.025, 0.10)
    assert config.paths.processed_dir.name == "demo"
    assert config.paths.results_dir.name == "demo"
    assert config.paths.site_dir.parts[-3:] == ("processed", "demo", "site")
