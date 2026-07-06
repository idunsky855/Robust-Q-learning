from pathlib import Path

from ears_q_learning.config import load_config
from ears_q_learning.data import validate_snapshot_metadata_collection


def test_primary_configuration_includes_checksum_verified_inputs() -> None:
    config = load_config(Path("configs/primary.yaml"))

    assert len(config.paths.raw_snapshots) == 3
    assert all(
        snapshot.exists() and metadata.exists()
        for snapshot, metadata in config.paths.raw_snapshots
    )
    validated = validate_snapshot_metadata_collection(config.paths.raw_snapshots)
    assert len(validated) == 3
    assert config.paths.cost_input is not None
    assert config.paths.cost_input.exists()
