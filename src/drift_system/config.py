from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    random_state: int = 42
    test_size: float = 0.2

    drift_threshold: float = 0.15  
    min_batch_size: int = 200

    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")


CFG = Config()
