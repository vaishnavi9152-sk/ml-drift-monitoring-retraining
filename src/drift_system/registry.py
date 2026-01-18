import shutil
from pathlib import Path
from datetime import datetime

from drift_system.config import CFG


def promote_model(tag: str = "prod"):
    
    src = CFG.artifacts_dir / "model.joblib"
    if not src.exists():
        raise FileNotFoundError("model.joblib not found. Train first.")

    dst_dir = CFG.artifacts_dir / "registry" / tag
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / "model.joblib"
    shutil.copy2(src, dst)

    meta = dst_dir / "meta.txt"
    meta.write_text(f"promoted_at={datetime.utcnow().isoformat()}Z\n")

    return str(dst)
