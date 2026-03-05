from pathlib import Path

def run_dir(experiment: str, seed: int, mode: str) -> Path:
    """
    Standard location for artifacts from one experiment/mode/seed.
    """
    return Path("runs") / experiment / f"seed_{seed}" / mode