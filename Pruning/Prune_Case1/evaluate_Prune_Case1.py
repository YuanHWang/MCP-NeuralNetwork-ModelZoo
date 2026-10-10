from pathlib import Path
import sys


def _find_pruning_dir():
    here = Path(__file__).resolve().parent
    for path in (here, *here.parents):
        if (path / "pruning_utils.py").is_file():
            return path
    raise RuntimeError(
        "Could not find pruning_utils.py in this folder or any parent folder."
    )


PRUNING_DIR = _find_pruning_dir()
sys.path.insert(0, str(PRUNING_DIR))

from pruning_utils import run_case  # noqa: E402


if __name__ == "__main__":
    run_case("case1")
