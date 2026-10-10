from pathlib import Path
import sys


def _find_project_root():
    here = Path(__file__).resolve().parent
    for path in (here, *here.parents):
        if (path / "MCPBRNN_lib_tools").is_dir():
            return path
    raise RuntimeError(
        "Could not find project root containing MCPBRNN_lib_tools."
    )


PROJECT_DIR = _find_project_root()
sys.path.insert(0, str(PROJECT_DIR))

from MCPBRNN_lib_tools.pruning_utils import run_case  # noqa: E402


if __name__ == "__main__":
    run_case("partial")
