from pathlib import Path

_PROJECT_ROOT = Path.cwd() if Path.cwd().name != "tests" else Path.cwd().parent

GRID_FOLDER = _PROJECT_ROOT / "datasets" / "grid"
BC_FOLDER = _PROJECT_ROOT / "datasets" / "GridIntelligibilityDatabase"

TEST_FOLDER = _PROJECT_ROOT / "tests"

TEST_GRID_FOLDER = TEST_FOLDER / "grid"
TEST_BC_FOLDER = TEST_FOLDER / "grid_bc"
