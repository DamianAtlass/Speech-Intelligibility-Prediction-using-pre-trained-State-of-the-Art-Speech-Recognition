from pathlib import Path

PROJECT_ROOT = Path.cwd() if Path.cwd().name != "tests" else Path.cwd().parent

GRID_FOLDER = PROJECT_ROOT / "datasets" / "grid"
BC_FOLDER = PROJECT_ROOT / "datasets" / "GridIntelligibilityDatabase"

TEST_FOLDER = PROJECT_ROOT / "tests"

TEST_GRID_FOLDER = TEST_FOLDER / "grid"
TEST_BC_FOLDER = TEST_FOLDER / "grid_bc"
