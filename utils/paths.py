from pathlib import Path

PROJECT_ROOT = Path.cwd() if Path.cwd().name != "tests" else Path.cwd().parent

GRID_FOLDER = PROJECT_ROOT / "datasets" / "grid"
BC_FOLDER = PROJECT_ROOT / "datasets" / "GridIntelligibilityDatabase"
CPC_DEMO_FOLDER = PROJECT_ROOT / "datasets/cpc/clarity_CPC3_data/clarity_demo_data"
TEST_FOLDER = PROJECT_ROOT / "tests"

TEST_GRID_FOLDER = TEST_FOLDER / "grid"
TEST_BC_FOLDER = TEST_FOLDER / "grid_bc"

AVSEC_FOLDER = PROJECT_ROOT / "datasets/AVSEC"
AVSEC2_ENHANCED_SAMPLES = AVSEC_FOLDER/"avsec2_enhanced_samples_output"
AVSEC2_LTR = AVSEC_FOLDER/"AVSEC-LTR-Dataset/AVSEC-2"

