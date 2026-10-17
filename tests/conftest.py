import sys
from pathlib import Path

# Make src/ importable: `from svm import PegasosSVM`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
