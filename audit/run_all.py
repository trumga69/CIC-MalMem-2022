import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

STEPS = [
    "basic_audit.py",
    "feature_distributions.py",
    "counts_bi_mult.py",
    "conflicting_labels.py",
    "leakage_risk.py",
]

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for steps in STEPS:
    print("="*70)
    print(f"Running: {steps}", flush=True)
    res = subprocess.run([sys.executable, "-X", "utf8", str(BASE_DIR / steps)], cwd=BASE_DIR.parent)
    if res.returncode != 0: 
        print(f"{steps} thất bại exit code {res.returncode}")
        sys.exit(res.returncode)
    print()
print("Đã chạy xong")
for f in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"  - {f.name}")
