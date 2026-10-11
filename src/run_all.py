import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

parser = argparse.ArgumentParser()
parser.add_argument("--skip-sweep", action="store_true",
                    help="skip run_stiffness_sweep.py (the slowest step, ~25-30 min)")
args = parser.parse_args()

STEPS = [
    ("Ground truth (eps=0.1)",               "ground_truth.py",        False),
    ("Ground truths (stiffness sweep)",      "ground_truth_multi.py",  False),
    ("Original comparison (4 methods x 3 seeds)", "run_experiments.py", False),
    ("Figures 1-5",                          "make_figures.py",        False),
    ("Stiffness sweep (45 runs)",            "run_stiffness_sweep.py", True),
    ("Figures 6-7",                          "make_sweep_figures.py",  False),
]

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)

total_start = time.time()
for i, (name, script, is_sweep) in enumerate(STEPS, 1):
    if is_sweep and args.skip_sweep:
        print(f"\n[{i}/{len(STEPS)}] SKIPPED: {name}")
        continue

    print(f"\n[{i}/{len(STEPS)}] {name}  ({script})")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], cwd=SRC)
    if result.returncode != 0:
        print(f"\nFAILED at step {i}: {script}. Fix the error above and rerun.")
        sys.exit(result.returncode)
    print(f"    done in {time.time() - t0:.0f}s")

print(f"\nAll steps finished in {(time.time() - total_start) / 60:.1f} min.")
print(f"Results: {ROOT / 'results'}\nFigures: {ROOT / 'figures'}")