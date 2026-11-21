import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


STEPS = [
    ("make_system_diagram", "Fig. 1 system diagram", []),
    ("exp00_baseline", "baseline checks, Fig. 2, control table", []),
    ("exp01_task_vs_mismatch", "Ext. 1: Fig. 3, Fig. 4, ext1 table", []),
    ("exp02_general_model", "general model: Fig. 5, general_model table", []),
    ("exp03_robust_realizable", "Ext. 2: Fig. 6, Fig. 6b, ext2 table", []),
    ("exp04_probing", "Ext. 3: Fig. 7, Fig. 7b, Fig. 8, ext3 table", []),
    ("exp05_scoring_rules", "Ext. 4: Fig. 9, ext4 table", ["exp04_probing"]),
    ("make_summary_tables", "summary tables",
     ["exp00_baseline", "exp01_task_vs_mismatch", "exp03_robust_realizable", "exp05_scoring_rules"]),
]


def run(cmd, env):
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    return proc.returncode, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="rebuild Ext. 2 figures from results/ext2_sweep.json instead of re-running its search")
    parser.add_argument("--skip-tests", action="store_true", help="skip the pytest gate")
    parser.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps (in pipeline order)")
    args = parser.parse_args()

    known = [s[0] for s in STEPS]
    if args.only:
        bad = [s for s in args.only if s not in known]
        if bad:
            parser.error(f"unknown step(s) {bad}; choose from {known}")
    steps = [s for s in STEPS if not args.only or s[0] in args.only]

    env = dict(os.environ)
    # small matrices: BLAS threads only add overhead and can exhaust OpenBLAS memory in exp03
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(var, "1")
    env["PYTHONUNBUFFERED"] = "1"
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    timings = []
    if not args.skip_tests:
        print("=" * 78 + "\n[gate] numerical checks: pytest\n" + "=" * 78, flush=True)
        code, dt = run([sys.executable, "-m", "pytest", "-q"], env)
        timings.append(("pytest", dt, code))
        if code != 0:
            print("\nTests failed, nothing regenerated.")
            sys.exit(code)

    for i, (module, what, deps) in enumerate(steps, 1):
        missing = [d for d in deps if d not in [s[0] for s in steps]]
        if missing and args.only:
            print(f"note: {module} reads saved outputs of {missing}, which are not being rerun")
        cmd = [sys.executable, "-m", f"experiments.{module}"]
        if module == "exp03_robust_realizable" and args.quick:
            if not (ROOT / "results" / "ext2_sweep.json").exists():
                print("--quick needs results/ext2_sweep.json; running the full Ext. 2 search instead")
            else:
                cmd.append("--figures-only")
                what += " (figures only, saved search)"
        print("\n" + "=" * 78 + f"\n[{i}/{len(steps)}] {module}: {what}\n" + "=" * 78, flush=True)
        code, dt = run(cmd, env)
        timings.append((module, dt, code))
        if code != 0:
            print(f"\n{module} failed (exit {code}); stopping.")
            break

    print("\n" + "=" * 78 + "\nSummary\n" + "=" * 78)
    for name, dt, code in timings:
        print(f"  {'OK  ' if code == 0 else 'FAIL'} {name:28s} {dt:8.1f} s")
    print(f"  total {sum(t[1] for t in timings) / 60:.1f} min")
    sys.exit(max(t[2] for t in timings) if timings else 0)


if __name__ == "__main__":
    main()
