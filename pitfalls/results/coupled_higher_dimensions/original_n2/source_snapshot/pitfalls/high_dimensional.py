"""Run or replot the coupled n=2,4,8,16 hard-example experiment.

Use --help for grid settings, or --plot-only to audit and regenerate figures
from saved CSVs without fitting any controllers.
"""
import os
from pathlib import Path
import sys

# These must be set before NumPy imports; small dense systems favor one thread.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ese618-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Retain the public construction API used by mathematical tests and notebooks.
from pitfalls.coupled.system import higher_dimensional_pair, verify_pair, exact_schur_certificate
from pitfalls.coupled.experiment import test_states, main

if __name__ == "__main__":
    main()
