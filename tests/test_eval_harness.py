"""The eval harness is itself code that gates releases — so it gets tested."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_golden_set_is_well_formed():
    spec = json.loads((ROOT / "eval" / "golden.json").read_text())
    assert spec["thresholds"].keys() == {"retrieval_hit_rate", "groundedness_rate",
                                         "keyword_recall"}
    assert len(spec["cases"]) >= 10
    for case in spec["cases"]:
        assert {"id", "question", "expected_source", "expected_keywords"} <= case.keys()


def test_offline_eval_passes_thresholds():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_eval.py")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, f"offline eval failed:\n{proc.stdout}\n{proc.stderr}"
    assert "BELOW FLOOR" not in proc.stdout
