import contextlib
import io
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_enhanced as runner


class Observer:
    cycle = 1
    best_metric_cycle = 0

    def suggest_next(self):
        return dict(strategy="test", action_type="test", reasoning="fixture")

    def observe(self, action, result):
        self.observed = result
        return SimpleNamespace(magnitude=lambda: 0, velocity=0, pattern_match=None)

    def decide(self, result):
        self.decision = result


class RunnerTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(text, encoding="utf-8")
            return runner.parse_results(str(path))

    def test_actual_training_summary(self):
        result = self.parse("val_bpb:          0.997900\npeak_vram_mb: 4096.0\n")
        self.assertEqual(result["metric_value"], 0.9979)
        self.assertEqual(result["outcome"], "pending")
        self.assertEqual(result["peak_vram_mb"], 4096)

    def test_scientific_notation(self):
        self.assertEqual(self.parse("val_bpb: 9.979e-1\n")["metric_value"], 0.9979)

    def test_invalid_metric_is_crash(self):
        for metric in ("nan", "inf", "-1", "1.2garbage", "1.2.3", "1e309"):
            with self.subTest(metric=metric):
                self.assertEqual(self.parse("val_bpb: " + metric + "\n")["outcome"], "crash")

    def test_missing_or_traceback_metric_is_crash(self):
        self.assertEqual(self.parse("no summary\n")["outcome"], "crash")
        self.assertEqual(self.parse("val_bpb: 0.9\nTraceback\nRuntimeError: failed\n")["outcome"], "crash")

    def experiment(self, training, best=1.0):
        observer = Observer()
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if command[0] == "uv":
                if isinstance(training, Exception):
                    raise training
                return training
            output = "fixture\n" if "--format=%s" in command else "abcdef0\n"
            return subprocess.CompletedProcess(command, 0, output, "")

        with tempfile.TemporaryDirectory() as directory:
            old_directory = os.getcwd()
            os.chdir(directory)
            try:
                # Existing edits must survive discard, crashes and timeouts.
                edit = Path("train.py")
                edit.write_text("user edit\n", encoding="utf-8")
                with patch.object(runner.subprocess, "run", side_effect=fake_run):
                    with contextlib.redirect_stdout(io.StringIO()):
                        updated = runner.run_experiment(observer, best, "results.tsv")
                self.assertEqual(edit.read_text(encoding="utf-8"), "user edit\n")
                row = Path("results.tsv").read_text()
                log = Path("run.log").read_text(encoding="utf-8")
            finally:
                os.chdir(old_directory)
        self.assertFalse(any("reset" in call or "checkout" in call or "restore" in call for call in calls))
        self.last_log = log
        return updated, observer, row

    def test_keep_records_real_metric_and_improvement(self):
        best, observer, row = self.experiment(subprocess.CompletedProcess([], 0, "val_bpb: 0.9\n", ""))
        self.assertEqual(best, 0.9)
        self.assertEqual(observer.decision["outcome"], "keep")
        self.assertAlmostEqual(observer.observed["metric_delta"], 0.1)
        self.assertIn("0.900000", row)

    def test_discard_preserves_history_and_edits(self):
        best, observer, row = self.experiment(subprocess.CompletedProcess([], 0, "val_bpb: 1.1\n", ""))
        self.assertEqual(best, 1.0)
        self.assertEqual(observer.decision["outcome"], "discard")

    def test_nonzero_exit_never_kept(self):
        best, observer, row = self.experiment(subprocess.CompletedProcess([], 1, "val_bpb: 0.9\n", "failed"))
        self.assertEqual(best, 1.0)
        self.assertEqual(observer.decision["outcome"], "crash")
        self.assertIn("\tcrash\t", row)

    def test_timeout_records_crash_without_reset(self):
        error = subprocess.TimeoutExpired(["uv", "run", "train.py"], 600, output=b"partial\n", stderr=b"timeout\n")
        best, observer, row = self.experiment(error)
        self.assertEqual(best, 1.0)
        self.assertEqual(observer.decision["outcome"], "crash")
        self.assertIn("600", observer.decision["error"])

    def test_missing_launcher_records_crash(self):
        best, observer, row = self.experiment(FileNotFoundError("uv unavailable"))
        self.assertEqual(best, 1.0)
        self.assertEqual(observer.decision["outcome"], "crash")

    def test_timeout_with_partial_good_metric_still_crashes(self):
        error = subprocess.TimeoutExpired([], 600, output=b"val_bpb: 0.9\n", stderr=b"partial stderr\n")
        best, observer, row = self.experiment(error)
        self.assertEqual(best, 1.0)
        self.assertEqual(observer.decision["outcome"], "crash")
        self.assertIn("val_bpb: 0.9", self.last_log)
        self.assertIn("partial stderr", self.last_log)

    def test_first_baseline_has_finite_delta(self):
        best, observer, row = self.experiment(subprocess.CompletedProcess([], 0, "val_bpb: 0.9\n", ""), best=math.inf)
        self.assertEqual(best, 0.9)
        self.assertEqual(observer.decision["metric_delta"], 0)

    def test_dry_run_does_not_launch_processes(self):
        with patch.object(runner.subprocess, "run") as process:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run_experiment(Observer(), 1.0, "unused.tsv", dry_run=True), 1.0)
            process.assert_not_called()

    def test_missing_log_is_crash_with_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result = runner.parse_results(str(Path(directory) / "missing.log"))
            self.assertEqual(result["outcome"], "crash")
            self.assertTrue(result["error"])

    def test_real_observer_persists_keep_discard_crash_and_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            old_directory = os.getcwd()
            os.chdir(directory)
            responses = iter([
                subprocess.CompletedProcess([], 0, "val_bpb: 0.9\n", ""),
                subprocess.CompletedProcess([], 0, "val_bpb: 1.1\n", ""),
                subprocess.CompletedProcess([], 1, "", "failed"),
                subprocess.CompletedProcess([], 0, "val_bpb: 0.8\n", ""),
            ])

            def fake_run(command, **kwargs):
                if command[0] == "uv":
                    return next(responses)
                return subprocess.CompletedProcess(command, 0, "fixture", "")

            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    with runner.MetaObserver("smoke", "test", db_path="smoke.db", log_dir="logs") as observer:
                        best = math.inf
                        with patch.object(runner.subprocess, "run", side_effect=fake_run):
                            for cycle in range(1, 5):
                                observer.cycle = cycle
                                best = runner.run_experiment(observer, best, "results.tsv")
                        self.assertEqual(best, 0.8)
                        rows = observer.memory.conn.execute(
                            "SELECT outcome, metric_delta FROM observations ORDER BY cycle_id"
                        ).fetchall()
                        self.assertEqual([row[0] for row in rows], ["keep", "discard", "crash", "keep"])
                        self.assertTrue(all(math.isfinite(row[1]) for row in rows))
                    with runner.MetaObserver("smoke", "test", db_path="smoke.db", log_dir="logs") as recovered:
                        self.assertEqual(recovered.best_metric, 0.8)
            finally:
                os.chdir(old_directory)


if __name__ == "__main__":
    unittest.main()
