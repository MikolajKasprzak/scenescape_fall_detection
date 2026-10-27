import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from benchmark_v2 import evaluate as evaluation
from benchmark_v2.yolo11_code.main import (
    FallAlerts, comparison_speed, frame_times, main, process_clip,
    reports, retained_detections, write_comparison,
)


class AlertTests(unittest.TestCase):
    def test_continuous_fall_and_first_frame(self):
        alerts = FallAlerts()
        self.assertEqual(alerts.update(0, 0.8), {
            "video_time_s": 0, "confidence": 0.8, "first_observation": True})
        self.assertIsNone(alerts.update(1, 0.9))
        self.assertIsNone(alerts.update(2, 0.9))

    def test_gap_boundaries_start_at_first_fall_free_frame(self):
        for return_time, emits in ((1.249, False), (1.25, True), (1.251, True)):
            with self.subTest(return_time=return_time):
                alerts = FallAlerts()
                alerts.update(0, 0.8)
                self.assertIsNone(alerts.update(1, None))
                event = alerts.update(return_time, 0.7)
                self.assertEqual(event is not None, emits)
                if event:
                    self.assertEqual(event["video_time_s"], return_time)
                    self.assertFalse(event["first_observation"])

    def test_empty_frames_and_later_first_fall(self):
        alerts = FallAlerts()
        self.assertIsNone(alerts.update(0, None))
        self.assertIsNone(alerts.update(0.1, None))
        self.assertFalse(alerts.update(2, 0.5)["first_observation"])
        self.assertIsNone(alerts.update(2.1, None))
        self.assertIsNone(alerts.update(2.35, None))
        self.assertIsNotNone(alerts.update(2.4, 0.5))

    def test_short_gaps_reset_and_new_clip_resets(self):
        alerts = FallAlerts()
        alerts.update(0, 0.8)
        for timestamp, confidence in ((0.1, None), (0.2, 0.8), (0.3, None), (0.4, 0.9)):
            self.assertIsNone(alerts.update(timestamp, confidence))
        self.assertIsNotNone(FallAlerts().update(0, 0.8))

    def test_bad_timestamps_and_gap(self):
        for gap in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FallAlerts(gap)
        for timestamp in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                FallAlerts().update(timestamp, None)
        alerts = FallAlerts()
        alerts.update(1, None)
        with self.assertRaises(ValueError):
            alerts.update(1, None)

    def test_confidence_boundary_and_class_filter(self):
        boxes = [[1, 2, 3, 4, 0.399, 1], [1, 2, 3, 4, 0.4, 1],
                 [1, 2, 3, 4, 0.9, 0], [1, 2, 3, 4, 0.9, 2]]
        retained = retained_detections(boxes, 0.4)
        self.assertEqual([box["class_id"] for box in retained], [1, 0])
        self.assertEqual(retained[0]["confidence"], 0.4)
        self.assertEqual(retained_detections([], 0.4), [])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "data" / "chute01"
        self.directory.mkdir(parents=True)
        self.output = self.root / "results"
        self.output.mkdir()
        self.baseline = self.root / "baseline"
        self.baseline.mkdir()
        self.settings = {"confidence": 0.4, "fall_free_gap_s": 0.25,
                         "image_size": 640, "device": "cpu", "playback_speed": 0.25,
                         "yolo_results": str(self.output), "scenescape_results": str(self.baseline)}

    def fake_model(self, rows):
        prediction = MagicMock()
        prediction.boxes.data.cpu.return_value.tolist.return_value = rows
        model = MagicMock()
        model.predict.return_value = [prediction]
        model.predictor.args = SimpleNamespace(imgsz=640, conf=0.4)
        return model

    def run_clip(self, reads, rows=None, failure=None):
        cv2 = MagicMock()
        capture = cv2.VideoCapture.return_value
        capture.read.side_effect = [(available, object()) for available in reads]
        model = self.fake_model(rows or [])
        if failure:
            model.predict.side_effect = failure
        with patch("benchmark_v2.yolo11_code.main.frame_times", return_value=([0, 0.1], 0, "recovered frame")):
            result = process_clip(self.directory, self.output, model, cv2, self.settings)
        capture.release.assert_called_once()
        for prediction_call in model.predict.call_args_list:
            self.assertEqual(prediction_call.kwargs["device"], "cpu")
        return result

    def save_baseline(self, name="chute01", speed=0.25, status="complete"):
        (self.baseline / f"{name}.json").write_text(json.dumps({
            "chute": name, "playback_speed": speed, "status": status, "events": []}))

    def test_zero_boxes_complete_and_trace_has_every_frame(self):
        self.settings["device"] = "cuda"
        result = self.run_clip([True, True, False])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["events"], [])
        self.assertEqual(result["processed_frames"], 2)
        self.assertEqual(result["decoder_warnings"], "recovered frame")
        trace = (self.output / "chute01" / "predictions.jsonl").read_text().splitlines()
        self.assertEqual([json.loads(line)["video_time_s"] for line in trace], [0, 0.1])
        saved = json.loads((self.output / "chute01.json").read_text())
        self.assertEqual(saved, result)

    def test_multiple_boxes_one_alert_and_evaluator_accepts_schema(self):
        rows = [[0, 0, 10, 10, 0.8, 1], [1, 1, 11, 11, 0.9, 1], [0, 0, 5, 5, 0.99, 0]]
        result = self.run_clip([True, True, False], rows)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["confidence"], 0.9)
        self.assertNotIn("person_id", result["events"][0])
        (self.directory / "ground_truth.json").write_text(json.dumps({
            "reviewed": True, "falls": [{"start_s": 0, "end_s": 1}]}))
        with patch("benchmark_v2.evaluate.video_duration", return_value=2):
            row = evaluation.evaluate_clip(self.directory, self.output, 4, 1)
        self.assertEqual((row["status"], row["tp"], row["fp"]), ("scored", 1, 0))

    def test_partial_decode_failed(self):
        result = self.run_clip([True, False])
        self.assertEqual(result["status"], "failed")
        self.assertIn("1 of 2", result["error"])

    def test_extra_frame_failed(self):
        result = self.run_clip([True, True, True])
        self.assertEqual(result["status"], "failed")
        self.assertIn("more frames", result["error"])

    def test_inference_failure_failed(self):
        result = self.run_clip([True], failure=RuntimeError("bad inference"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("bad inference", result["error"])

    def test_missing_video_failed(self):
        with patch("benchmark_v2.yolo11_code.main.frame_times", side_effect=ValueError("missing video")):
            result = process_clip(self.directory, self.output, None, None, self.settings)
        self.assertEqual(result["status"], "failed")
        self.assertIn("missing video", result["error"])

    def test_baseline_speeds_include_failed_runs_and_are_not_modified(self):
        self.save_baseline()
        self.save_baseline("chute02", status="no_observed_people")
        paths = [self.baseline / "chute01.json", self.baseline / "chute02.json"]
        original = [path.read_bytes() for path in paths]
        directories = [self.directory, self.directory.with_name("chute02"), self.directory.with_name("chute03")]
        self.assertEqual(comparison_speed(self.baseline, directories), 0.25)
        self.assertEqual([path.read_bytes() for path in paths], original)
        self.save_baseline("chute02", speed=0.5)
        with self.assertRaisesRegex(ValueError, "mixed"):
            comparison_speed(self.baseline, directories)

    def test_baseline_missing_or_invalid_speed(self):
        with self.assertRaisesRegex(ValueError, "no matching"):
            comparison_speed(self.baseline, [self.directory])
        for speed in (0, 1, None, True):
            self.save_baseline(speed=speed)
            with self.assertRaises(ValueError):
                comparison_speed(self.baseline, [self.directory])

    def test_output_collision_leaves_existing_files_untouched(self):
        self.save_baseline()
        sentinel = self.output / "existing.txt"
        sentinel.write_text("keep me")
        with self.assertRaises(SystemExit) as raised:
            main(["--data-dir", str(self.directory.parent), "--output", str(self.output),
                  "--scenescape-results", str(self.baseline)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(sentinel.read_text(), "keep me")

    def probe(self, frames=None, declared="2", origin="5", warnings=""):
        metadata = {"streams": [{"start_time": origin, "nb_frames": declared}],
                    "format": {"start_time": origin, "duration": "2"},
                    "frames": [{"best_effort_timestamp_time": timestamp}
                               for timestamp in (frames if frames is not None else ("5", "5.1"))]}
        return subprocess.CompletedProcess([], 0, json.dumps(metadata), warnings)

    def test_frame_timestamps_normalize_origin_and_keep_warnings(self):
        with patch("benchmark_v2.yolo11_code.main.subprocess.run", return_value=self.probe(warnings="damaged block")):
            timestamps, origin, warnings = frame_times(Path("video.avi"))
        self.assertEqual(timestamps[0], 0)
        self.assertAlmostEqual(timestamps[1], 0.1)
        self.assertEqual(origin, 5)
        self.assertEqual(warnings, "damaged block")

    def test_bad_frame_timestamps_and_count_rejected(self):
        for metadata in (self.probe(frames=[]), self.probe(frames=("5", "5")),
                         self.probe(frames=("5", "nan")), self.probe(declared="3"),
                         self.probe(origin="nan"), self.probe(frames=("4", "5"))):
            with self.subTest(metadata=metadata), patch(
                    "benchmark_v2.yolo11_code.main.subprocess.run", return_value=metadata):
                with self.assertRaises(ValueError):
                    frame_times(Path("video.avi"))

    def scored(self, name, events):
        return {"chute": name, "status": "scored", **evaluation.score(
            [{"start_s": 1, "end_s": 2}], [{"video_time_s": time} for time in events], 10, 4)}

    def test_paired_totals_use_common_set_and_report_exclusions(self):
        yolo = [self.scored("chute01", [1, 1.5]), self.scored("chute02", [1]),
                {"chute": "chute03", "status": "unreviewed"}]
        scene = [self.scored("chute01", [1]),
                 {"chute": "chute02", "status": "failed", "reason": "no_observed_people"},
                 {"chute": "chute03", "status": "unreviewed"}]
        common = write_comparison(yolo, scene, self.output, self.settings)
        self.assertEqual(common, ["chute01"])
        with (self.output / "comparison.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["common_clips"] for row in rows], ["1", "1"])
        self.assertEqual([(row["tp"], row["fp"]) for row in rows], [("1", "1"), ("1", "0")])
        text = (self.output / "comparison.md").read_text()
        self.assertIn("no_observed_people", text)
        self.assertIn("unreviewed", text)
        self.assertIn("median_delay_source_s", text)
        self.assertIn("| YOLO11 | 1 | 1 | 1 | 0 |", text)
        self.assertIn("| chute01 | 1/1/0 | 1/0/0 |", text)
        self.assertNotIn("training-data", text)
        self.assertNotIn("Training-set overlap", text)

    def test_empty_common_set_is_not_a_zero_score(self):
        rows = [{"chute": "chute01", "status": "failed"}]
        self.assertEqual(write_comparison(rows, rows, self.output, self.settings), [])
        self.assertIn("NOT COMPARABLE", (self.output / "comparison.md").read_text())
        with (self.output / "comparison.csv").open() as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])
        with self.assertRaises(ValueError):
            write_comparison(rows, [], self.output, self.settings)

    def test_evaluate_only_needs_no_model_and_preserves_results(self):
        self.save_baseline()
        (self.output / "run.json").write_text(json.dumps(self.settings))
        before = (self.output / "run.json").read_bytes()
        with patch("benchmark_v2.yolo11_code.main.reports", return_value=0) as report:
            code = main(["--evaluate-only", "--data-dir", str(self.directory.parent),
                         "--output", str(self.output), "--scenescape-results", str(self.baseline)])
        self.assertEqual(code, 0)
        report.assert_called_once()
        self.assertEqual((self.output / "run.json").read_bytes(), before)
        self.save_baseline(speed=0.5)
        with self.assertRaises(SystemExit) as raised:
            main(["--evaluate-only", "--data-dir", str(self.directory.parent),
                  "--output", str(self.output), "--scenescape-results", str(self.baseline)])
        self.assertEqual(raised.exception.code, 2)

    def test_reports_mark_incomplete_coverage_with_nonzero_exit(self):
        self.save_baseline(status="no_observed_people")
        (self.directory / "ground_truth.json").write_text(json.dumps({"reviewed": True, "falls": []}))
        (self.output / "chute01.json").write_text(json.dumps({
            "chute": "chute01", "playback_speed": 0.25, "status": "complete", "events": []}))
        args = SimpleNamespace(output=self.output, data_dir=self.directory.parent,
                               scenescape_results=self.baseline, chute=None)
        with patch("benchmark_v2.evaluate.video_duration", return_value=2):
            self.assertEqual(reports(args, self.settings), 1)
        self.assertTrue((self.output / "reports" / "yolo" / "report.md").is_file())
        self.assertTrue((self.output / "reports" / "scenescape" / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()