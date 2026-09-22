"""Tests for serve.py and cli.py: HTTP server and command-line interface."""
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import pytest
from pathlib import Path
from unittest.mock import patch

from vlc_ua.judge.backend import ConstantJudge
from vlc_ua.judge.types import Choice, Noul
from vlc_ua.judge.serve import serve
from vlc_ua.judge import cli


REPO_ROOT = str(Path(__file__).resolve().parents[2])


def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class TestServe:
    """Test HTTP server."""

    def test_serve_accepts_request(self):
        """Start serve in daemon thread and POST a request."""
        port = find_free_port()
        judge = ConstantJudge(probs={"a": 0.6, "b": 0.4})

        # Start server in daemon thread
        server_thread = threading.Thread(
            target=serve,
            args=(judge, "127.0.0.1", port),
            daemon=True
        )
        server_thread.start()

        # Give server time to start
        time.sleep(0.5)

        # POST a request
        url = f"http://127.0.0.1:{port}/v1/systemone"
        body = {
            "state": {"text": "test"},
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "Pick one",
                    "criteria": {"a": None, "b": None}
                }
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                response = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            pytest.skip(f"Could not connect to server: {e}")

        assert "answers" in response
        assert "q1" in response["answers"]
        assert "probabilities" in response["answers"]["q1"]

    def test_serve_404_unknown_path(self):
        """POST to unknown path should return 404."""
        port = find_free_port()
        judge = ConstantJudge()

        server_thread = threading.Thread(
            target=serve,
            args=(judge, "127.0.0.1", port),
            daemon=True
        )
        server_thread.start()
        time.sleep(0.5)

        url = f"http://127.0.0.1:{port}/unknown/path"
        body = {"state": {}, "questions": {}}

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                response = json.loads(r.read().decode("utf-8"))
            # Server should return 404
            assert False, "Should have raised URLError"
        except urllib.error.HTTPError as e:
            assert e.code == 404
        except urllib.error.URLError:
            pytest.skip("Network error")

    def test_serve_answer_shape(self):
        """Verify answer shape in server response."""
        port = find_free_port()
        judge = ConstantJudge(probs={"yes": 0.7, "no": 0.3})

        server_thread = threading.Thread(
            target=serve,
            args=(judge, "127.0.0.1", port),
            daemon=True
        )
        server_thread.start()
        time.sleep(0.5)

        url = f"http://127.0.0.1:{port}/v1/systemone"
        body = {
            "state": {},
            "questions": {
                "q1": {
                    "type": "noul",
                    "instructions": "Is it true?"
                }
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                response = json.loads(r.read().decode("utf-8"))

            ans = response["answers"]["q1"]
            assert ans["type"] == "noul"
            assert "probabilities" in ans
            assert "noul" in ans or "p" not in ans  # noul key or no p
        except urllib.error.URLError:
            pytest.skip("Network error")


class TestCLITask:
    """Test CLI task command."""

    def test_cli_task_attribution(self):
        """Test 'vlc-judge task attribution' command."""
        result = subprocess.run(
            [sys.executable, "-m", "vlc_ua.judge.cli", "task", "attribution"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT
        )

        assert result.returncode == 0
        task = json.loads(result.stdout)
        assert "attribution" in task
        assert task["attribution"]["type"] == "choice"

    def test_cli_task_departure_pair(self):
        """Test 'vlc-judge task departure_pair' command."""
        result = subprocess.run(
            [sys.executable, "-m", "vlc_ua.judge.cli", "task", "departure_pair"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT
        )

        assert result.returncode == 0
        task = json.loads(result.stdout)
        assert "departure_pair" in task


class TestCLIGoldAttribution:
    """Test CLI gold-attribution command."""

    def test_cli_gold_attribution_from_txt(self, tmp_path):
        """Test building gold from .txt files."""
        txt_dir = tmp_path / "texts"
        txt_dir.mkdir()

        # Create test .txt files
        doc1 = txt_dir / "doc1.txt"
        doc1.write_text("""Рух справи.
        """ + "x" * 300)

        doc2 = txt_dir / "doc2.txt"
        doc2.write_text("""Позиція Верховного Суду.
        """ + "y" * 300)

        out_file = tmp_path / "gold.jsonl"

        result = subprocess.run(
            [sys.executable, "-m", "vlc_ua.judge.cli",
             "gold-attribution", "--texts", str(txt_dir),
             "--out", str(out_file), "--per-doc", "2"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT
        )

        assert result.returncode == 0
        assert out_file.exists()

        counts = json.loads(result.stdout)
        assert isinstance(counts, dict)


class TestCLIRun:
    """Test CLI run command."""

    def test_cli_run_keyword_backend(self, tmp_path):
        """Test 'vlc-judge run --backend keyword'."""
        # Create task file
        task_file = tmp_path / "task.json"
        task_data = {
            "attribution": {
                "type": "choice",
                "instructions": "Who?",
                "criteria": {
                    "court": None, "party": None, "lower": None, "facts": None, "procedural": None
                }
            }
        }
        task_file.write_text(json.dumps(task_data))

        # Create gold file
        gold_file = tmp_path / "gold.jsonl"
        gold_rows = [
            {
                "id": "1",
                "state": {"fragment": "Верховний Суд виходить з того"},
                "question": "attribution",
                "gold": "court",
                "sample": "random"
            }
        ]
        gold_file.write_text("\n".join(json.dumps(r) for r in gold_rows))

        out_file = tmp_path / "result.json"

        result = subprocess.run(
            [sys.executable, "-m", "vlc_ua.judge.cli",
             "run", "--backend", "keyword",
             "--task", str(task_file),
             "--gold", str(gold_file),
             "--out", str(out_file)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT
        )

        assert result.returncode == 0
        assert out_file.exists()

        run_output = json.loads(out_file.read_text())
        assert "answers" in run_output
        assert "backend" in run_output


class TestCLIReport:
    """Test CLI report command."""

    def test_cli_report_random_slice(self, tmp_path):
        """Test report generation on random slice."""
        # Create task file
        task_file = tmp_path / "task.json"
        task_data = {
            "q": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": {"a": None, "b": None}
            }
        }
        task_file.write_text(json.dumps(task_data))

        # Create gold file with enough rows for split
        gold_file = tmp_path / "gold.jsonl"
        gold_rows = [
            {
                "id": str(i),
                "state": {},
                "question": "q",
                "gold": "a" if i % 2 == 0 else "b",
                "sample": "random"
            }
            for i in range(50)
        ]
        gold_file.write_text("\n".join(json.dumps(r) for r in gold_rows))

        # Create run result
        run_file = tmp_path / "run.json"
        run_data = {
            "backend": "test",
            "task_version": "v1",
            "answers": {
                str(i): {
                    "type": "choice",
                    "probabilities": {"a": 0.6, "b": 0.4},
                    "choice": "a",
                    "confidence": 0.5
                }
                for i in range(50)
            },
            "seconds": {str(i): 0.1 for i in range(50)},
            "failures": {}
        }
        run_file.write_text(json.dumps(run_data))

        result = subprocess.run(
            [sys.executable, "-m", "vlc_ua.judge.cli",
             "report", str(run_file),
             "--task", str(task_file),
             "--gold", str(gold_file)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT
        )

        assert result.returncode == 0

        report = json.loads(result.stdout)
        assert "backend" in report
        assert "n_random" in report
        assert "temperature" in report


class TestCLIServe:
    """Test CLI serve command."""

    def test_cli_serve_keyword_backend(self):
        """Test 'vlc-judge serve --backend keyword'."""
        port = find_free_port()

        # Create task file
        task_data = {
            "attribution": {
                "type": "choice",
                "instructions": "Who?",
                "criteria": {
                    "court": None, "party": None, "lower": None, "facts": None, "procedural": None
                }
            }
        }

        # Create a temp task file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(task_data, f)
            task_file = f.name

        try:
            # Start server in subprocess
            proc = subprocess.Popen(
                [sys.executable, "-m", "vlc_ua.judge.cli",
                 "serve", "--backend", "keyword",
                 "--task", task_file,
                 "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=REPO_ROOT,
                text=True
            )

            # Give server time to start
            time.sleep(1)

            # Try to connect
            url = f"http://127.0.0.1:{port}/v1/systemone"
            body = {
                "state": {"fragment": "Верховний Суд"},
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Who?",
                        "criteria": {"court": None, "party": None, "lower": None, "facts": None, "procedural": None}
                    }
                }
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )

            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    response = json.loads(r.read().decode("utf-8"))

                assert "answers" in response
            except urllib.error.URLError:
                pytest.skip("Could not connect to server")
            finally:
                proc.terminate()
                proc.wait(timeout=2)
        finally:
            import os
            os.unlink(task_file)


class TestCalibrateOnAHoldout:
    """The trainer's own dev split is drawn from the rulings it trained on, so
    the temperature fitted there does not transfer. Measured 22.09.2026 on the
    full holdout: head v6 stored 1.9075 gave ECE 0.2552, holdout-fitted 0.9810
    gave 0.0326, and not calibrating at all gave 0.0317."""

    def _fixture(self, tmp_path, stored):
        import json

        task = tmp_path / "t.json"
        task.write_text(json.dumps({"q": {"type": "choice", "instructions": "i",
                                          "criteria": {"yes": "y", "no": "n"}}}), encoding="utf-8")
        gold = tmp_path / "g.jsonl"
        rows, answers = [], {}
        for i in range(60):
            truth = "yes" if i % 2 else "no"      # обидва класи, інакше підгонка вироджена
            hit = i % 4 != 0                      # 75% правильних
            said = truth if hit else ("no" if truth == "yes" else "yes")
            rows.append({"id": f"r{i}", "state": "текст", "question": "q",
                         "gold": truth, "sample": "random"})
            p = 0.99 if said == "yes" else 0.01   # впевнений незалежно від того, правий чи ні
            answers[f"r{i}"] = {"type": "choice", "probabilities": {"yes": p, "no": 1 - p}}
        gold.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        run = tmp_path / "run.json"
        run.write_text(json.dumps({"backend": "b", "task_version": "v1", "answers": answers}),
                       encoding="utf-8")
        head = tmp_path / "head.json"
        head.write_text(json.dumps({"temperatures": {"q": stored}}), encoding="utf-8")
        return task, gold, run, head

    def test_it_replaces_a_temperature_that_does_not_transfer(self, tmp_path, capsys):
        import json

        from vlc_ua.judge.cli import main

        task, gold, run, head = self._fixture(tmp_path, stored=20.0)

        main(["calibrate", str(run), "--task", str(task), "--gold", str(gold),
              "--model-dir", str(tmp_path), "--write"])

        meta = json.loads(head.read_text(encoding="utf-8"))
        assert meta["temperatures"]["q"] != 20.0
        block = meta["calibration"]["q"]
        # Команда не обіцяє, що підігнана температура завжди краща — вона
        # обіцяє, що температура взята з даних, яких навчання не бачило, і
        # що поруч лежать усі три ECE, щоб людина бачила, яка з них яка.
        assert set(block) >= {"n_dev", "n_test", "temperature",
                              "ece_at_this", "ece_at_stored", "ece_uncalibrated", "gold"}
        assert block["temperature"] == meta["temperatures"]["q"]
        assert block["n_dev"] + block["n_test"] == 60

    def test_without_write_the_file_is_untouched(self, tmp_path):
        import json

        from vlc_ua.judge.cli import main

        task, gold, run, head = self._fixture(tmp_path, stored=20.0)

        main(["calibrate", str(run), "--task", str(task), "--gold", str(gold),
              "--model-dir", str(tmp_path)])

        assert json.loads(head.read_text(encoding="utf-8"))["temperatures"]["q"] == 20.0

    def test_a_slice_too_small_to_calibrate_is_left_alone(self, tmp_path, capsys):
        import json

        from vlc_ua.judge.cli import main

        task, gold, run, head = self._fixture(tmp_path, stored=20.0)
        rows = gold.read_text(encoding="utf-8").splitlines()[:10]
        gold.write_text("\n".join(rows), encoding="utf-8")

        main(["calibrate", str(run), "--task", str(task), "--gold", str(gold),
              "--model-dir", str(tmp_path), "--write"])

        assert json.loads(head.read_text(encoding="utf-8"))["temperatures"]["q"] == 20.0
        assert "too few" in capsys.readouterr().err

    def test_a_torch_run_against_an_onnx_directory_is_called_out(self, tmp_path, capsys):
        """Quantisation does not preserve the temperature: head v6 measured
        0.0326 on torch at T=0.9810 and 0.1098 on int8 at that same T, needing
        1.9341 of its own. Calibrating on the wrong artefact is silent, so the
        command says it out loud."""
        import json

        from vlc_ua.judge.cli import main

        task, gold, run, head = self._fixture(tmp_path, stored=20.0)
        d = json.loads(run.read_text(encoding="utf-8"))
        d["fingerprint"] = "/somewhere/head|_TorchImpl|1024"
        run.write_text(json.dumps(d), encoding="utf-8")
        (tmp_path / "model.onnx").write_bytes(b"not really a model")

        main(["calibrate", str(run), "--task", str(task), "--gold", str(gold),
              "--model-dir", str(tmp_path), "--write"])

        assert "does not survive the quantiser" in capsys.readouterr().err
        assert json.loads(head.read_text(encoding="utf-8"))["calibration"]["q"]["fingerprint"] \
            == "/somewhere/head|_TorchImpl|1024"

    def test_an_onnx_run_passes_without_a_warning(self, tmp_path, capsys):
        import json

        from vlc_ua.judge.cli import main

        task, gold, run, head = self._fixture(tmp_path, stored=20.0)
        d = json.loads(run.read_text(encoding="utf-8"))
        d["fingerprint"] = "/somewhere/head|_OnnxImpl|1024"
        run.write_text(json.dumps(d), encoding="utf-8")
        (tmp_path / "model.onnx").write_bytes(b"not really a model")

        main(["calibrate", str(run), "--task", str(task), "--gold", str(gold),
              "--model-dir", str(tmp_path), "--write"])

        assert "quantiser" not in capsys.readouterr().err
