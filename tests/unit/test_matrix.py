"""Run queue publishing (racingmania.matrix.publish_all) and the Hugging Face collection lookup (publish.py).

Global context: the queue runs for days while report code changes on main; building inside the queue's own
process once crashed it (a new paper template met the render.py imported at queue start), and a Hugging Face
rate limit on collection creation stopped it on the same day (reports/notes/phase-2.md).
"""
import sys
from types import SimpleNamespace

from racingmania import matrix, publish


def test_publish_pulls_then_runs_publish_script_in_child(monkeypatch):
    calls = []
    monkeypatch.setattr(matrix.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)) or SimpleNamespace(returncode=0))
    matrix.publish_all("Run x/seed1 trained", "x/seed1", "trained")
    assert calls[0][0] == ["git", "pull", "--ff-only"] and calls[0][1]["check"] is False
    assert calls[1][0] == [sys.executable, "scripts/08_publish.py", "--message", "Run x/seed1 trained",
                           "--run", "x/seed1", "--status", "trained"]
    assert calls[1][1]["cwd"] == matrix.REPO_ROOT
    calls.clear()
    matrix.publish_all("Docs only")
    assert calls[1][0][-2:] == ["--message", "Docs only"]


def test_failed_publish_is_logged_and_the_queue_goes_on(monkeypatch, caplog):
    monkeypatch.setattr(matrix.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=1))
    matrix.publish_all("Run x/seed1 trained", "x/seed1", "trained")  # must not raise
    assert "publish failed" in caplog.text


class FakeApi:
    def __init__(self, existing):
        self.existing, self.created = existing, 0

    def list_collections(self, owner, limit):
        return [SimpleNamespace(title=t, slug=f"{owner}/{t.lower().replace(' ', '-')}-123") for t in self.existing]

    def create_collection(self, title, namespace, description):
        self.created += 1
        return SimpleNamespace(slug=f"{namespace}/new-456")


def test_collection_is_found_by_reading_and_created_only_when_missing(monkeypatch):
    monkeypatch.setenv("HF_NAMESPACE", "someone")
    monkeypatch.delenv("HF_COLLECTION_SLUG", raising=False)
    monkeypatch.setattr(publish, "_collection", None)
    api = FakeApi(["Other", "RacingMania RL"])
    assert publish._collection_slug(api) == "someone/racingmania-rl-123" and api.created == 0
    assert publish._collection_slug(api) == "someone/racingmania-rl-123" and api.created == 0  # cached
    monkeypatch.setattr(publish, "_collection", None)
    api = FakeApi(["Other"])
    assert publish._collection_slug(api) == "someone/new-456" and api.created == 1


def test_paused_matrix_does_not_start_the_queue(monkeypatch, tmp_path, caplog):
    m = tmp_path / "run_matrix.yaml"
    m.write_text("paused: {date: '2026-11-14', note: test}\nphases:\n- name: p\n  runs:\n  - {experiment: generalist, seed: 9}\n", encoding="utf-8")
    monkeypatch.setattr(matrix, "MATRIX_FILE", m)
    monkeypatch.setattr(matrix, "run_gpu", lambda args: (_ for _ in ()).throw(AssertionError("must not train while paused")))
    matrix.run_queue(publish=False)
    assert "paused since 2026-11-14" in caplog.text
