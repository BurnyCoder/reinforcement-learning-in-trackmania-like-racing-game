"""Run queue: publishing pulls main first, then builds in a fresh process (racingmania.matrix.publish_all).

Global context: the queue runs for days while report code changes on main; building inside the queue's own
process once crashed it (a new paper template met the render.py imported at queue start).
"""
import sys

from racingmania import matrix


def test_publish_pulls_then_runs_publish_script_in_child(monkeypatch):
    calls = []
    monkeypatch.setattr(matrix.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    matrix.publish_all("Run x/seed1 trained", "x/seed1", "trained")
    assert calls[0][0] == ["git", "pull", "--ff-only"] and calls[0][1]["check"] is False
    assert calls[1][0] == [sys.executable, "scripts/08_publish.py", "--message", "Run x/seed1 trained",
                           "--run", "x/seed1", "--status", "trained"]
    assert calls[1][1]["check"] is True and calls[1][1]["cwd"] == matrix.REPO_ROOT
    calls.clear()
    matrix.publish_all("Docs only")
    assert calls[1][0][-2:] == ["--message", "Docs only"]
