"""Browser tests of the RacingMania game (web/): load, drive, sub-path hosting, medals, AI driver.

Global context: the game must work as a static site at a server root (local dev server) and under a
sub-path (GitHub Pages). These tests start their own http.server (conftest.py), open the page in
headless Chromium and play it through the real keyboard handlers. The game exposes a read-only
`window.racingmania.state` (web/js/main.js) so a test can check the car's position and clock.
"""

import json

import pytest

from conftest import ROOT, WEB

pytestmark = pytest.mark.web


def open_menu(page, url):
    """Open the game and wait until the track list is rendered."""
    page.goto(url)
    page.wait_for_selector("#track-list [data-track]")


def start(page, track, mode):
    """Select a track in the menu and start a mode ('race', 'race-ai', 'watch')."""
    page.click(f'#track-list [data-track="{track}"]')
    page.wait_for_selector("#track-detail h2")
    page.click(f'button[data-mode="{mode}"]')
    page.wait_for_selector("#hud", state="visible")


def state(page):
    return page.evaluate("window.racingmania.state")


def use_bare_index(page):
    """Serve an index with only the five campaign tracks and no models or ghosts, so a test of the game
    itself does not depend on which trained models and extra tracks have been published to web/assets."""
    index = json.loads((WEB / "assets" / "index.json").read_text())
    index = {**index, "tracks": [t for t in index["tracks"] if t.get("kind") == "campaign"], "models": [], "ghosts": {}}
    page.route("**/assets/index.json", lambda route: route.fulfill(json=index))


def test_page_loads_cleanly_and_lists_tracks(page, site_url, console):
    use_bare_index(page)
    open_menu(page, site_url)
    page.wait_for_timeout(1000)  # let the background track loads and a few frames run
    names = page.eval_on_selector_all("#track-list [data-track]", "els => els.map(e => e.dataset.track)")
    assert names == [f"campaign-0{i}" for i in range(1, 6)]
    assert page.locator("#track-detail h2").inner_text() == "First Flight"
    assert page.locator('button[data-mode="watch"]').count() == 0  # no models listed -> no AI modes
    assert console == []


def test_holding_gas_speeds_up_and_moves_the_car(page, site_url, console):
    open_menu(page, site_url)
    start(page, "campaign-01", "race")
    before = state(page)
    assert int(page.inner_text("#hud-speed-value")) == 0
    page.keyboard.down("ArrowUp")
    page.wait_for_timeout(3000)
    speed = int(page.inner_text("#hud-speed-value"))
    after = state(page)
    page.keyboard.up("ArrowUp")
    assert speed > 20, f"HUD speed {speed} km/h after 3 s of gas"
    assert after["position"][0] - before["position"][0] > 5, (before, after)  # the first straight runs along +X
    assert after["time"] > 1.0 and page.inner_text("#hud-timer") != "00:00.00"
    page.keyboard.press("Enter")  # instant restart: clock back to zero, car at the start
    page.wait_for_timeout(100)
    restarted = state(page)
    assert restarted["time"] < 0.5 and abs(restarted["position"][0] - before["position"][0]) < 1
    assert console == []


def test_page_works_under_a_sub_path(page, serve, console):
    root_url = serve(ROOT)  # repository root served: the game lives at /web/, as on GitHub Pages
    use_bare_index(page)
    open_menu(page, root_url + "/web/")
    assert page.locator("#track-list [data-track]").count() == 5
    start(page, "campaign-02", "race")  # the track file is fetched relative to the sub-path
    assert state(page)["mode"] == "race"
    assert console == []


def test_medal_targets_follow_the_reference_time(page, site_url, console):
    track = json.loads((WEB / "assets" / "tracks" / "campaign-01.json").read_text())
    track["reference"] = {"ticks": 3000, "time": 30.0, "actions": []}  # a certified reference run of 30 s
    page.route("**/assets/tracks/campaign-01.json", lambda route: route.fulfill(json=track))
    use_bare_index(page)
    open_menu(page, site_url)
    targets = page.locator("#track-detail .targets").inner_text()
    for text in ("Gold", "00:30.00", "Silver", "00:34.50", "Bronze", "00:40.50"):
        assert text in targets
    assert "Author" not in targets  # no AI ghost listed
    start(page, "campaign-01", "race")
    assert page.locator("#hud-medals li").count() == 3
    assert console == []


def test_unseen_tracks_in_the_index_are_listed_and_playable(page, site_url, console):
    """Held-out tracks added by scripts/12_export_unseen_track.py appear in the menu and load like any track."""
    index = json.loads((WEB / "assets" / "index.json").read_text())
    unseen = [t["name"] for t in index["tracks"] if t.get("kind") == "unseen"]
    if not unseen:
        pytest.skip("no unseen track exported")
    open_menu(page, site_url)
    names = page.eval_on_selector_all("#track-list [data-track]", "els => els.map(e => e.dataset.track)")
    assert set(unseen) <= set(names)
    start(page, unseen[0], "race")
    page.wait_for_timeout(500)
    assert state(page)["mode"] == "race"
    assert console == []


def make_constant_policy(path, K):
    """A stand-in for an exported policy with the docs/formats.md interface that always chooses
    steer = none and pedal = gas. Every input feeds the output (times zero), so ONNX Runtime checks
    the names and shapes the game sends."""
    import onnx
    from onnx import TensorProto, helper

    shapes = {"chassis": [1, 13], "steer": [1, 1], "contacts": [1, 4], "prev_action": [1, 2],
              "win_center": [1, K, 3], "win_left": [1, K, 3], "win_gap": [1, K]}
    nodes = [helper.make_node("ReduceSum", [n], [f"{n}_sum"], keepdims=0) for n in shapes]  # sum over all axes
    nodes += [helper.make_node("Sum", [f"{n}_sum" for n in shapes], ["total"]),
              helper.make_node("Mul", ["total", "zero"], ["nothing"]),
              helper.make_node("Add", ["steer_const", "nothing"], ["steer_logits"]),
              helper.make_node("Add", ["pedal_const", "nothing"], ["pedal_logits"])]
    inits = [helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
             helper.make_tensor("steer_const", TensorProto.FLOAT, [1, 3], [0.0, 1.0, 0.0]),  # none
             helper.make_tensor("pedal_const", TensorProto.FLOAT, [1, 3], [1.0, 0.0, 0.0])]  # gas
    graph = helper.make_graph(
        nodes, "constant_policy",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in shapes.items()],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 3]) for n in ("steer_logits", "pedal_logits")],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # well within what onnxruntime-web 1.31 reads
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_ai_driver_runs_an_onnx_policy(page, site_url, console, tmp_path):
    K = 4
    model_path = tmp_path / "constant.onnx"
    make_constant_policy(model_path, K)
    index = json.loads((WEB / "assets" / "index.json").read_text())
    index["models"] = [{"name": "constant", "file": "assets/models/constant.onnx", "K": K, "stride": 3, "description": "test stand-in"}]
    page.route("**/assets/index.json", lambda route: route.fulfill(json=index))
    page.route("**/assets/models/constant.onnx", lambda route: route.fulfill(body=model_path.read_bytes(), content_type="application/octet-stream"))
    open_menu(page, site_url)
    assert page.locator('button[data-mode="race-ai"]').count() == 1
    start(page, "campaign-01", "watch")
    page.wait_for_timeout(3000)
    s = state(page)
    assert s["mode"] == "watch" and s["time"] > 1.0
    assert s["position"][0] > 15, s  # always-gas policy drove down the first straight
    assert abs(s["position"][2]) < 0.5, s  # and did not steer
    assert s["aiLate"] == 0, s
    assert "checkpoint" in page.inner_text("#hud-ai")
    # No action delay: the live AI is exactly where a synchronous simulation with "gas" from the first
    # tick is after the same number of ticks (the closed loop of training and the Python evaluator).
    reference = page.evaluate("""async (n) => {
        const { RaceSim } = await import('./js/core/sim.js');
        const [cfg, track] = await Promise.all(['assets/vehicle.json', 'assets/tracks/campaign-01.json'].map((u) => fetch(u).then((r) => r.json())));
        const sim = new RaceSim(cfg, track);
        for (let i = 0; i < n; i++) sim.tick(1, 0);
        return sim.state().slice(0, 3);
    }""", s["tick"])
    assert reference == pytest.approx(s["position"], abs=1e-9), (reference, s)
    assert console == []
