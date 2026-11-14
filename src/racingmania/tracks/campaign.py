"""The five hand-made campaign tracks: the game's main tracks and the specialist experiment's tracks.

Global context: campaign tracks are written as explicit segment plans (same format as the
generator's, ``ribbon.build_track``) so they are reproducible from code. They form a hand-made,
out-of-distribution set for the generalist (reports/preregistration.md, section 3) and the
single-track training set for each specialist. Track 1 is deliberately simple; later tracks add
hills, banked sweepers, tight corners and bigger jumps.
"""

from __future__ import annotations

from .ribbon import build_track

S = lambda length, dh=0.0, jump=None: {"kind": "straight", "length": length, "dh": dh, **({"jump": jump} if jump else {})}
A = lambda radius, angle, d, dh=0.0, bank=0.0: {"kind": "arc", "radius": radius, "angle": angle, "dir": d, "dh": dh, "bank": bank}
J = lambda gap, kicker_h=1.2, kicker_len=12.0, drop=0.5, landing_len=26.0, at=30.0: {
    "at": at, "kicker_len": kicker_len, "kicker_h": kicker_h, "gap": gap, "drop": drop, "landing_len": landing_len}

CAMPAIGN_PLANS = {
    "campaign-01": [  # First Flight: two easy corners and one small jump
        S(50), A(60, 90, 1), S(110, jump=J(8.0)), A(50, 90, -1, bank=6), S(60), A(70, 120, 1), S(90)],
    "campaign-02": [  # Rolling Hills: climbs and descents through wide corners
        S(40), S(80, dh=4), A(80, 60, -1, dh=-3), S(70, dh=-3), A(45, 110, 1, dh=5, bank=8), S(90, dh=-4),
        A(60, 80, -1, dh=2), S(60, dh=3), A(90, 70, 1, dh=-5), S(90)],
    "campaign-03": [  # Banked Sweepers: long fast arcs with strong banking
        S(40), A(100, 140, 1, bank=14), S(50), A(90, 150, -1, bank=14), S(120, jump=J(10.0, drop=1.0)),
        A(110, 120, 1, bank=12), S(40), A(70, 90, -1, bank=10), S(90)],
    "campaign-04": [  # Technical: tight corners that punish late braking
        S(40), A(25, 90, -1), S(40), A(24, 120, 1), S(60), A(30, 150, -1, bank=5), S(30), A(22, 90, 1),
        S(80, dh=2), A(28, 100, -1), S(40), A(26, 80, 1), S(90)],
    "campaign-05": [  # Grand Tour: everything, with two jumps
        S(40), A(70, 70, 1, bank=6), S(120, jump=J(12.0, kicker_h=1.5)), A(40, 120, -1, dh=4, bank=8),
        S(70, dh=-4), A(30, 100, 1), S(110, jump=J(9.0, drop=1.5)), A(90, 140, -1, bank=12), S(60, dh=5),
        A(50, 90, 1, dh=-5), S(50), A(26, 110, -1), S(90)],
}


def campaign_track(name: str) -> dict:
    """Build one campaign track from its plan."""
    return build_track(CAMPAIGN_PLANS[name], name=name, generator_info={"campaign": True})


def campaign_tracks() -> list[dict]:
    """All five campaign tracks, in order."""
    return [campaign_track(n) for n in sorted(CAMPAIGN_PLANS)]
