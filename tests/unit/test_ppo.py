"""PPO checks: GAE against a NumPy reference, and the factored action head."""
import numpy as np
import torch

from racingmania.rl.networks import ActorCritic
from racingmania.rl.ppo import gae


def test_gae_matches_numpy_reference():
    rng = np.random.default_rng(0)
    S, N, g, lam = 7, 3, 0.99, 0.95
    r, v, nv = rng.normal(size=(S, N)), rng.normal(size=(S, N)), rng.normal(size=N)
    d = (rng.random((S, N)) < 0.2).astype(float)
    ref = np.zeros((S, N))
    last = np.zeros(N)
    for t in range(S - 1, -1, -1):
        nxt = nv if t == S - 1 else v[t + 1]
        delta = r[t] + g * nxt * (1 - d[t]) - v[t]
        last = delta + g * lam * (1 - d[t]) * last
        ref[t] = last
    out = gae(*(torch.tensor(x) for x in (r, v, d, nv)), g, lam).numpy()
    assert np.allclose(out, ref)


def test_factored_logprob_is_sum_of_heads():
    torch.manual_seed(0)
    ac = ActorCritic(12)
    obs = torch.randn(5, 12)
    raw, a, logp, ent = ac.act(obs)
    assert torch.equal(a, raw[:, 0] * 3 + raw[:, 1])
    lp2, ent2 = ac.evaluate_actions(obs, raw)
    assert torch.allclose(logp, lp2) and torch.allclose(ent, ent2)
    out = ac.actor(obs)
    manual = (torch.log_softmax(out[:, :3], -1).gather(1, raw[:, :1]).squeeze(1)
              + torch.log_softmax(out[:, 3:], -1).gather(1, raw[:, 1:]).squeeze(1))
    assert torch.allclose(logp, manual, atol=1e-6)
