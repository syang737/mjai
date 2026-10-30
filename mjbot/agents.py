"""Neural agent and agent loading by name."""

import numpy as np
import torch

from .features import planes
from .model import Net, load, masked_logits
from .runner import Decision
from .teacher import Teacher


class NetAgent:
    """Batched policy. `on_act(decisions, x, actions, logp)` lets the RL loop record what it needs."""

    def __init__(self, net: Net, greedy: bool = True, name: str = "net", on_act=None):
        self.net = net.eval()
        self.greedy = greedy
        self.name = name
        self.on_act = on_act

    @torch.no_grad()
    def act(self, decisions: list[Decision]) -> list[int]:
        x = torch.from_numpy(np.stack([planes(d) for d in decisions]))
        mask = torch.from_numpy(np.stack([d.mask for d in decisions]))
        logits = masked_logits(self.net(x)["logits"], mask)
        if self.greedy:
            a = logits.argmax(1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        logp = logits.log_softmax(1).gather(1, a[:, None]).squeeze(1)
        if self.on_act:
            self.on_act(decisions, x, a, logp)
        return a.tolist()


def make_agent(spec: str):
    """'teacher' or a checkpoint path (greedy)."""
    if spec == "teacher":
        return Teacher()
    return NetAgent(load(spec), greedy=True, name=spec)
