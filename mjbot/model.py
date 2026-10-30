"""Policy network with belief heads and an oracle critic.

Trunk: 1-D ResNet over the 34 tile types. Heads:
- policy: 46 logits (mask illegal ones outside)
- belief: per opponent, P(ron on tile t), holdings distribution over 0-4 copies, P(tenpai)
- value: critic that additionally sees the oracle planes (opponents' hands, waits). Only used in
  training, so the actor stays a function of public information.
With `belief_to_policy`, predicted ron probabilities and expected holdings are fed into the
policy head (experiment b in the plan); otherwise beliefs are auxiliary-only (experiment a).
"""

import torch
import torch.nn.functional as F
from torch import nn

from .actions import N_ACTIONS
from .features import N_ORACLE, N_PLANES


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm1d(ch)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        return F.relu(x + self.b2(self.c2(F.relu(self.b1(self.c1(x))))))


def _trunk(in_ch: int, ch: int, blocks: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(in_ch, ch, 3, padding=1, bias=False), nn.BatchNorm1d(ch), nn.ReLU(),
                         *[ResBlock(ch) for _ in range(blocks)])


class Net(nn.Module):
    def __init__(self, ch: int = 96, blocks: int = 5, belief_to_policy: bool = False):
        super().__init__()
        self.config = dict(ch=ch, blocks=blocks, belief_to_policy=belief_to_policy)
        self.belief_to_policy = belief_to_policy
        self.trunk = _trunk(N_PLANES, ch, blocks)
        self.belief = nn.Conv1d(ch, 3 * 6, 1)  # per opponent: 1 ron logit + 5 holdings logits
        self.tenpai = nn.Linear(ch, 3)
        pol_in = ch + (6 if belief_to_policy else 0)
        self.pol_conv = nn.Sequential(nn.Conv1d(pol_in, 32, 1), nn.ReLU())
        self.pol_fc = nn.Linear(32 * 34 + ch, N_ACTIONS)
        self.critic = _trunk(N_PLANES + N_ORACLE, ch // 2, 2)
        self.v_fc = nn.Sequential(nn.Linear(ch // 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        b = self.belief(h).view(-1, 3, 6, 34)
        ron = b[:, :, 0]
        hold = b[:, :, 1:].transpose(2, 3)  # (B, 3, 34, 5)
        pooled = h.mean(-1)
        p_in = h
        if self.belief_to_policy:
            exp_hold = (hold.softmax(-1) * torch.arange(5, device=x.device)).sum(-1) / 4
            p_in = torch.cat([h, ron.sigmoid(), exp_hold], 1)
        logits = self.pol_fc(torch.cat([self.pol_conv(p_in).flatten(1), pooled], 1))
        return dict(logits=logits, ron=ron, holdings=hold, tenpai=self.tenpai(pooled))

    def value(self, x: torch.Tensor, oracle: torch.Tensor) -> torch.Tensor:
        return self.v_fc(self.critic(torch.cat([x, oracle], 1)).mean(-1)).squeeze(-1)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e9)


def belief_loss(out, ron, holdings, tenpai) -> dict[str, torch.Tensor]:
    return dict(
        ron=F.binary_cross_entropy_with_logits(out["ron"], ron.float()),
        holdings=F.cross_entropy(out["holdings"].reshape(-1, 5), holdings.long().reshape(-1)),
        tenpai=F.binary_cross_entropy_with_logits(out["tenpai"], tenpai.float()),
    )


def save(net: Net, path) -> None:
    torch.save(dict(config=net.config, state=net.state_dict()), path)


def load(path) -> Net:
    ck = torch.load(path, map_location="cpu")
    net = Net(**ck["config"])
    net.load_state_dict(ck["state"])
    return net
