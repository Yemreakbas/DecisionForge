"""The phase 3 world model: a JEPA encoder, an action-conditioned predictor and
the probes that let a planner score an imagined latent.

Everything the Unity side has to reproduce lives *inside* these modules, so the
exported ONNX graphs take raw observations exactly as ``TacticalObservation``
fills them. There is no separate normalisation step to keep in sync.

    Encoder    grid (B, 7, 32, 32) + self (B, 24)  ->  z (B, D)
    Predictor  z (B, D) + intent (B, 10) one-hot    ->  z' (B, D)
    Probes     z (B, D)                             ->  (B, P) in [0, 1]

Latents are layer-normalised without affine parameters, so every latent sits on
the same shell and a squared distance between two latents means the same thing
everywhere. The predictor ends in the same normalisation, which keeps a rollout
on that shell no matter how many steps it runs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Grid channels that accumulate past 1 (see run.json / ObservationChannel.cs).
# They are squashed with log1p inside the encoder.
ACCUMULATING_CHANNELS = ("EnemyBelief", "AllyPresence", "NoiseHeat")

# Self fields that echo the action channel: LastIntent01 at t+1 *is* intent_t
# (checked: 100% of transitions), and IntentHoldTime01 resets exactly when the
# intent switches. Left in, the target latent carries the intent's own label and
# the predictor can "predict" it by copying instead of learning consequences --
# which would also make the intent-identification score meaningless. The
# encoder zeroes them; the input stays 24 wide, so the observation contract and
# the Unity side are untouched.
ECHO_FIELDS = ("LastIntent01", "IntentHoldTime01")


@dataclass
class ModelConfig:
    channels: list = field(default_factory=list)      # grid channel names, model order
    self_fields: list = field(default_factory=list)   # self field names, model order
    intents: list = field(default_factory=list)       # intent names, model order
    probes: list = field(default_factory=list)        # probe names, output order
    grid_size: int = 32
    latent_dim: int = 128
    width: int = 64
    predictor_hidden: int = 512
    intent_embed: int = 32

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)


def latent_norm(z: torch.Tensor, dim: int) -> torch.Tensor:
    # dim is passed as a plain int: z.shape[-1:] would be traced as a dynamic
    # value and the ONNX exporter refuses a dynamic normalized_shape.
    return F.layer_norm(z, (dim,))


class Encoder(nn.Module):
    """Small CNN over the tactical grid, fused with an MLP over the self vector."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c, w = len(cfg.channels), cfg.width
        self.latent_dim = cfg.latent_dim
        mask = torch.tensor([1.0 if name in ACCUMULATING_CHANNELS else 0.0 for name in cfg.channels])
        self.register_buffer("log_mask", mask.view(1, c, 1, 1))
        keep = torch.tensor([0.0 if name in ECHO_FIELDS else 1.0 for name in cfg.self_fields])
        self.register_buffer("self_mask", keep.view(1, -1))

        self.conv = nn.Sequential(
            nn.Conv2d(c, w // 2, 3, padding=1), nn.GELU(),                    # 32x32
            nn.Conv2d(w // 2, w, 3, stride=2, padding=1), nn.GELU(),          # 16x16
            nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.GELU(),           # 8x8
            nn.Conv2d(2 * w, 2 * w, 3, stride=2, padding=1), nn.GELU(),       # 4x4
            nn.Conv2d(2 * w, w, 1), nn.GELU(),
        )
        spatial = (cfg.grid_size // 8) ** 2
        self.grid_fc = nn.Linear(w * spatial, 256)
        self.self_fc = nn.Sequential(nn.Linear(len(cfg.self_fields), 64), nn.GELU())
        self.head = nn.Sequential(nn.GELU(), nn.Linear(256 + 64, 256), nn.GELU(),
                                  nn.Linear(256, cfg.latent_dim))

    def forward(self, grid: torch.Tensor, self_vec: torch.Tensor) -> torch.Tensor:
        grid = grid.float()
        grid = grid + self.log_mask * (torch.log1p(grid.clamp_min(0)) - grid)
        g = self.grid_fc(self.conv(grid).flatten(1))
        s = self.self_fc(self_vec.float() * self.self_mask)
        return latent_norm(self.head(torch.cat([g, s], dim=1)), self.latent_dim)


class Predictor(nn.Module):
    """One 0.1 s tick in latent space: z_t, intent_t -> z_t+1 (residual)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_intents = len(cfg.intents)
        self.latent_dim = cfg.latent_dim
        self.intent_embed = nn.Linear(self.num_intents, cfg.intent_embed, bias=False)
        h = cfg.predictor_hidden
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.intent_embed, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, cfg.latent_dim),
        )

    def forward(self, z: torch.Tensor, intent_onehot: torch.Tensor) -> torch.Tensor:
        a = self.intent_embed(intent_onehot.float())
        return latent_norm(z + self.net(torch.cat([z, a], dim=1)), self.latent_dim)

    def step(self, z: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        return self(z, F.one_hot(intent.long(), self.num_intents))

    def rollout(self, z: torch.Tensor, intents: torch.Tensor) -> torch.Tensor:
        """intents (B, H) -> predicted latents (B, H, D) for steps t+1 .. t+H."""
        out = []
        for k in range(intents.shape[1]):
            z = self.step(z, intents[:, k])
            out.append(z)
        return torch.stack(out, dim=1)


class Probes(nn.Module):
    """Readouts trained on frozen target latents; the planner's scoring function.

    Every output is a probability or a 0..1 quantity, so all heads share one
    sigmoid and one ONNX output.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.latent_dim, 128), nn.GELU(),
                                 nn.Linear(128, len(cfg.probes)))

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z))


class WorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.predictor = Predictor(cfg)
        self.probes = Probes(cfg)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for p_t, p_o in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_t.lerp_(p_o, 1.0 - momentum)


def vicreg_regulariser(z: torch.Tensor) -> tuple:
    """Variance and covariance terms from VICReg, over a batch of latents.

    The EMA target alone usually prevents collapse; this makes it certain, and
    the mean std doubles as the collapse monitor printed during training.
    """
    z = z.float()
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = off.pow(2).sum() / d
    return var_loss, cov_loss, std.mean()


def save_checkpoint(path, model: WorldModel, extra: dict) -> None:
    torch.save({"config": model.cfg.to_dict(), "state": model.state_dict(), **extra}, path)


def load_checkpoint(path, device="cpu") -> tuple:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = WorldModel(ModelConfig.from_dict(ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt
