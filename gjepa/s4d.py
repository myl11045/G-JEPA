"""S4D temporal physical grounding used by G-JEPA."""

import torch
from torch import nn


class S4DLayer(nn.Module):
    """Diagonal recurrence with input projection and per-step physical readout.

    This implementation initializes real-valued diagonal decays and keeps them
    fixed by default. Each forward call starts with zero recurrent memory.
    """

    def __init__(self, d_in, d_out, d_state=64, dt=0.1, train_A=False):
        super().__init__()
        self.d_state = d_state
        self.dt = dt
        n = torch.arange(d_state, dtype=torch.float32)
        A_d0 = torch.exp(-(n + 0.5) * dt)
        if train_A:
            self.A = nn.Parameter(A_d0.clone())
        else:
            self.register_buffer("A", A_d0.clone())
        self.B = nn.Linear(d_in, d_state, bias=False)
        self.C = nn.Parameter(
            torch.randn(d_state, d_out) / torch.sqrt(torch.tensor(float(d_state)))
        )
        self.D = nn.Parameter(torch.zeros(d_out))
        self.ln = nn.LayerNorm(d_in)

    def forward(self, u):
        B, T, _ = u.shape
        u = self.ln(u)
        A = self.A
        h = torch.zeros(B, self.d_state, device=u.device, dtype=u.dtype)
        outs = []
        for t in range(T):
            h = A * h + self.B(u[:, t])
            outs.append(h @ self.C + self.D)
        return torch.stack(outs, dim=1)


class S4DDecoder(nn.Module):
    """Stacked causal state-space layers followed by a physical-state head.

    A single instance is shared by the observation and prediction branches;
    calling it separately keeps their recurrent memories independent.
    """

    def __init__(
        self,
        d_in,
        d_out=7,
        d_state=64,
        layers=2,
        dt=0.1,
        train_A=False,
        hidden=128,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = d_in
        for _ in range(layers):
            self.layers.append(
                S4DLayer(in_dim, hidden, d_state=d_state, dt=dt, train_A=train_A)
            )
            in_dim = hidden
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, u):
        x = u
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
