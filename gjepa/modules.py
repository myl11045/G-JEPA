import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math

# Re-export these names for existing model configurations.
from .s4d import S4DDecoder, S4DLayer  # noqa: F401


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class GRUStateDecoder(nn.Module):
    """Parameter-matched causal observer used in the S4D ablation.

    Input and output semantics exactly match S4DDecoder: every output at time t
    can depend only on inputs up to t.  A one-layer 64-wide GRU plus a small
    per-step head has 54,279 trainable parameters for d_in=192, d_out=7.
    """

    def __init__(self, d_in, d_out=7, hidden=64, head_hidden=64):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, d_out),
        )

    def forward(self, u):
        x, _ = self.gru(u)
        return self.head(x)


class CausalTransformerStateDecoder(nn.Module):
    """Small causal Transformer observer matched to the S4D parameter budget.

    It has no learned absolute-position table or stochastic dropout.  Sinusoidal
    positions and a strict upper-triangular mask preserve order and prevent any
    future-state leakage.  The default model has 56,183 trainable parameters for
    d_in=192, d_out=7 (within 1.4% of the 55,431-parameter S4D observer).
    """

    def __init__(
        self,
        d_in,
        d_out=7,
        model_dim=48,
        layers=2,
        heads=4,
        ff_dim=128,
        dropout=0.0,
        head_hidden=48,
        max_len=64,
    ):
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.input_proj = nn.Linear(d_in, model_dim)
        block = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, d_out),
        )
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / model_dim)
        )
        pe = torch.zeros(max_len, model_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("position", pe.unsqueeze(0), persistent=True)

    def forward(self, u):
        length = u.size(1)
        if length > self.position.size(1):
            raise ValueError(
                f"sequence length {length} exceeds configured maximum "
                f"{self.position.size(1)}"
            )
        x = self.input_proj(u) + self.position[:, :length].to(dtype=u.dtype)
        causal_mask = torch.triu(
            torch.ones(length, length, device=u.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.encoder(x, mask=causal_mask)
        return self.head(x)


class CausalAttentionStateDecoder(nn.Module):
    """Parameter-matched attention-only causal state observer.

    Unlike a Transformer encoder block, temporal mixing contains only one
    multi-head self-attention operation and its residual connection.  The
    final MLP is applied independently at every time step.  For d_in=192 and
    d_out=7, the default configuration has 55,555 trainable parameters.
    """

    def __init__(
        self,
        d_in,
        d_out=7,
        model_dim=92,
        heads=4,
        head_hidden=32,
        dropout=0.0,
        max_len=64,
    ):
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.input_proj = nn.Linear(d_in, model_dim)
        self.attn_norm = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, d_out),
        )
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / model_dim)
        )
        pe = torch.zeros(max_len, model_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("position", pe.unsqueeze(0), persistent=True)

    def forward(self, u):
        length = u.size(1)
        if length > self.position.size(1):
            raise ValueError(
                f"sequence length {length} exceeds configured maximum "
                f"{self.position.size(1)}"
            )
        x = self.input_proj(u) + self.position[:, :length].to(dtype=u.dtype)
        causal_mask = torch.triu(
            torch.ones(length, length, device=u.device, dtype=torch.bool),
            diagonal=1,
        )
        qkv = self.attn_norm(x)
        attended, _ = self.attn(
            qkv, qkv, qkv, attn_mask=causal_mask, need_weights=False
        )
        return self.head(x + attended)


def build_state_decoder(kind, d_in, d_out):
    """Construct an observer with identical sequence I/O semantics."""
    if kind == "s4d":
        return S4DDecoder(d_in=d_in, d_out=d_out)
    if kind == "gru":
        return GRUStateDecoder(d_in=d_in, d_out=d_out)
    if kind == "transformer":
        return CausalTransformerStateDecoder(d_in=d_in, d_out=d_out)
    if kind == "transformer_wide":
        # T=3-oriented, parameter-matched Transformer observer: trade the
        # original second block for a wider token representation.
        return CausalTransformerStateDecoder(
            d_in=d_in,
            d_out=d_out,
            model_dim=64,
            layers=1,
            heads=4,
            ff_dim=176,
            dropout=0.0,
            head_hidden=48,
            max_len=64,
        )
    if kind == "attention_only":
        return CausalAttentionStateDecoder(d_in=d_in, d_out=d_out)
    raise ValueError(f"unknown observer type: {kind}")
