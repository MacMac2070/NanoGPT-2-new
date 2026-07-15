"""Paper-faithful decoder-only GPT, following "Attention Is All You Need"
(Vaswani et al. 2017). Classes appear in the paper's section order, except
that the block and full stack (section 3.1) come last because Python needs
the components defined first.

    3.2.1  scaled_dot_product_attention
    3.2.2  MultiHeadAttention
    3.3    PositionwiseFeedForward
    3.5    PositionalEncoding
    3.1    DecoderBlock (masked self-attention + feed-forward, post-norm)
    3.4    GPT (embeddings, tied pre-softmax linear)

Decoder-only: no encoder and no cross-attention, so each block has two
sub-layers, not the paper's three. Sub-layer 1 is masked multi-head
self-attention (each character gathers information from earlier characters
only); sub-layer 2 is the feed-forward network (each character processes the
information it received). Around each sub-layer sit the two helpers:
Add (the residual connection) and Norm (LayerNorm), in the paper's post-norm
order LayerNorm(x + Sublayer(x)).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """Section 3.2.1, equation (1): Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.

    Query = what each character is looking for right now.
    Key   = the label each character advertises about itself.
    Value = the actual content each character carries.

    Shapes: query/key/value are (..., T, d_k); typically (B, h, T, d_k).
    mask is broadcastable to (..., T, T) and True where attention is allowed.

    Steps: score every query against every key (one number per pair), scale
    by 1/sqrt(d_k) so large d_k does not push the softmax into regions of
    tiny gradients, mask, softmax, then blend the values by weight.
    Returns (output, attention_weights).
    """
    d_k = query.size(-1)
    # Scale by sqrt(d_k), the per-head width, not sqrt(d_model).
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)

    # Masking happens between scoring and softmax: future positions are set
    # to -inf so that e^-inf = 0 and they receive exactly zero weight.
    # (If -inf ever produces NaN on MPS, a large finite negative such as
    # -1e9 is the documented fallback.)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)

    # Dropout on the attention weights follows the reference implementation
    # (tensor2tensor); the paper's section 5.4 text lists only the sub-layer
    # output and embedding-sum sites.
    if dropout is not None:
        weights = dropout(weights)

    return weights @ value, weights


class MultiHeadAttention(nn.Module):
    """Section 3.2.2: h parallel attention heads, concatenated, then W^O.

    A head is one full attention cycle with its own weight matrices. The
    512-wide vector is split into h=8 sections of d_k=64 dimensions; each
    head runs its own attention and softmax, the outputs sit side by side
    (concat back to 512) and W^O lets the head outputs communicate.

    The per-head projections W_Q_i, W_K_i, W_V_i (each d_model x d_k) are
    stored fused as one d_model x d_model linear each for Q, K and V; that
    is exactly equivalent to h separate matrices stacked column-wise.
    The paper defines these as pure matrices, so bias=False throughout.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0, "d_model must divide evenly across heads"
        self.n_head = n_head
        self.d_k = d_model // n_head  # d_k = d_v = d_model / h

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # W^Q
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # W^K
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # W^V
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # W^O
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape  # batch, time, d_model

        # Project, then split the width into heads: (B, T, C) -> (B, h, T, d_k).
        q = self.w_q(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)

        out, _ = scaled_dot_product_attention(q, k, v, mask, self.attn_dropout)

        # Concat heads side by side: (B, h, T, d_k) -> (B, T, C), then W^O.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(out)


class PositionwiseFeedForward(nn.Module):
    """Section 3.3, equation (2): FFN(x) = max(0, x W1 + b1) W2 + b2.

    Applied to each position separately and identically: expand d_model to
    d_ff (512 to 2048, more space to check for patterns), ReLU (keeps
    positives, turns negatives to 0), then shrink back to d_model.
    The paper's equation has explicit biases, so bias=True here.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),   # W1, b1
            nn.ReLU(),                             # max(0, .)
            nn.Linear(d_ff, d_model, bias=True),   # W2, b2
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    """Section 3.5: fixed sinusoidal position stamps.

    Attention compares meaning but has no built-in sense of sequence, so
    each position gets its own d_model-wide vector, added to the character's
    embedding. The stamp follows a fixed, regular pattern (not learned):

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Stored as a buffer, not a parameter: it moves with .to(device) but is
    never trained and never appears in the optimiser.
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        # 10000^(-2i/d_model) computed in log space for numerical stability.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
        # Shape (1, max_len, d_model) so it broadcasts over the batch.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # Add (not replace) the stamp for the first T positions.
        return x + self.pe[:, : x.size(1), :]


class DecoderBlock(nn.Module):
    """Section 3.1: one repeatable block of the decoder stack.

    Sub-layer 1: masked multi-head self-attention (gather information).
    Sub-layer 2: position-wise feed-forward (process information).

    Each sub-layer is wrapped by the two helpers in the paper's post-norm
    order: x = LayerNorm(x + Dropout(Sublayer(x))). The Add keeps a direct
    path for gradients; the Norm adjusts the numbers back to a normal size
    so they do not spiral too big or small. Dropout on the sub-layer output
    before the residual add is the section 5.4 placement.
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        # Two separate Norm helpers and two separate Dropouts, one per sub-layer.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Sub-layer 1: masked self-attention, then Add & Norm.
        x = self.norm1(x + self.dropout1(self.attn(x, mask)))
        # Sub-layer 2: feed-forward, then Add & Norm.
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class GPT(nn.Module):
    """The full decoder-only transformer.

    Input embedding -> + positional encoding -> dropout -> N blocks ->
    linear -> softmax (the softmax lives inside cross_entropy for training
    and inside generate() for sampling).

    Section 3.4: the token embedding and the pre-softmax linear share the
    same weight matrix, and the embedding lookup is multiplied by
    sqrt(d_model). With the embedding initialised at std 1/sqrt(d_model)
    (as in the reference implementation), the scale brings characters up to
    roughly unit size, comparable with the position stamps.
    """

    def __init__(self, vocab_size: int, n_embd: int, n_layer: int,
                 n_head: int, block_size: int, d_ff: int, dropout: float):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd

        # Section 3.4: every character in the vocabulary has a learned vector.
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # Section 3.5: fixed position stamps up to block_size positions.
        self.pos_encoding = PositionalEncoding(n_embd, block_size)
        # Section 5.4: dropout on the sum of embeddings and positional
        # encodings, applied once, after the sum.
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            DecoderBlock(n_embd, n_head, d_ff, dropout) for _ in range(n_layer)
        )

        # Pre-softmax linear. No extra final LayerNorm: the post-norm stack
        # already ends with the last block's second Norm helper.
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialise the shared matrix once, then tie. std = 1/sqrt(d_model)
        # so that the sqrt(d_model) multiplier in forward() lands the
        # embeddings at unit scale, and the tied pre-softmax projection gets
        # a sensibly small init.
        nn.init.normal_(self.token_embedding.weight,
                        mean=0.0, std=n_embd ** -0.5)
        # Section 3.4 weight tying: one shared tensor, not a copy.
        self.lm_head.weight = self.token_embedding.weight

        # Causal mask: True at and below the diagonal (allowed), False above
        # (future). Blocks each character from looking at any character that
        # comes after it. Built once, sliced to the sequence length.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, idx, targets=None):
        """idx: (B, T) integer character ids. Returns (logits, loss).

        logits has shape (B, T, vocab_size); loss is cross-entropy against
        targets (the inputs shifted one right) or None when generating.
        """
        B, T = idx.shape
        assert T <= self.block_size, (
            f"sequence length {T} exceeds block_size {self.block_size}"
        )

        tok = self.token_embedding(idx) * math.sqrt(self.n_embd)  # 3.4 scale
        x = self.pos_encoding(tok)          # add the position stamps (3.5)
        x = self.embed_dropout(x)           # 5.4, once, after the sum

        mask = self.causal_mask[:T, :T]     # (T, T), broadcasts to (B, h, T, T)
        for block in self.blocks:
            x = block(x, mask)

        logits = self.lm_head(x)            # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten so every position is one classification over the vocab.
            loss = F.cross_entropy(
                logits.view(B * T, -1), targets.reshape(B * T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """Extend idx (B, T) by max_new_tokens sampled characters.

        Linear + softmax turn the final position's vector into "scores for
        each character on how likely it will come up next"; we sample from
        that distribution and append. Call model.eval() first so dropout is
        off. (If torch.multinomial misbehaves on MPS, sampling on CPU for
        just that call is the documented fallback.)
        """
        for _ in range(max_new_tokens):
            # Crop the context to the last block_size characters, since the
            # position stamps only cover block_size positions.
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # last position, scaled by temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    # Self-test on dummy data: run `python model.py`. Small dims for speed;
    # real dims come from config.py via train.py.
    torch.manual_seed(0)
    V, B, T = 82, 4, 16
    d_model, n_head, d_ff = 32, 4, 128

    # 3.2.1: weights sum to 1; future positions get exactly zero weight.
    q = torch.randn(B, n_head, T, d_model // n_head)
    k = torch.randn(B, n_head, T, d_model // n_head)
    v = torch.randn(B, n_head, T, d_model // n_head)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    out, w = scaled_dot_product_attention(q, k, v, causal)
    assert out.shape == v.shape
    assert torch.allclose(w.sum(-1), torch.ones(B, n_head, T)), "rows must sum to 1"
    assert w.masked_select(~causal).abs().max() == 0, "future weight must be 0"
    print("scaled_dot_product_attention OK: shapes, row sums, masking")

    # 3.2.2: shape preserved through multi-head attention.
    mha = MultiHeadAttention(d_model, n_head, dropout=0.0)
    x = torch.randn(B, T, d_model)
    assert mha(x, causal).shape == (B, T, d_model)
    print("MultiHeadAttention OK: (B, T, d_model) in and out")

    # 3.3: shape preserved through the feed-forward network.
    ffn = PositionwiseFeedForward(d_model, d_ff)
    assert ffn(x).shape == (B, T, d_model)
    print("PositionwiseFeedForward OK")

    # 3.5: stamp table shape and value range.
    pe = PositionalEncoding(d_model, max_len=T)
    assert pe.pe.shape == (1, T, d_model)
    assert pe.pe.abs().max() <= 1.0
    assert pe(x).shape == x.shape
    print("PositionalEncoding OK: buffer shape, values in [-1, 1]")

    # 3.1: block preserves shape; dropout differs in train, matches in eval.
    block = DecoderBlock(d_model, n_head, d_ff, dropout=0.1)
    block.train()
    assert block(x, causal).shape == (B, T, d_model)
    assert not torch.equal(block(x, causal), block(x, causal)), \
        "train mode should be stochastic (dropout active)"
    block.eval()
    assert torch.equal(block(x, causal), block(x, causal)), \
        "eval mode should be deterministic (dropout off)"
    print("DecoderBlock OK: post-norm shape, dropout train/eval behaviour")

    # Full model: logits shape, loss near ln(V) at random init, tying, generate.
    model = GPT(vocab_size=V, n_embd=d_model, n_layer=2, n_head=n_head,
                block_size=T, d_ff=d_ff, dropout=0.1)
    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, V)
    assert torch.isfinite(loss)
    print(f"GPT OK: logits {tuple(logits.shape)}, "
          f"loss {loss.item():.3f} (ln({V}) = {math.log(V):.3f})")

    assert model.lm_head.weight is model.token_embedding.weight, \
        "3.4 tying must share one tensor, not copy it"
    print("weight tying OK: lm_head.weight IS token_embedding.weight")

    model.eval()
    out = model.generate(idx[:1, :4], max_new_tokens=8)
    assert out.shape == (1, 12)
    print(f"generate OK: 4 -> {out.shape[1]} tokens")
    print("model.py self-test passed")
