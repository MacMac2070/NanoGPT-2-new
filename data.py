"""Subword (BPE) data pipeline: corpus, tokenizer, split and batching.

The tokenizer is a byte-level BPE trained from scratch on the corpus itself
(config.bpe_vocab_size subword tokens), never a pretrained vocabulary. In
the paper's terms a "token" here is a learned subword unit, and the
embedding table in model.py is the learned vector for each of them
(section 3.4, Embeddings and Softmax).

The trained tokenizer is saved to config.TOKENIZER_PATH so train.py and
generate.py always tokenise identically. Delete that file to force a
retrain (needed if input.txt ever changes).
"""

from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

import config


def load_text(path: Path) -> str:
    """Read the whole corpus as one UTF-8 string."""
    return Path(path).read_text(encoding="utf-8")


def train_tokenizer() -> Tokenizer:
    """Train a byte-level BPE tokenizer on the corpus and save it to disk.

    Byte-level BPE starts from all 256 possible bytes, so any text can be
    encoded (there is no unknown-token case) and decode(encode(s)) == s
    exactly. add_prefix_space=False and the ByteLevel decoder are both
    required for that exact round trip. The 256 byte symbols count towards
    the config.bpe_vocab_size target; the rest are learned merges.
    """
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=config.bpe_vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
    )
    tokenizer.train([str(config.DATA_PATH)], trainer)
    config.TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(config.TOKENIZER_PATH))
    print(f"trained BPE tokenizer, saved to {config.TOKENIZER_PATH}")
    return tokenizer


def get_tokenizer() -> Tokenizer:
    """Load the saved tokenizer from disk, or train one if none exists yet."""
    if config.TOKENIZER_PATH.is_file():
        tokenizer = Tokenizer.from_file(str(config.TOKENIZER_PATH))
        print(f"loaded BPE tokenizer from {config.TOKENIZER_PATH}")
        return tokenizer
    return train_tokenizer()


def encode(s: str, tokenizer: Tokenizer) -> list[int]:
    """Text to subword integer ids."""
    return tokenizer.encode(s).ids


def decode(ids, tokenizer: Tokenizer) -> str:
    """Integer ids back to text. Accepts a list or a 1-D tensor."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return tokenizer.decode(ids)


def train_val_split(data: torch.Tensor, frac: float = 0.9):
    """First 90% for training, final 10% for validation."""
    n = int(frac * len(data))
    return data[:n], data[n:]


def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor,
              block_size: int, batch_size: int, device: str):
    """Sample a random batch of contiguous token chunks.

    x is (batch_size, block_size) of inputs; y is the same chunk shifted one
    position right, so y[b, t] is the token that follows x[b, t]. That
    alignment is what turns "read text, guess next token" into a
    supervised target at every position.
    """
    data = train_data if split == "train" else val_data
    assert len(data) > block_size, (
        f"{split} split ({len(data):,} tokens) must be longer than "
        f"block_size ({block_size})"
    )
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    # nn.Embedding and cross_entropy both need int64 ids, on the model's device.
    return x.to(device), y.to(device)


def load_dataset():
    """Convenience wrapper used by train.py: corpus -> tokenizer -> split."""
    text = load_text(config.DATA_PATH)
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != config.bpe_vocab_size:
        print(f"warning: trained vocab_size {vocab_size} != target "
              f"{config.bpe_vocab_size}; the model follows the tokenizer")
    data = torch.tensor(encode(text, tokenizer), dtype=torch.long)
    train_data, val_data = train_val_split(data)
    return tokenizer, vocab_size, train_data, val_data


if __name__ == "__main__":
    # Self-test: run `python data.py` to check the pipeline end to end.
    # Run it twice: the first run trains and saves the tokenizer, the
    # second must load the identical one from disk.
    import io

    text = load_text(config.DATA_PATH)
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"corpus: {config.DATA_PATH}")
    print(f"corpus length: {len(text):,} characters")
    print(f"vocab_size: {vocab_size}")

    # Round trip through the tokenizer, including the awkward cases: a
    # leading newline, leading spaces, curly quotes, an em dash and an
    # accented character. Byte-level BPE must reproduce all of them exactly.
    samples = (
        text[:50],
        "\n  Harry looked up. “Blimey” — said Ron, éagerly.\n",
    )
    for sample in samples:
        assert decode(encode(sample, tokenizer), tokenizer) == sample, (
            f"encode/decode mismatch on {sample!r}"
        )
    print(f"tokenizer round trip OK on {len(samples)} samples, "
          f"e.g. {samples[0]!r}")

    # Reloading from disk must give the same vocabulary and the same ids.
    reloaded = Tokenizer.from_file(str(config.TOKENIZER_PATH))
    assert reloaded.get_vocab_size() == vocab_size, "reloaded vocab differs"
    assert encode(text[:1000], reloaded) == encode(text[:1000], tokenizer), (
        "reloaded tokenizer encodes differently"
    )
    print("reload from disk OK: same vocab, same ids")

    # The checkpoint stores the tokenizer as a JSON string; it must survive
    # torch.save -> torch.load(weights_only=True) -> Tokenizer.from_str.
    buf = io.BytesIO()
    torch.save({"tokenizer_json": tokenizer.to_str()}, buf)
    buf.seek(0)
    restored = Tokenizer.from_str(
        torch.load(buf, weights_only=True)["tokenizer_json"])
    assert restored.get_vocab_size() == vocab_size
    assert decode(encode(samples[0], restored), restored) == samples[0]
    print("checkpoint tokenizer_json round trip OK")

    ids = encode(text, tokenizer)
    ratio = len(text) / len(ids)
    print(f"corpus tokens: {len(ids):,}  "
          f"compression: {ratio:.2f} characters per token")
    assert 3.0 < ratio < 5.0, (
        f"compression ratio {ratio:.2f} outside the sane 3-5 band for "
        f"English at this vocab size"
    )

    data = torch.tensor(ids, dtype=torch.long)
    train_data, val_data = train_val_split(data)
    assert len(train_data) + len(val_data) == len(ids)
    print(f"train: {len(train_data):,}  val: {len(val_data):,} tokens "
          f"(sum {len(train_data) + len(val_data):,})")

    x, y = get_batch("train", train_data, val_data,
                     config.block_size, config.batch_size, "cpu")
    print(f"batch shapes: x {tuple(x.shape)}  y {tuple(y.shape)}  dtype {x.dtype}")

    # Targets must be the inputs shifted one position right.
    assert torch.equal(x[0, 1:], y[0, :-1]), "y is not x shifted by one"
    print("alignment check: y[0][:-1] == x[0][1:] OK")
    print(f"x[0] starts: {decode(x[0][:20], tokenizer)!r}")
    print(f"y[0] starts: {decode(y[0][:20], tokenizer)!r}")
    print("data.py self-test passed")
