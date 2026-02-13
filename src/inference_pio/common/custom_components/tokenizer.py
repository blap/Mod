"""
Custom Tokenizer Implementation - Dependency Free
Replacing transformers.AutoTokenizer with efficient custom BPE logic.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union
from ...core.engine.backend import Tensor

logger = logging.getLogger(__name__)

# Pre-computed byte-to-unicode mapping
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class CustomBPETokenizer:
    """
    Efficient BPE Tokenizer implementation without external dependencies (except standard lib).
    Compatible with GPT-2/RoBERTa/Qwen style vocabularies.
    """

    def __init__(self, vocab_file: str, merges_file: str, errors: str = "replace", unk_token: str = "<|endoftext|>"):
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with open(merges_file, "r", encoding="utf-8") as f:
            merges = f.read().split("\n")[1:-1]

        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

        # Regex pattern for tokenization (GPT-2 style)
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.unk_token = unk_token
        self.unk_token_id = self.encoder.get(unk_token, 0)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self._get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)

        word = " ".join(word)
        self.cache[token] = word
        return word

    def _get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def encode(self, text: str) -> List[int]:
        bpe_tokens = []
        # Fallback regex if p{L} not supported in python re (it isn't by default without regex module)
        # Using simpler approximation for standard lib re
        pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")

        for token in re.findall(pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder.get(bpe_token, self.unk_token_id)
                            for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        text = "".join([self.decoder.get(token, self.unk_token) for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def __call__(self, text: str, return_tensors: Optional[str] = None, **kwargs):
        """Standard call interface compatible with transformers."""
        ids = self.encode(text)

        if return_tensors == "pt" or return_tensors == "backend":
            # Return our custom Tensor
            # Shape [1, Seq]
            t_ids = Tensor([1, len(ids)])
            # Fill data
            float_ids = [float(x) for x in ids]
            t_ids.load(float_ids)

            t_mask = Tensor([1, len(ids)])
            t_mask.fill(1.0)

            return {"input_ids": t_ids, "attention_mask": t_mask}

        return {"input_ids": ids}

    @property
    def pad_token_id(self): return self.unk_token_id
    @property
    def eos_token_id(self): return self.unk_token_id
    @property
    def vocab_size(self): return len(self.encoder)

class SimpleTokenizer:
    """
    Fallback Tokenizer for when BPE files are missing.
    Maps characters to IDs and splits by whitespace.
    """
    def __init__(self):
        self.vocab = {chr(i): i for i in range(256)}
        self.decoder = {i: chr(i) for i in range(256)}
        self.unk_token = "<unk>"
        self.unk_id = 0

    def encode(self, text: str) -> List[int]:
        # Simple char/byte encoding fallback
        return [ord(c) % 256 for c in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.decoder.get(t, "?") for t in tokens])

    def __call__(self, text: str, **kwargs):
        return {"input_ids": self.encode(text)}

    @property
    def pad_token_id(self): return 0
    @property
    def eos_token_id(self): return 0
    @property
    def vocab_size(self): return 256

def load_custom_tokenizer(model_path: Optional[str]) -> Union[CustomBPETokenizer, SimpleTokenizer]:
    try:
        if model_path:
            vocab_file = os.path.join(model_path, "vocab.json")
            merges_file = os.path.join(model_path, "merges.txt")
            if os.path.exists(vocab_file) and os.path.exists(merges_file):
                return CustomBPETokenizer(vocab_file, merges_file)
    except Exception as e:
        logger.warning(f"Failed to load custom BPE tokenizer from {model_path}: {e}")

    logger.warning(f"Using SimpleTokenizer fallback for {model_path}")
    return SimpleTokenizer()
