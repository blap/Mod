"""
Custom Tokenizer Implementation - Dependency Free
Replacing transformers.AutoTokenizer with efficient custom BPE logic.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
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

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class CustomBPETokenizer:
    """
    Efficient BPE Tokenizer implementation without external dependencies (except standard lib).
    Compatible with GPT-2/RoBERTa/Qwen style vocabularies.
    """

    def __init__(self, vocab_file: str, merges_file: str, errors: str = "replace", unk_token: str = "<|endoftext|>"):
        with open(vocab_file, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with open(merges_file, encoding="utf-8") as f:
            bpe_data = f.read().split("\n")[1:-1]

        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should match GPT-2 regex for tokenization (using standard re compatible patterns)
        # \p{L} -> [^\W\d_] (Unicode letters)
        # \p{N} -> \d (Unicode numbers)
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )

        self.unk_token = unk_token
        self.unk_token_id = self.encoder.get(unk_token, 0) # Fallback to 0 if not found

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

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
                except:
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
                pairs = get_pairs(word)

        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        bpe_tokens = []

        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        text = "".join([self.decoder.get(token, self.unk_token) for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def __call__(self, text, return_tensors=None, **kwargs):
        """Mimic transformers tokenizer call interface."""
        ids = self.encode(text)
        if return_tensors == "pt":
            import torch
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids))}
        return {"input_ids": ids}

    @property
    def pad_token_id(self):
        return self.unk_token_id

    @property
    def eos_token_id(self):
        return self.unk_token_id

def load_custom_tokenizer(model_path: str) -> CustomBPETokenizer:
    """Factory to load tokenizer from model directory."""
    vocab_file = os.path.join(model_path, "vocab.json")
    merges_file = os.path.join(model_path, "merges.txt")

    if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
        # Fallback to searching inside subdirectories or different filenames if needed
        # For now, strict check
        raise FileNotFoundError(f"Tokenizer files not found in {model_path}")

    return CustomBPETokenizer(vocab_file, merges_file)
