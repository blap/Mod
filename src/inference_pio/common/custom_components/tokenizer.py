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

    def __init__(self, vocab_file: str = None, merges_file: str = None, errors: str = "replace", unk_token: str = "<|endoftext|>"):
        self.encoder = {}
        self.decoder = {}
        self.bpe_ranks = {}
        self.cache = {}
        self.errors = errors
        self.unk_token = unk_token
        self.unk_token_id = 0

        if vocab_file and merges_file:
            self.load(vocab_file, merges_file)

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Should match GPT-2 regex for tokenization (using standard re compatible patterns)
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )

    def load(self, vocab_file: str, merges_file: str):
        """Load vocab and merges."""
        with open(vocab_file, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_file, encoding="utf-8") as f:
            bpe_data = f.read().split("\n")[1:-1]

        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.unk_token_id = self.encoder.get(self.unk_token, 0)

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
        if not text:
            return []

        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder.get(bpe_token, self.unk_token_id) for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(text) for text in texts]

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        text = "".join([self.decoder.get(token, self.unk_token) for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def __call__(self, text: Union[str, List[str]], return_tensors: Optional[str] = None, **kwargs):
        """Mimic transformers tokenizer call interface."""
        if isinstance(text, str):
            ids = self.encode(text)
            if return_tensors == "pt":
                import torch
                return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids))}
            return {"input_ids": ids}
        elif isinstance(text, list):
            batch_ids = self.encode_batch(text)
            if return_tensors == "pt":
                import torch
                # Pad to max length in batch
                max_len = max(len(ids) for ids in batch_ids)
                padded_ids = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in batch_ids]
                mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch_ids]
                return {
                    "input_ids": torch.tensor(padded_ids),
                    "attention_mask": torch.tensor(mask)
                }
            return {"input_ids": batch_ids}
        else:
            raise ValueError(f"Unsupported input type: {type(text)}")

    @property
    def pad_token_id(self):
        return self.unk_token_id

    @property
    def eos_token_id(self):
        return self.unk_token_id

def load_custom_tokenizer(model_path: str) -> CustomBPETokenizer:
    """Factory to load tokenizer from model directory."""
    tokenizer = CustomBPETokenizer()
    vocab_file = os.path.join(model_path, "vocab.json")
    merges_file = os.path.join(model_path, "merges.txt")

    if os.path.exists(vocab_file) and os.path.exists(merges_file):
        tokenizer.load(vocab_file, merges_file)
    else:
        logger.warning(f"Tokenizer files not found in {model_path}. Initialized empty tokenizer.")

    return tokenizer
