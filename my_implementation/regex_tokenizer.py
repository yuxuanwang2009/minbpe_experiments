"""
Regex-based tokenizer built on the same mechanics as BasicTokenizer, but
splits text with a regex pattern (GPT-4 style) before applying merges.
"""

import regex as re


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer:
    def __init__(self, merges=None, vocab=None, pattern=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.merges = {} if merges is None else merges
        self.vocab = {i: bytes([i]) for i in range(256)} if vocab is None else vocab

    def merge(self, ids, pair, new_token, byte_shuffle=None):
        """Replace all non-overlapping occurrences of pair with new_token."""
        i = 0
        out_ids = []
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out_ids.append(new_token)
                i += 2
            else:
                out_ids.append(ids[i])
                i += 1
        return out_ids

    def _encode_chunk(self, chunk_bytes, byte_shuffle=None):
        ids = list(chunk_bytes)
        if byte_shuffle is not None:
            for i, id in enumerate(ids):
                if id<256:
                    ids[i] = byte_shuffle[id]
        for pair, new_token in self.merges.items():
            ids = self.merge(ids, pair, new_token, byte_shuffle)                
        return ids

    def train(self, text, vocab_size, special_characters: list[str] = None, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        chunks = re.findall(self.compiled_pattern, text)
        chunk_ids = [list(chunk.encode("utf-8")) for chunk in chunks] # list converts bytes to a list of ints

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        for step in range(num_merges):
            counts = {}
            for ids in chunk_ids:
                for a, b in zip(ids, ids[1:]):
                    pair = (a, b)
                    counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            next_pair = max(counts, key=counts.get)
            new_token = 256 + step
            merges[next_pair] = new_token
            vocab[new_token] = vocab[next_pair[0]] + vocab[next_pair[1]]
            if verbose:
                print(f"merge {step+1}/{num_merges}: {next_pair} -> {new_token} had {counts[next_pair]} occurrences")
            chunk_ids = [self.merge(ids, next_pair, new_token) for ids in chunk_ids]

        self.merges = merges
        self.vocab = vocab

    def encode(self, text, byte_shuffle=None):
        chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in chunks:
            ids.extend(self._encode_chunk(chunk.encode("utf-8"), byte_shuffle))
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[i] for i in ids)
        return text_bytes.decode("utf-8", errors="replace")
