# minbpe_experiments / my_implementation

This folder is a lightweight fork of Karpathy’s `minbpe` ideas, with a much faster encoding algorithm. Encoding time complexity is now $O(N \log N)$ compared to the naive $O(N^2)$.

## Key differences vs Karpathy
- Added a fast encoder (`_encode_chunk_fast`) that uses a doubly linked list plus a min-heap of merge candidates to avoid rescanning all merges per pass.
- Supports optional `byte_shuffle` mapping so the tokenizer can emit the same ids as `tiktoken` (see `demo.py`).
- Regex-based tokenizer (`RegexTokenizer`) keeps the GPT-4 style split pattern but also supports saving/loading learned merges to JSON.
- Explicit linked-list utilities (`tokenizer_utils.py`) used by the fast path; basic byte-level tokenizer (`basic_tokenizer.py`) kept as a simple baseline.

## Files
- `regex_tokenizer.py` – training, slow/fast encode paths, save/load.
- `tokenizer_utils.py` – `Node` and DLL builder used by the fast encoder.
- `basic_tokenizer.py` – minimal byte-level BPE trainer/encoder.
- `demo.py` – shows how to rebuild `tiktoken`’s merge table and compare outputs.

## Quick start
```bash
# run the demo comparison with tiktoken
python demo.py
```
