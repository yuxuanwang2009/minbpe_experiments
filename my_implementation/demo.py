from pathlib import Path
import sys

# ensure repo root is on path so my_implementation is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_implementation.basic_tokenizer import BasicTokenizer
from my_implementation.regex_tokenizer import RegexTokenizer
import tiktoken
enc = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer


def bpe(mergeable_ranks, token, max_rank):
    # helper function used in get_gpt4_merges() to reconstruct the merge forest
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # also see https://github.com/openai/tiktoken/issues/60
    # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]] 
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank
    return merges

def main():
    text = (ROOT / "tests/taylorswift.txt").read_text(encoding="utf-8")
    # print("\n=== RegexTokenizer (20 merges) ===")
    rt = RegexTokenizer()
    rt.train(text, vocab_size=276)
    rt_vocab = rt.vocab
    for i, (pair, new_id) in enumerate(rt.merges.items()):
        if i >= 20:
            break
        a = rt_vocab[pair[0]].decode("utf-8", errors="replace")
        b = rt_vocab[pair[1]].decode("utf-8", errors="replace")
        merged = rt_vocab[new_id].decode("utf-8", errors="replace")
        # print(f"{i+1:2}: ({a!r}, {b!r}) -> {merged!r}")

    # match this
    ids = enc.encode("hello world!!!? (안녕하세요!) 王宇轩王安琪 😉")
    text = enc.decode(ids) # get the same text back
    print(ids)

    merges = recover_merges(enc._mergeable_ranks)
    vocab = {enc._mergeable_ranks[bytes([idx])]: bytes([idx]) for idx in range(256)}
    for pair, idx in merges.items():
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

    rt2 = RegexTokenizer(merges, vocab)
    byte_shuffle = {i:enc._mergeable_ranks[bytes([i])] for i in range(256)}
    ids2 = rt2.encode("hello world!!!? (안녕하세요!) 王宇轩王安琪 😉", byte_shuffle)
    text2 = rt2.decode(ids2)
    print(ids2)
    text3=[]
    for id in ids2:
        text3.append(rt2.decode([id]))
    print(text2)


if __name__ == "__main__":
    main()
