"""
Skeleton for the BasicTokenizer implementation.

Step 1: fill in the train, encode, and decode methods to build a byte-level
tokenizer. Keep merges/vocab bookkeeping on the instance.
"""


class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def merge(self, textBytes, nextMerge, newToken):
        i = 0
        textBytesNew = []
        while i < len(textBytes):
            if i + 1 < len(textBytes) and (textBytes[i], textBytes[i+1]) == nextMerge:
                textBytesNew.append(newToken)
                i += 2  # skip the second element of the pair
            else:
                textBytesNew.append(textBytes[i])
                i += 1
        return textBytesNew

    def train(self, text, vocab_size, verbose=False):
        """Train the tokenizer on the provided text."""
        textBytes = list(text.encode("utf-8")) # list converts bytes to a list of ints
        for newToken in range(256, vocab_size):
            count = {}
            for tok, tok2 in zip(textBytes, textBytes[1:]):
                pair = (tok, tok2)
                if pair in count:
                    count[pair] += 1
                else:
                    count[pair] = 1
            nextMerge = max(count, key=count.get)
            self.merges[nextMerge] = newToken
            self.vocab[newToken] = self.vocab[nextMerge[0]] + self.vocab[nextMerge[1]]

            textBytes = self.merge(textBytes,nextMerge,newToken)

    def encode(self, text):
        """Convert input text into a sequence of token ids."""
        textBytes = list(text.encode("utf-8"))
        for nextMerge, newToken in self.merges.items():
            textBytes = self.merge(textBytes, nextMerge, newToken)
        return textBytes
           

    def decode(self, ids):
        """Convert token ids back into text."""
        text_bytes = b"".join(self.vocab[i] if i in self.vocab else bytes([i]) for i in ids)
        return text_bytes.decode("utf-8", errors="replace")
