from transformers import BertTokenizer
from nltk.tokenize import RegexpTokenizer

class TokenizerWrapper:
    def __init__(self, vocab=None, max_length=128, tokenizer=None):
        self.max_length = max_length
        self.tokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = RegexpTokenizer(r"[a-zA-Z0-9]+|[!?'´`]+")
            self.vocab = vocab if vocab else {"<PAD>": 0, "<UNK>": 1}
    
    def build_vocab(self, texts):
        if self.tokenizer is None:
            special_tokens = set(self.vocab.keys())
            all_tokens = set()
            for text in texts:
                tokens = self.tokenizer.tokenize(text.lower())
                all_tokens.update(tokens)
            for token in all_tokens:
                if token not in special_tokens:
                    self.vocab[token] = len(self.vocab)
    
    def tokenize(self, text):
        if self.model_type == "bert":
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return tokens
        elif self.model_type == "rnn":
            tokens = self.tokenizer.tokenize(text.lower())
            token_ids = [
                self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens
            ]
            padded_tokens = token_ids[:self.max_length] + [self.vocab["<PAD>"]] * (self.max_length - len(token_ids))
            return {
                "input_ids": padded_tokens,
                "attention_mask": [1 if i < len(tokens) else 0 for i in range(self.max_length)],
            }

# Example Usage:
# For BERT
bert_tokenizer = TokenizerWrapper(model_type="bert", max_length=128)
bert_tokens = bert_tokenizer.tokenize("Example sentence for BERT tokenization.")

# For RNN
rnn_tokenizer = TokenizerWrapper(model_type="rnn", max_length=128)
rnn_tokenizer.build_vocab(train_df["text"])  # Build vocab from training data
rnn_tokens = rnn_tokenizer.tokenize("Example sentence for RNN tokenization.")
