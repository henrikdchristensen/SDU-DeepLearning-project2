from torch import nn

class RNN_LSTM_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, padding_idx=0):
        super(RNN_LSTM_model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, bias=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_hidden = output[:, -1, :]  # extract last hidden state for each sequence
        last_hidden = self.dropout(last_hidden) # apply dropout to the last hidden state
        logits = self.fc(last_hidden) # pass last hidden state through the fully connected layer
        return logits