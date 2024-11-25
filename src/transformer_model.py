import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# the self attention is just like described in deep learning by Bishop, so i will not change it.
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_key):
        super().__init__()
        # Three separate linear layers for the queries, keys, and values
        self.w_q = nn.Linear(d_model, d_key)
        self.w_k = nn.Linear(d_model, d_key)
        self.w_v = nn.Linear(d_model, d_model)
    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        # Compute the attention weights
        a = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
        a = F.softmax(a, dim=-1)
        # Apply the attention weights
        z = a @ v
        return z
# Same as in book, shouldn't need any change
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, d_key, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(d_model, d_key) for _ in range(n_heads)])
        # Down projection back to model dimension
        # Alternatively, we could also split the input into n_heads and concatenate the output
        self.w_o = nn.Linear(n_heads * d_model, d_model)
    def forward(self, x):
        return self.w_o(torch.cat([h(x) for h in self.heads], dim=-1))
# maybe change siLU activation function?
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_key, n_heads, dropout1, dropout2, mlp_factor=4, ):
        super().__init__()
        # We need to init two layer norms because they have parameters
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, d_key, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        # a feedforward module
        if dropout1 > 0:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_factor * d_model),
                nn.Dropout(p = dropout1),
                nn.SiLU(),  # Swish activation function, f(x) = x * sigmoid(x)
                nn.Linear(mlp_factor * d_model, d_model),
                nn.Dropout(p = dropout2)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_factor * d_model),
                nn.SiLU(),  # Swish activation function, f(x) = x * sigmoid(x)
                nn.Linear(mlp_factor * d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, mlp_factor * d_model),
                nn.SiLU(),
                nn.Linear(d_model, mlp_factor * d_model),
            )
    def forward(self, x):
        # Residual connections and pre-layernorm
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
class TransformerClassifier(nn.Module):
    def __init__(self, n_embeds, n_classes, d_model=256, d_key=64, n_heads=2, mlp_factor=4, n_layers=2, device = "cpu", dropout1 = 0, dropout2 = 0):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.token_embedding = nn.Embedding(n_embeds, d_model)
        self.transformer_model = nn.Sequential(*[TransformerBlock(d_model, d_key, n_heads, dropout1, dropout2, mlp_factor) for _ in range(n_layers)])
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, n_classes))

    def sinusoidalPositionEncoding(self, input):
        #create empty matrix
        r_n_matrix = torch.empty((input.size(dim=1), self.d_model))
        r_n_matrix = r_n_matrix.to(self.device)
        # fill all areas of empty matrix
        for n in range(input.size(dim=1)):
            for i in range(self.d_model):
                if i % 2 == 0:
                    r_n_matrix[n, i] =  math.sin(n / 10000 ** (i / self.d_model))
                if i % 2 == 1:
                    r_n_matrix[n, i] = math.cos(n / 10000 ** (i / self.d_model))
        # add with input
        input_hat = input + r_n_matrix
        # return modified input
        return input_hat
    
    def forward(self, x):
        e = self.token_embedding(x)
        # sinusoidal positional encoding
        s = self.sinusoidalPositionEncoding(e)
        h = self.transformer_model(s)
        h = h.mean(dim=1) # Average pooling on the sequence dimension
        y = self.classifier(self.final_layer_norm(h))
        return y
