import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def generate(self, start_word, length, word2idx, idx2word, device):
        self.eval()
        if start_word not in word2idx:
            start_word = list(word2idx.keys())[0]  # fallback to first word

        word_idx = torch.tensor([[word2idx[start_word]]]).to(device)
        hidden = None
        words = [start_word]

        with torch.no_grad():
            for _ in range(length - 1):
                output, hidden = self.forward(word_idx, hidden)
                probs = F.softmax(output[0, -1], dim=0)
                word_idx = torch.multinomial(probs, num_samples=1).unsqueeze(0)
                words.append(idx2word[word_idx.item()])

        return " ".join(words)
