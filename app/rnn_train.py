import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from app.rnn_model import RNNTextGenerator
from helper_lib.trainer import train_model
from app.bigram_model import simple_tokenizer

# 1. Dataset
class TextDataset(Dataset):
    def __init__(self, corpus, tokenizer, seq_len=30):
        self.tokens = tokenizer(corpus)
        self.seq_len = seq_len
        self.vocab = list(set(self.tokens))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.indices = [self.word2idx[w] for w in self.tokens]

    def __len__(self):
        return len(self.indices) - self.seq_len

    def __getitem__(self, idx):
        seq = self.indices[idx: idx + self.seq_len]
        target = self.indices[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(seq), torch.tensor(target)

# 2. Example corpus (replace with real one)
corpus = """
The Count of Monte Cristo is a novel written by Alexandre Dumas.
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.
We are now using RNN-based models for text generation.
"""

dataset = TextDataset(corpus, tokenizer=simple_tokenizer)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNTextGenerator(vocab_size=len(dataset.vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Train
train_model(model, train_loader, criterion, optimizer, device=device, epochs=10)

# 5. Save model and vocab
MODEL_PATH = "app/model_rnn.pth"
VOCAB_PATH = "app/rnn_vocab.pth"

torch.save(model.state_dict(), MODEL_PATH)
torch.save({"word2idx": dataset.word2idx, "idx2word": dataset.idx2word}, VOCAB_PATH)
