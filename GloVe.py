import numpy as np
import torch
import torch.nn as nn

# Load GloVe embeddings (50D)
embedding_dict = {}
with open("glove.6B.50d.txt", "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embedding_dict[word] = vector


embedding_dim = 50  # GloVe 50D
vocab_size = 10000  # Limit vocabulary for efficiency

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for i, word in enumerate(embedding_dict.keys()):
    if i < vocab_size:
        embedding_matrix[i] = embedding_dict[word]

embedding_layer = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
