import torch
import torch.nn as nn
import torch.optim as optim


# Define the model class (should match your BiLSTM implementation)
class BiLSTMNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(BiLSTMNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Output should match num_classes

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# Model parameters
num_epochs = 15  
vocab_size = 10000  
embedding_dim = 50  
hidden_dim = 128
num_classes = 9  # Number of NER labels


model = BiLSTMNER(vocab_size, embedding_dim, hidden_dim, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Save trained model
torch.save(model.state_dict(), "model.pth")
print(" Model saved successfully!")




