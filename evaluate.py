import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

# Define NER label names
label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

# Define model class (Ensure this matches the trained model)
class BiLSTMNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(BiLSTMNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# Define model parameters
vocab_size = 10000 
embedding_dim = 50 
hidden_dim = 128
num_classes = len(label_names)

# Load trained model
model = BiLSTMNER(vocab_size, embedding_dim, hidden_dim, num_classes)
try:
    model.load_state_dict(torch.load("model.pth"))
    model.eval()  # Set model to evaluation mode
    print(" Model loaded successfully!")
except FileNotFoundError:
    print("⚠ No trained model found. Make sure 'model.pth' exists.")


test_inputs = torch.randint(0, vocab_size, (10, 20))  # 10 samples, each with 20 tokens
test_labels = torch.randint(0, num_classes, (10, 20))  # Corresponding labels

test_dataset = TensorDataset(test_inputs, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=-1)

            all_preds.extend(predictions.view(-1).tolist())
            all_labels.extend(labels.view(-1).tolist())

    print(" Evaluation completed. Generating report...")
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=1))


# Run evaluation
evaluate(model, test_dataloader)
