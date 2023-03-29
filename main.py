import torch.optim as optim
import gensim as gensim
import gensim.downloader as gd
import torch.nn as nn
import torch
from utils import *
from matplotlib import pyplot as plt

input_size = 50
hidden_size = 64
num_layers = 5
num_classes = 2
num_epochs = 50
batch_size = 64
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_set, test_set = load_data("IMDB Dataset.csv", 10000, 0.9)
word_emb = gd.load("glove-wiki-gigaword-50")

# Entrainement du modèle
train_X, train_y = prepare_data(train_set, word_emb)
test_X, test_y = prepare_data(test_set, word_emb)
train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    for i, (reviews, labels) in enumerate(train_loader):
        reviews = reviews.to(device)
        labels = labels.to(device)
        outputs = model(reviews)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


# Evaluation du modèle sur les données d'entrainement
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for reviews, labels in train_loader:
        reviews = reviews.to(device)
        labels = labels.to(device)
        outputs = model(reviews)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Accuracy de Train: {accuracy:.2f}%")

# Evaluation du modèle sur les données de test
model.eval()
correct = 0
total = 0
test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
with torch.no_grad():
    for reviews, labels in test_loader:
        reviews = reviews.to(device)
        labels = labels.to(device)
        outputs = model(reviews)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Accuracy de test: {accuracy:.2f}%")
