import random
import pandas as pd
import torch
import gensim as gensim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # ht = sigmoid(Wrec · ht−1 + Win · xt )
        out, _ = self.rnn(x, h0)
        # hT : the last output of the RNN
        # Probability : yT = sigmoid(Wout · hT )
        out = self.fc(out[:, -1, :])
        return out
    
def load_data(csv_file="IMDB Dataset.csv", num_reviews=10000, split_ratio=0.9):
    df = pd.read_csv(csv_file)
    pos_reviews = df[df["sentiment"] == "positive"]["review"].tolist()[:num_reviews // 2]
    neg_reviews = df[df["sentiment"] == "negative"]["review"].tolist()[:num_reviews // 2]
    reviews = [(review, 1) for review in pos_reviews] + [(review, 0) for review in neg_reviews]
    random.shuffle(reviews)
    split_idx = int(len(reviews) * split_ratio)
    train_data = reviews[:split_idx]
    test_data = reviews[split_idx:]
    return train_data, test_data

def preprocess_review(review, word_emb, seq_length=50):
    review = gensim.utils.simple_preprocess(review)
    embedded_review = []
    for word in review:
        if word in word_emb:
            embedded_review.append(word_emb[word])
    if len(embedded_review) < seq_length:
        embedded_review += [word_emb["unk"]] * (seq_length - len(embedded_review))
    else:
        embedded_review = embedded_review[:seq_length]
    return torch.tensor(embedded_review).float()

def prepare_data(data, word_emb, seq_length=50):
    X = []
    y = []
    for review, label in data:
        embedded_review = preprocess_review(review, word_emb, seq_length)
        X.append(embedded_review)
        y.append(label)
    X = torch.stack(X)
    y = torch.tensor(y).long()
    return X, y