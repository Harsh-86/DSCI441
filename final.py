# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import re
import string
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from torch.optim import NAdam
from collections import defaultdict

from google.colab import drive
drive.mount('/content/drive')

# Initialize tqdm for pandas
tqdm.pandas()

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Configuration
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Twitter preprocessing constants
EMOTICON_SET = {
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-(', ':-(', ':(', ':-c', ':c', ':-<', ':<', ':-[', ':[',
    ':-||', '>:[', ':{', ':@', ":'-(", ":'(", 'D:<', 'D:', 'D8', 'D;', 'D=',
    'DX', 'v.v', "D-':", ">_>", "^_^", "-_-", "o_o", "O_O", "x_x", "X_X",
    "<3", "</3", "\\o/", "*\\0/*", "♥", "❌", "✅", "🔥", "😊", "😠", "😡", "🤔"
}

def preprocess_tweet(text):
    """Paper-aligned Twitter text preprocessing"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', text, flags=re.MULTILINE)

    # Replace user mentions
    text = re.sub(r'@\w+', '<USER>', text)

    # Tokenize with Twitter-aware rules
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
    tokens = tokenizer.tokenize(text)

    processed = []
    for token in tokens:
        # Preserve emojis/emoticons
        if token in EMOTICON_SET:
            processed.append(token)
        # Process hashtags
        elif token.startswith('#'):
            cleaned = token[1:].replace('_', ' ')
            processed.extend(cleaned.split())
        # Remove non-emoji punctuation
        elif token in string.punctuation:
            continue
        # Basic cleaning
        else:
            cleaned = token.translate(str.maketrans('', '', string.punctuation))
            if cleaned and cleaned not in stop_words:
                processed.append(cleaned)

    return ' '.join(processed)

def load_data():
    """Load and preprocess dataset with sampling"""
    csv_path = "/content/drive/MyDrive/Colab Notebooks/data.csv"
    df = pd.read_csv(
        csv_path,
        encoding='latin-1',
        header=None,
        usecols=[0, 5],
        names=['sentiment', 'text']
    )

    #df =  pd.read_csv(csv_path, encoding="ISO-8859-1", header=None , names=['label', 'ids', 'data', 'flag' , 'user','sentence'])
    df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})

    # Reduced dataset sampling (keep this portion)
    fraction = 0.3
    negative = df[df['sentiment'] == 0].sample(frac=fraction, random_state=RANDOM_SEED)
    positive = df[df['sentiment'] == 1].sample(frac=fraction, random_state=RANDOM_SEED)
    df = pd.concat([negative, positive]).reset_index(drop=True)

    print("Preprocessing text...")
    df['text'] = df['text'].progress_apply(preprocess_tweet)

    return train_test_split(
        df, test_size=0.2, stratify=df['sentiment'], random_state=RANDOM_SEED)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class HybridModel(nn.Module):
    def __init__(self, rnn_type, hidden_dim):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        # RNN configuration from paper
        if rnn_type == 'BiLSTM':
            self.rnn = nn.LSTM(self.roberta.config.hidden_size, hidden_dim,
                             bidirectional=True, batch_first=True)
            hidden_dim *= 2
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.roberta.config.hidden_size, hidden_dim,
                             batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(self.roberta.config.hidden_size, hidden_dim,
                            batch_first=True)

        # Paper-specified classifier
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        features = self.roberta(input_ids, attention_mask).last_hidden_state
        rnn_out, _ = self.rnn(features)
        x = rnn_out[:, -1, :]  # Last timestep
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def train_model(model, train_loader, val_loader, epochs=10):  # Increased epochs
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = NAdam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct = 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)

                outputs = model(**inputs)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # Calculate metrics
        train_loss_val = train_loss / len(train_loader)
        train_acc_val = train_correct / len(train_loader.dataset)
        val_loss_val = val_loss / len(val_loader)
        val_acc_val = val_correct / len(val_loader.dataset)

        history['train_loss'].append(train_loss_val)
        history['train_acc'].append(train_acc_val)
        history['val_loss'].append(val_loss_val)
        history['val_acc'].append(val_acc_val)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss_val:.4f} | Acc: {train_acc_val:.4f}")
        print(f"Val Loss:   {val_loss_val:.4f} | Acc: {val_acc_val:.4f}")

    return history

def ensemble_predict(models, loader, method='average'):
    all_probs = defaultdict(list)
    true_labels = []

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            true_labels.extend(batch['labels'].numpy())

            for name, model in models.items():
                model.eval()
                outputs = model(**inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs[name].append(probs)

    # Paper's ensemble strategy
    avg_probs = np.zeros((len(true_labels), 2))
    majority = np.zeros(len(true_labels))

    for name in models:
        model_probs = np.concatenate(all_probs[name], axis=0)
        avg_probs += model_probs
        majority += model_probs.argmax(axis=1)

    avg_probs /= len(models)
    majority = (majority >= len(models)//2 + 1).astype(int)

    if method == 'average':
        return true_labels, avg_probs.argmax(axis=1)
    elif method == 'majority':
        return true_labels, majority

if __name__ == "__main__":
    # Load and prepare data
    train_df, test_df = load_data()
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Create datasets
    train_dataset = SentimentDataset(train_df['text'].tolist(),
                                    train_df['sentiment'].tolist(),
                                    tokenizer)
    test_dataset = SentimentDataset(test_df['text'].tolist(),
                                   test_df['sentiment'].tolist(),
                                   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Initialize models
    models = {
        'RoBERTa-LSTM': HybridModel('LSTM', 256),
        'RoBERTa-BiLSTM': HybridModel('BiLSTM', 128),
        'RoBERTa-GRU': HybridModel('GRU', 256)
    }

    # Train individual models
    for name, model in models.items():
        print(f"\n{'='*40}\nTraining {name}\n{'='*40}")
        _ = train_model(model, train_loader, test_loader, epochs=3)  # Paper-like training

    # Ensemble evaluation
    print("\n\nEnsemble Evaluation:")

    # Average Ensemble
    y_true, y_pred = ensemble_predict(models, test_loader, 'average')
    print("\nAverage Ensemble:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Majority Voting
    y_true, y_pred = ensemble_predict(models, test_loader, 'majority')
    print("\nMajority Voting:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# prompt: visualise the above both confusion matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming y_true and y_pred are already defined from the ensemble_predict function calls
# Example for the average ensemble
cm_avg = confusion_matrix(y_true, y_pred) # from the 'average' ensemble
plt.figure(figsize=(8, 6))
sns.heatmap(cm_avg, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Average Ensemble)')
plt.show()


# Example for the majority voting ensemble (replace y_true and y_pred with the correct variables from the majority voting ensemble)
y_true, y_pred = ensemble_predict(models, test_loader, 'majority')
cm_maj = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_maj, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Majority Voting)')
plt.show()
