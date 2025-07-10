import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords if not already
nltk.download('stopwords')

# 1. Load and preprocess dataset
# Assuming dataset is in CSV format with columns 'text' and 'label'
df = pd.read_csv('spam.csv', encoding='latin-1')  # Adjust filename as needed

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['cleaned'] = df['text'].apply(clean_text)  # Adjust column name as needed

# 2. Vectorization (using TF-IDF for FFNN and LSTM)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label'].map({'ham': 0, 'spam': 1})  # Adjust mapping as needed

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. FFNN Model
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 5. LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        output, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.fc(x)
        return x

# 6. Transformer Model (using Hugging Face)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = df.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# 7. Train FFNN
ffnn_model = FFNN(input_dim=X_train.shape[1], hidden_dim=128, output_dim=2)
ffnn_optimizer = optim.Adam(ffnn_model.parameters())
ffnn_criterion = nn.CrossEntropyLoss()

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.toarray())
y_train_tensor = torch.LongTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test.toarray())
y_test_tensor = torch.LongTensor(y_test.values)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(3):
    for batch_X, batch_y in train_loader:
        ffnn_optimizer.zero_grad()
        outputs = ffnn_model(batch_X)
        loss = ffnn_criterion(outputs, batch_y)
        loss.backward()
        ffnn_optimizer.step()

# 8. Train LSTM
lstm_model = LSTM(input_dim=X_train.shape[1], hidden_dim=128, output_dim=2)
lstm_optimizer = optim.Adam(lstm_model.parameters())
lstm_criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(3):
    for batch_X, batch_y in train_loader:
        lstm_optimizer.zero_grad()
        outputs = lstm_model(batch_X.unsqueeze(1))  # Add sequence dimension
        loss = lstm_criterion(outputs, batch_y)
        loss.backward()
        lstm_optimizer.step()

# 9. Train Transformer
trainer.train()

# 10. Evaluate models
ffnn_model.eval()
lstm_model.eval()

with torch.no_grad():
    ffnn_preds = ffnn_model(X_test_tensor).argmax(dim=1)
    lstm_preds = lstm_model(X_test_tensor.unsqueeze(1)).argmax(dim=1)

ffnn_report = classification_report(y_test, ffnn_preds)
lstm_report = classification_report(y_test, lstm_preds)

# Evaluate Transformer
transformer_preds = trainer.predict(tokenized_datasets['test'])
transformer_report = classification_report(y_test, transformer_preds.argmax(axis=1))

print("FFNN Results:")
print(ffnn_report)
print("LSTM Results:")
print(lstm_report)
print("Transformer Results:")
print(transformer_report)
