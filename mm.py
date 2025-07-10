# spam_classifier.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import nltk
from nltk.corpus import stopwords
import re

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download stopwords
nltk.download('stopwords')

# Load dataset and rename columns
print("Loading dataset...")
df = pd.read_csv('C:\\Users\\Krushna\\Downloads\\Chatbot\\spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean text
print("Cleaning text...")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['cleaned'] = df['text'].apply(clean_text)

# Vectorization (TF-IDF)
print("Vectorizing text...")
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FFNN Model
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

# Train FFNN
print("Training FFNN...")
ffnn_model = FFNN(input_dim=X_train.shape[1], hidden_dim=128, output_dim=2).to(device)
ffnn_optimizer = optim.Adam(ffnn_model.parameters())
ffnn_criterion = nn.CrossEntropyLoss()

X_train_tensor = torch.FloatTensor(X_train.toarray()).to(device)
y_train_tensor = torch.LongTensor(y_train.values).to(device)
X_test_tensor = torch.FloatTensor(X_test.toarray()).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(3):
    ffnn_model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        ffnn_optimizer.zero_grad()
        outputs = ffnn_model(batch_X)
        loss = ffnn_criterion(outputs, batch_y)
        loss.backward()
        ffnn_optimizer.step()

# Evaluate FFNN
print("Evaluating FFNN...")
ffnn_model.eval()
with torch.no_grad():
    # Convert y_test (Series) to numpy array
    ffnn_preds = ffnn_model(X_test_tensor).argmax(dim=1).cpu().numpy()
ffnn_report = classification_report(y_test.values, ffnn_preds)

print("FFNN Results:")
print(ffnn_report)

# Transformer Training
print("Training Transformer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True)

hf_dataset = Dataset.from_pandas(df[['text', 'label']])
tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)
print(f"Transformer model is on device: {next(model.parameters()).device}")


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="no",  # disables checkpoints
    report_to="none",    # disables wandb if installed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

trainer.train()

# Evaluate Transformer
print("Evaluating Transformer...")
transformer_preds = trainer.predict(tokenized_datasets['test']).predictions.argmax(axis=1)
transformer_report = classification_report(tokenized_datasets['test']['label'], transformer_preds)
print("Transformer Results:")
print(transformer_report)
