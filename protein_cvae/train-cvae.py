import sys
sys.path.append('/content/training/protein_cvae')

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from cvae_model import CVAE
from esm_utils import get_esm_embedding
import numpy as np

# Load dataset
df = pd.read_csv("crystallization_dataset.csv")
df = df.dropna(subset=["sequence", "resolution", "details", "pH", "temperature"])

# One-hot encode resolution class
encoder = OneHotEncoder(sparse_output=False)
res_class_onehot = encoder.fit_transform(df[["resolution_class"]])

# TF-IDF vector for condition details
tfidf = TfidfVectorizer(max_features=50)
condition_vec = tfidf.fit_transform(df["details"]).toarray()

# Embed protein sequences using ESM
df["embedding"] = df["sequence"].apply(lambda s: get_esm_embedding(s))
X_embed = np.stack(df["embedding"].values)

# Combine inputs
X_input = X_embed
C_cond = res_class_onehot
Y_output = np.hstack([df[["pH", "temperature"]].values, condition_vec])

# Train/test split
X_train, X_test, C_train, C_test, Y_train, Y_test = train_test_split(X_input, C_cond, Y_output, test_size=0.2)

# Convert to tensors, ensuring they are float32
X_train, C_train, Y_train = map(lambda x: torch.tensor(x, dtype=torch.float32), (X_train, C_train, Y_train))
X_test, C_test, Y_test = map(lambda x: torch.tensor(x, dtype=torch.float32), (X_test, C_test, Y_test))

# Train CVAE
model = CVAE(input_dim=1280, condition_dim=C_train.shape[1], output_dim=Y_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    y_pred, mu, logvar = model(X_train, C_train)
    loss, _, _ = model.loss_function(y_pred, Y_train, mu, logvar)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
