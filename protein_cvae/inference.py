import torch
import numpy as np
from cvae_model import CVAE
from esm_utils import get_esm_embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import joblib

# --- Configuration ---
MODEL_PATH = "trained_cvae.pt"
ENCODER_PATH = "resolution_encoder.pkl"
TFIDF_PATH = "tfidf_vectorizer.pkl"

# --- Load model and preprocessing tools ---
model = CVAE(input_dim=1280, condition_dim=3, output_dim=52)  # adjust output_dim if needed
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

encoder = joblib.load(ENCODER_PATH)
tfidf = joblib.load(TFIDF_PATH)

# --- Predict function ---
def predict_conditions(sequence, resolution_class="H"):
    # Encode conditioning
    cond_vec = encoder.transform([[resolution_class]])
    cond_tensor = torch.tensor(cond_vec, dtype=torch.float32)

    # Embed protein
    seq_embed = get_esm_embedding(sequence)
    x_tensor = torch.tensor([seq_embed], dtype=torch.float32)

    # Sample z from N(0,1)
    z = torch.randn(1, model.latent_dim)

    # Decode output
    with torch.no_grad():
        output = model.decode(z, cond_tensor).numpy()[0]

    # Split output
    pH = output[0]
    temp = output[1]
    tfidf_vector = output[2:]

    # Inverse TF-IDF (approximate: get top terms)
    terms = tfidf.get_feature_names_out()
    top_indices = tfidf_vector.argsort()[-5:][::-1]
    condition_terms = [terms[i] for i in top_indices]

    return {
        "pH": round(pH, 2),
        "temperature": round(temp, 1),
        "ingredients": condition_terms
    }

# --- Example Usage ---
if __name__ == "__main__":
    sequence = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQR"  # Example protein
    result = predict_conditions(sequence, resolution_class="H")
    print("Predicted Crystallization Conditions:")
    print(result)
