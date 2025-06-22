
import torch
import esm

# Load once globally
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
model.eval()

def get_esm_embedding(sequence: str):
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    embedding = token_representations[0, 1:-1].mean(0).numpy()
    return embedding
