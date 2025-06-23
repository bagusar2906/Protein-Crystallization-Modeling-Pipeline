import requests
import os

def download_cif(pdb_id, save_dir="pdb_mmCIF"):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    r = requests.get(url)
    if r.ok:
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/{pdb_id.lower()}.cif", "w") as f:
            f.write(r.text)

def download_multiple_cifs(pdb_ids, save_dir="pdb_mmCIF"):
    for pdb_id in pdb_ids:
        try:
            download_cif(pdb_id, save_dir)
            print(f"Downloaded {pdb_id}")
        except Exception as e:
            print(f"Failed to download {pdb_id}: {e}")

# Read PDB IDs from pdb_ids.txt (one per line)
with open("pdb_ids.txt", "r") as f:
    pdb_ids = [line.strip() for line in f if line.strip()]

download_multiple_cifs(pdb_ids)
