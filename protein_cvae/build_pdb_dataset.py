
import os
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from io import StringIO
import gemmi

# --- CONFIG ---
DOWNLOAD_DIR = "pdb_mmCIF"
PDB_ID_FILE = "pdb_ids.txt"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# --- Download CIF ---
def download_cif(pdb_id, save_dir):
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    try:
        r = requests.get(url, timeout=10)
        if r.ok:
            with open(os.path.join(save_dir, f"{pdb_id.lower()}.cif"), "w") as f:
                f.write(r.text)
            return True
    except Exception as e:
        print(f"[!] Failed to download {pdb_id}: {e}")
    return False

# --- Extract metadata ---
def extract_metadata(cif_path):
    try:
        doc = gemmi.cif.read_file(cif_path)[0]
        res = float(doc.find_value("_refine.ls_d_res_high"))
        ph = doc.find_value("_exptl_crystal_grow.pH")
        temp = doc.find_value("_exptl_crystal_grow.temp")
        detail = doc.find_value("_exptl_crystal_grow.details")
        return {
            "pdb_id": os.path.basename(cif_path).split('.')[0].lower(),
            "resolution": res,
            "pH": float(ph) if ph else None,
            "temperature": float(temp) if temp else None,
            "details": detail.strip('"').strip("'") if detail else ""
        }
    except Exception as e:
        print(f"[!] Error parsing {cif_path}: {e}")
        return None

# --- Fetch protein sequence (FASTA) ---
def fetch_fasta(pdb_id):
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}"
    try:
        r = requests.get(url, timeout=10)
        if r.ok:
            records = list(SeqIO.parse(StringIO(r.text), "fasta"))
            return str(records[0].seq) if records else ""
    except Exception as e:
        print(f"[!] Failed to fetch FASTA for {pdb_id}: {e}")
    return ""

# --- Build dataset ---
def build_dataset(download_dir, max_files=1000, out_csv="clean_crystallization_dataset.csv"):
    if not os.path.exists(PDB_ID_FILE):
        raise FileNotFoundError(f"{PDB_ID_FILE} not found. Run fetch_pdb_ids.py first.")

    with open(PDB_ID_FILE) as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    pdb_ids = pdb_ids[:max_files]
    all_data = []

    for pdb_id in tqdm(pdb_ids, desc="Processing PDBs"):
        cif_path = os.path.join(download_dir, f"{pdb_id.lower()}.cif")
        if not os.path.exists(cif_path):
            downloaded = download_cif(pdb_id, download_dir)
            if not downloaded:
                continue

        metadata = extract_metadata(cif_path)
        if metadata and metadata.get("resolution", 99) < 4.0:
            seq = fetch_fasta(pdb_id)
            if seq and len(seq) > 100:
                metadata["sequence"] = seq
                all_data.append(metadata)
            else:
                print(f"[!] No sequence or too short for {pdb_id}")
        else:
            print(f"[!] Skipping {pdb_id}: invalid resolution or metadata")

    df = pd.DataFrame(all_data)
    print("Final columns:", df.columns.tolist())
    missing = [col for col in ["sequence", "pH", "resolution"] if col not in df.columns]
    if missing:
        print(f"[!] Missing expected columns: {missing}")
    else:
        df.dropna(subset=["sequence", "pH", "resolution"], inplace=True)

    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} entries to {out_csv}")

if __name__ == "__main__":
    build_dataset(DOWNLOAD_DIR, max_files=1000)
