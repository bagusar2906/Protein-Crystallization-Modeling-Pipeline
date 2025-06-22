
import gemmi
import os
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from io import StringIO
import requests

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
    except:
        return None

def fetch_fasta(pdb_id):
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}"
    try:
        response = requests.get(url, timeout=10)
        if response.ok:
            records = list(SeqIO.parse(StringIO(response.text), "fasta"))
            return str(records[0].seq) if records else ""
    except:
        return ""

def build_dataset(mmcif_dir, out_csv="clean_crystallization_dataset.csv", max_files=1000):
    all_data = []
    processed = 0
    for root, dirs, files in os.walk(mmcif_dir):
        for f in tqdm(files):
            if not f.endswith(".cif"): continue
            pdb_id = f.split(".")[0]
            metadata = extract_metadata(os.path.join(root, f))
            if metadata and metadata["resolution"] < 4.0:
                seq = fetch_fasta(pdb_id)
                if seq and len(seq) > 100:
                    metadata["sequence"] = seq
                    all_data.append(metadata)
            processed += 1
            if processed >= max_files:
                break

    df = pd.DataFrame(all_data)
    df.dropna(subset=["sequence", "pH", "resolution"], inplace=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} entries to {out_csv}")

if __name__ == "__main__":
    build_dataset("pdb_mmCIF", max_files=1000)
