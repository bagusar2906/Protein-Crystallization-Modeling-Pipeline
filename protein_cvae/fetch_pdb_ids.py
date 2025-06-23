
import requests

def fetch_pdb_ids(limit=10000):
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": "X-RAY DIFFRACTION"
            }
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": limit},
            "results_content_type": ["experimental"]
        }
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    response = requests.post(url, json=query)
    if not response.ok:
        raise Exception("Failed to fetch PDB IDs")

    data = response.json()
    pdb_ids = [entry["identifier"] for entry in data.get("result_set", [])]

    with open("pdb_ids.txt", "w") as f:
        for pdb_id in pdb_ids:
            f.write(pdb_id + "\n")

    print(f"Saved {len(pdb_ids)} PDB IDs to pdb_ids.txt")

if __name__ == "__main__":
    fetch_pdb_ids()
