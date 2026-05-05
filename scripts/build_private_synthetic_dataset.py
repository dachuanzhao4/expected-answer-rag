from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from expected_answer_rag.datasets import Document, Query, RetrievalDataset, export_local_dataset


PROJECTS = [
    "Aster Vale",
    "Boreal Ledger",
    "Cedar Loom",
    "Delta Fen",
    "Echelon Quay",
    "Frost Meridian",
    "Granite Orchard",
    "Harbor Sable",
]
OWNERS = [
    "Mira Voss",
    "Jonas Vale",
    "Priya Renn",
    "Caleb Orin",
    "Nadia Sol",
    "Theo Marr",
    "Iris Keel",
    "Owen Nyx",
]
LOCATIONS = ["Vault-14", "Node-27", "Depot-33", "Ring-48", "Stack-52", "Cell-68"]
VALUES = ["2029-Q1", "2029-Q2", "2030-Q3", "2031-Q4", "Tier-7", "Tier-9"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fictional private-style retrieval dataset.")
    parser.add_argument("--output", default="outputs_pi/private_synthetic")
    parser.add_argument("--num-records", type=int, default=120)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    corpus = []
    queries = []
    qrels = {}
    for idx in range(args.num_records):
        project = f"{rng.choice(PROJECTS)}-{1000 + idx}"
        owner = rng.choice(OWNERS)
        location = rng.choice(LOCATIONS)
        value = rng.choice(VALUES)
        incident = f"INC-{70000 + idx}"
        doc_id = f"d{idx:04d}"
        query_id = f"q{idx:04d}"
        text = (
            f"Internal operations note for {project}. "
            f"The accountable owner for incident {incident} is {owner}. "
            f"The asset is staged in {location}, and its renewal marker is {value}. "
            f"Escalation should cite the project code and incident identifier exactly."
        )
        corpus.append(Document(doc_id=doc_id, title=f"{project} operations note", text=text))
        queries.append(
            Query(
                query_id=query_id,
                text=f"Who is the accountable owner for incident {incident} in project {project}?",
                answers=(owner,),
                supporting_doc_ids=(doc_id,),
            )
        )
        qrels[query_id] = {doc_id: 1}
    dataset = RetrievalDataset(
        name="private_synthetic",
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        metadata={
            "dataset_type": "synthetic_private",
            "answer_metadata": True,
            "seed": args.seed,
            "num_records": args.num_records,
        },
    )
    output = export_local_dataset(dataset, args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
