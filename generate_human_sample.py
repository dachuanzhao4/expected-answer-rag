import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from expected_answer_rag.datasets import load_dataset
from expected_answer_rag.counterfactual import build_entity_counterfactual_dataset

def main():
    ds = load_dataset("hotpotqa", max_corpus=50, max_queries=50)
    
    # Entity CF
    cf_entity = build_entity_counterfactual_dataset(ds, alias_style="natural", include_values=False)
    
    # Entity + Value CF
    cf_ev = build_entity_counterfactual_dataset(ds, alias_style="natural", include_values=True)
    
    with open("docs/human_verification_sample.md", "w") as f:
        f.write("# Counterfactual Generation Human Verification Sample\n\n")
        f.write("This document provides a side-by-side comparison of the original text and the counterfactually generated text to verify that grammar, reasoning paths, and relations are preserved while entities are successfully obscured.\n\n")
        
        f.write("## 1. Document Renaming Examples\n\n")
        
        # Find 3 docs with at least one entity and one value
        found_docs = 0
        for i in range(len(ds.corpus)):
            doc_orig = ds.corpus[i]
            doc_entity = cf_entity.dataset.corpus[i]
            doc_ev = cf_ev.dataset.corpus[i]
            
            # Check if anything actually changed
            if doc_orig.text != doc_ev.text and found_docs < 3:
                f.write(f"### Document {doc_orig.doc_id}\n\n")
                f.write("**Original:**\n> " + doc_orig.text.replace("\n", "\n> ") + "\n\n")
                f.write("**Entity CF:**\n> " + doc_entity.text.replace("\n", "\n> ") + "\n\n")
                f.write("**Entity & Value CF:**\n> " + doc_ev.text.replace("\n", "\n> ") + "\n\n")
                f.write("---\n\n")
                found_docs += 1
            
        f.write("## 2. Query Renaming Examples\n\n")
        found_queries = 0
        for i in range(len(ds.queries)):
            q_orig = ds.queries[i]
            q_entity = cf_entity.dataset.queries[i]
            q_ev = cf_ev.dataset.queries[i]
            
            if q_orig.text != q_ev.text and found_queries < 3:
                f.write(f"### Query {q_orig.query_id}\n\n")
                f.write(f"- **Original:** {q_orig.text}\n")
                f.write(f"  - **Answers:** {q_orig.answers}\n")
                f.write(f"- **Entity CF:** {q_entity.text}\n")
                f.write(f"  - **Answers:** {q_entity.answers}\n")
                f.write(f"- **Entity & Value CF:** {q_ev.text}\n")
                f.write(f"  - **Answers:** {q_ev.answers}\n\n")
                found_queries += 1

if __name__ == "__main__":
    main()
