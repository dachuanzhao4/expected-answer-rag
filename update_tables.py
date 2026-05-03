import json
import re

keys = [
    "query_only",
    "raw_expected_answer_only",
    "hyde_doc_only",
    "query2doc_concat",
    "generative_relevance_feedback_concat",
    "corpus_steered_expansion_concat",
    "masked_expected_answer_only",
    "random_span_masking",
    "entity_only_masking",
    "generic_mask_slot",
    "wrong_answer_injection",
    "answer_candidate_constrained_template_only",
    "concat_query_answer_candidate_constrained_template"
]

names = [
    "query_only",
    "raw_expected_answer_only",
    "hyde_doc_only",
    "query2doc_concat",
    "generative_relevance_feedback",
    "corpus_steered_expansion_concat",
    "masked_expected_answer_only",
    "random_span_masking",
    "entity_only_masking",
    "generic_mask_slot",
    "wrong_answer_injection",
    "answer_candidate_constrained_template",
    "concat_query_answer_constrained"
]

def read_metrics(path):
    with open(path) as f:
        d = json.load(f)["metrics"]
        return [d.get(k, {}).get("ndcg@10", 0.0) for k in keys]

def format_table(header, rows_data):
    table = header + "\n"
    for i, n in enumerate(names):
        row = f"| `{n}` |"
        for dataset_metrics in rows_data:
            row += f" {dataset_metrics[i]:.3f} |"
        table += row + "\n"
    return table

nq_bm25_pub = read_metrics("outputs/nq_100_run.json")
nq_bm25_cf = read_metrics("outputs/nq_100_cf_run.json")
sc_bm25_pub = read_metrics("outputs/scifact_100_run.json")
sc_bm25_cf = read_metrics("outputs/scifact_100_cf_run.json")
hq_bm25_pub = read_metrics("outputs/hotpotqa_100_run.json")
hq_bm25_cf = read_metrics("outputs/hotpotqa_100_cf_run.json")

bm25_header = "| Method | NQ | SciFact | HotpotQA | NQ (CF) | SciFact (CF) | HotpotQA (CF) |\n|---|:---:|:---:|:---:|:---:|:---:|:---:|"
bm25_table = format_table(bm25_header, [nq_bm25_pub, sc_bm25_pub, hq_bm25_pub, nq_bm25_cf, sc_bm25_cf, hq_bm25_cf])

nq_dense_pub = read_metrics("outputs/nq_100_dense_run.json")
nq_dense_cf = read_metrics("outputs/nq_100_cf_dense_run.json")
sc_dense_pub = read_metrics("outputs/scifact_100_dense_run.json")
sc_dense_cf = read_metrics("outputs/scifact_100_cf_dense_run.json")
hq_dense_pub = read_metrics("outputs/hotpotqa_100_dense_run.json")
hq_dense_cf = read_metrics("outputs/hotpotqa_100_cf_dense_run.json")

dense_header = "| Method | NQ | SciFact | HotpotQA | NQ (CF) | SciFact (CF) | HotpotQA (CF) |\n|---|:---:|:---:|:---:|:---:|:---:|:---:|"
dense_table = format_table(dense_header, [nq_dense_pub, sc_dense_pub, hq_dense_pub, nq_dense_cf, sc_dense_cf, hq_dense_cf])

nq_ev = read_metrics("outputs/nq_100_cf_ev_run.json")
sc_ev = read_metrics("outputs/scifact_100_cf_ev_run.json")
hq_ev = read_metrics("outputs/hotpotqa_100_cf_ev_run.json")

ev_header = "| Method | NQ (CF E+V) | SciFact (CF E+V) | HotpotQA (CF E+V) |\n|---|:---:|:---:|:---:|"
ev_table = format_table(ev_header, [nq_ev, sc_ev, hq_ev])

with open("docs/stress_test_findings.md", "r") as f:
    text = f.read()

# Update N notation
text = text.replace("N=10", "N=100")

# Replace tables
text = re.sub(
    r"\| Method \| NQ \| SciFact \| HotpotQA \| NQ \(CF\) \| SciFact \(CF\) \| HotpotQA \(CF\) \|\n\|---\|:---:\|:---:\|:---:\|:---:\|:---:\|:---:\|\n(?:\|.*\|\n)+",
    bm25_table,
    text,
    count=1
)

dense_section_marker = "### C. Dense Retriever Stress Testing"
parts = text.split(dense_section_marker)
if len(parts) == 2:
    parts[1] = re.sub(
        r"\| Method \| NQ \| SciFact \| HotpotQA \| NQ \(CF\) \| SciFact \(CF\) \| HotpotQA \(CF\) \|\n\|---\|:---:\|:---:\|:---:\|:---:\|:---:\|:---:\|\n(?:\|.*\|\n)+",
        dense_table,
        parts[1],
        count=1
    )
    text = dense_section_marker.join(parts)

ev_section_marker = "### D. Entity and Value Counterfactual Ablation"
parts = text.split(ev_section_marker)
if len(parts) == 2:
    parts[1] = re.sub(
        r"\| Method \| NQ \(CF E\+V\) \| SciFact \(CF E\+V\) \| HotpotQA \(CF E\+V\) \|\n\|---\|:---:\|:---:\|:---:\|\n(?:\|.*\|\n)+",
        ev_table,
        parts[1],
        count=1
    )
    text = ev_section_marker.join(parts)

with open("docs/stress_test_findings.md", "w") as f:
    f.write(text)

