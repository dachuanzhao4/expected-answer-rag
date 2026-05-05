#!/usr/bin/env bash
set -euo pipefail

MAX_QUERIES="${MAX_QUERIES:-100}"
MAX_CORPUS="${MAX_CORPUS:-3000}"
OUT_DIR="${OUT_DIR:-outputs_pi/c${MAX_CORPUS}}"
METHODS="${METHODS:-query_only,raw_expected_answer_only,hyde_doc_only,query2doc_concat,concat_query_raw_expected,concat_query_answer_candidate_constrained_template,safe_rrf_v1,fawe_query2doc_beta0p25,fawe_safe_adaptive_beta}"
DATASETS="${DATASETS:-nq scifact hotpotqa}"

mkdir -p "$OUT_DIR/postprocess"

shopt -s nullglob
pi_records=( "$OUT_DIR"/*_records.jsonl )
legacy_records=( outputs/*_records.jsonl )
shopt -u nullglob

if [ "${#pi_records[@]}" -gt 0 ]; then
  conda run -n rag python scripts/collect_metric_error_bars.py \
    --records "${pi_records[@]}" \
    --methods "$METHODS" \
    --metrics ndcg@10,recall@20,recall@100,mrr@10 \
    --bootstrap-samples 1000 \
    --output-json "$OUT_DIR/postprocess/metric_error_bars.json" \
    --output-csv "$OUT_DIR/postprocess/metric_error_bars.csv"
fi

if [ "${#legacy_records[@]}" -gt 0 ]; then
  conda run -n rag python scripts/collect_metric_error_bars.py \
    --records "${legacy_records[@]}" \
    --methods "$METHODS" \
    --metrics ndcg@10,recall@20,recall@100,mrr@10 \
    --bootstrap-samples 1000 \
    --output-json "$OUT_DIR/postprocess/legacy_metric_error_bars.json" \
    --output-csv "$OUT_DIR/postprocess/legacy_metric_error_bars.csv"
fi

coverage_datasets="${DATASETS// /,}"
conda run -n rag python scripts/report_qrel_coverage.py \
  --datasets "$coverage_datasets" \
  --corpus-sizes "2000,${MAX_CORPUS}" \
  --max-queries "$MAX_QUERIES" \
  --cache-dir outputs/hf_cache \
  --output-json "$OUT_DIR/postprocess/qrel_coverage.json" \
  --output-csv "$OUT_DIR/postprocess/qrel_coverage.csv"

for ds in $DATASETS; do
  public_records="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}_bm25_records.jsonl"
  entity_records="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}_cf_bm25_records.jsonl"
  ev_records="${OUT_DIR}/${ds}_${MAX_QUERIES}_c${MAX_CORPUS}_cf_ev_bm25_records.jsonl"

  if [ ! -f "$public_records" ] || [ ! -f "$entity_records" ] || [ ! -f "$ev_records" ]; then
    echo "Skipping beta/content postprocess for $ds because one BM25 record file is missing."
    continue
  fi

  for objective in public average robust; do
    conda run -n rag python scripts/select_fawe_beta.py \
      --public-records "$public_records" \
      --entity-records "$entity_records" \
      --entity-value-records "$ev_records" \
      --family query2doc \
      --objective "$objective" \
      --metric ndcg@10 \
      --output "$OUT_DIR/postprocess/${ds}_heldout_beta_${objective}.json"
  done

  conda run -n rag python scripts/audit_expansion_content.py \
    --public-records "$public_records" \
    --entity-records "$entity_records" \
    --entity-value-records "$ev_records" \
    --metric ndcg@10 \
    --output "$OUT_DIR/postprocess/${ds}_expansion_content_audit.json" \
    --markdown-output "$OUT_DIR/postprocess/${ds}_expansion_content_audit.md"
done

conda run -n rag python scripts/plot_pi_figures.py \
  --input-dir "$OUT_DIR" \
  --output-dir "$OUT_DIR/figures"
