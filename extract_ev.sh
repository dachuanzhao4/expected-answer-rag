#!/bin/bash
extract() {
    file=$1
    echo "--- $file ---"
    jq -r '.metrics | [
        .query_only["ndcg@10"],
        .raw_expected_answer_only["ndcg@10"],
        .hyde_doc_only["ndcg@10"],
        .query2doc_concat["ndcg@10"],
        .generative_relevance_feedback_concat["ndcg@10"],
        .corpus_steered_expansion_concat["ndcg@10"],
        .masked_expected_answer_only["ndcg@10"],
        .random_span_masking["ndcg@10"],
        .entity_only_masking["ndcg@10"],
        .generic_mask_slot["ndcg@10"],
        .wrong_answer_injection["ndcg@10"],
        .answer_candidate_constrained_template_only["ndcg@10"]
    ] | @csv' $file
}

extract outputs/nq_10_cf_ev_run.json
extract outputs/scifact_10_cf_ev_run.json
extract outputs/hotpotqa_10_cf_ev_run.json
