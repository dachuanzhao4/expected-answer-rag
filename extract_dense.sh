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
        .answer_candidate_constrained_template_only["ndcg@10"],
        .concat_query_answer_candidate_constrained_template["ndcg@10"],
        .masked_expected_answer_only["ndcg@10"]
    ] | @csv' $file
}

extract outputs/nq_10_dense_run.json
extract outputs/nq_10_dense_cf_run.json
extract outputs/scifact_10_dense_run.json
extract outputs/scifact_10_dense_cf_run.json
extract outputs/hotpotqa_10_dense_run.json
extract outputs/hotpotqa_10_dense_cf_run.json
