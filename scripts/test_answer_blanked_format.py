from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from expected_answer_rag.answer_blanked import (  # noqa: E402
    REFUSAL_RE,
    build_answer_blanked_query2doc,
    build_lf_er_package,
    build_llm_lf_er_package,
    build_relation_keyword_query,
    infer_query_intent,
    sanitize_answer_blanked_query2doc,
    validate_answer_blanked_format,
)


def main() -> None:
    cases = [
        "who sings \"Love Will Keep Us Alive\" by the Eagles",
        "how many episodes are in Entity_2J0504 fire season 4",
        "where was Marie Curie born",
        "when does Work_AA0074 come out",
        "who is the leader of the SolacePoint0639 pc party",
        "what is the meaning of life",
        "how many Entity_LG1386 Entity_MP1451 games has the Entity_GL0390 played in",
        "Entity_RK1775 Entity_6E1276 belongs to which part of Entity_BH1441",
    ]
    rows = []
    for query in cases:
        skeleton = build_answer_blanked_query2doc(query)
        validation = validate_answer_blanked_format(query, skeleton)
        assert validation.ok, (query, skeleton, validation.issues)
        assert not REFUSAL_RE.search(skeleton), skeleton
        relation_query = build_relation_keyword_query(query)
        assert relation_query.strip(), query
        rows.append(
            {
                "query": query,
                "answer_type": infer_query_intent(query).answer_type,
                "skeleton": skeleton,
                "relation_query": relation_query,
            }
        )

    assert not REFUSAL_RE.search("who recorded i can't help falling in love with you")
    assert REFUSAL_RE.search("I can't find reliable information for this query.")

    sanitized = sanitize_answer_blanked_query2doc(
        "where was Marie Curie born",
        "Marie Curie was born in Warsaw and later worked in Paris.",
    )
    assert "[LOCATION]" in sanitized, sanitized
    assert "Warsaw" not in sanitized and "Paris" not in sanitized, sanitized
    assert validate_answer_blanked_format("where was Marie Curie born", sanitized).ok, sanitized

    sanitized = sanitize_answer_blanked_query2doc(
        "who sings \"Love Will Keep Us Alive\" by the Eagles",
        "Timothy B. Schmit sings Love Will Keep Us Alive by the Eagles.",
    )
    assert "[PERSON]" in sanitized, sanitized
    assert "Timothy" not in sanitized and "Schmit" not in sanitized, sanitized
    assert "Love Will Keep Us Alive" in sanitized and "Eagles" in sanitized, sanitized
    assert validate_answer_blanked_format("who sings \"Love Will Keep Us Alive\" by the Eagles", sanitized).ok

    sanitized = sanitize_answer_blanked_query2doc(
        "how many episodes are in Entity_2J0504 fire season 4",
        "Entity_2J0504 fire season 4 has 23 episodes and aired in 2015.",
    )
    assert "[NUMBER]" in sanitized, sanitized
    assert "23" not in sanitized and "2015" not in sanitized, sanitized
    assert "season 4" in sanitized and "Entity_2J0504" in sanitized, sanitized
    assert validate_answer_blanked_format("how many episodes are in Entity_2J0504 fire season 4", sanitized).ok

    sanitized = sanitize_answer_blanked_query2doc(
        "when does Work_AA0074 come out",
        "I can't find reliable information for Work_AA0074 because it looks like a placeholder.",
    )
    assert "[DATE]" in sanitized, sanitized
    assert "can't find" not in sanitized.lower() and "placeholder" not in sanitized.lower(), sanitized
    assert "Work_AA0074" in sanitized, sanitized
    assert validate_answer_blanked_format("when does Work_AA0074 come out", sanitized).ok

    invalid = validate_answer_blanked_format("where was Marie Curie born", "Marie Curie was born in Warsaw.")
    assert not invalid.ok and any(issue.startswith("missing_slot") for issue in invalid.issues), invalid

    wrong_slot = sanitize_answer_blanked_query2doc(
        "who sings \"Love Will Keep Us Alive\" by the Eagles",
        "[ENTITY] sings Love Will Keep Us Alive by the Eagles.",
    )
    assert "[PERSON]" in wrong_slot, wrong_slot
    assert validate_answer_blanked_format("who sings \"Love Will Keep Us Alive\" by the Eagles", wrong_slot).ok

    lf_er_cases = {
        "who sings love will keep us alive by Entity_LG0842": {
            "relation": "song_performer",
            "slot": "[PERSON]",
            "anchors": ["love will keep us alive", "Entity_LG0842"],
        },
        "who sings Entity_930927 book i wanna be like you": {
            "relation": "song_performer",
            "slot": "[PERSON]",
            "anchors": ["Entity_930927 book i wanna be like you"],
        },
        "how many episodes are in Entity_2J0504 fire season 4": {
            "relation": "episode_count",
            "slot": "[NUMBER]",
            "anchors": ["Entity_2J0504 fire", "4"],
        },
        "who plays V in Entity_NW1004 is the new black": {
            "relation": "cast_character",
            "slot": "[PERSON]",
            "anchors": ["V", "Entity_NW1004 is the new black"],
        },
        "who plays v on LumenGate1004 is the new black": {
            "relation": "cast_character",
            "slot": "[PERSON]",
            "anchors": ["v", "LumenGate1004 is the new black"],
        },
        "when does Work_AA0074 come out": {
            "relation": "release_date",
            "slot": "[DATE]",
            "anchors": ["Work_AA0074"],
        },
        "where was Marie Curie born": {
            "relation": "birthplace",
            "slot": "[LOCATION]",
            "anchors": ["Marie Curie"],
        },
        "where did the ashes from ash Entity_570251 originate": {
            "relation": "origin_location",
            "slot": "[LOCATION]",
            "anchors": ["ashes from ash Entity_570251"],
        },
        "what is the purpose of a Entity_8U1534 brake": {
            "relation": "purpose_function",
            "slot": "[ENTITY]",
            "anchors": ["Entity_8U1534 brake"],
        },
        "who did Entity_CQ1026 Entity_J21608 belong to before the u.s": {
            "relation": "previous_owner",
            "slot": "[PERSON]",
            "anchors": ["Entity_CQ1026 Entity_J21608", "u.s"],
        },
        "where are they building the new Entity_2J0664 Entity_4Y0698": {
            "relation": "construction_location",
            "slot": "[LOCATION]",
            "anchors": ["Entity_2J0664 Entity_4Y0698"],
        },
        "where is the setting for Entity_LG0778 and the Entity_LG1162": {
            "relation": "setting_location",
            "slot": "[LOCATION]",
            "anchors": ["Entity_LG0778 and the Entity_LG1162"],
        },
        "when did the us take over Entity_K91641 island": {
            "relation": "takeover_date",
            "slot": "[DATE]",
            "anchors": ["us", "Entity_K91641 island"],
        },
        "when were 7 books removed from the Entity_P51165": {
            "relation": "removal_date",
            "slot": "[DATE]",
            "anchors": ["7 books", "Entity_P51165"],
        },
        "Entity_RK1775 Entity_6E1276 belongs to which part of Entity_BH1441": {
            "relation": "part_of",
            "slot": "[ENTITY]",
            "anchors": ["Entity_RK1775 Entity_6E1276", "Entity_BH1441"],
        },
        "how many Entity_LG1386 Entity_MP1451 games has the Entity_GL0390 played in": {
            "relation": "games_played_count",
            "slot": "[NUMBER]",
            "anchors": ["Entity_LG1386 Entity_MP1451", "Entity_GL0390"],
        },
        "how long Entity_E61348 minister stay in office canada": {
            "relation": "tenure_length",
            "slot": "[NUMBER]",
            "anchors": ["Entity_E61348 minister", "canada"],
        },
        "who is Entity_P50845 on days of our lives": {
            "relation": "generic_relation",
            "slot": "[PERSON]",
            "anchors": ["Entity_P50845"],
        },
        "who performed the first c section in Number_NW0044": {
            "relation": "procedure_performer",
            "slot": "[PERSON]",
            "anchors": ["first c section in Number_NW0044"],
        },
        "how many seasons of prison break are on Entity_ZB0631": {
            "relation": "season_availability_count",
            "slot": "[NUMBER]",
            "anchors": ["prison break", "Entity_ZB0631"],
        },
    }
    for query, expected in lf_er_cases.items():
        package = build_lf_er_package(query)
        package_dict = package.to_dict()
        assert package.validation.ok, (query, package.validation.issues, package_dict)
        assert package.relation_name == expected["relation"], package_dict
        assert package.answer_slot == expected["slot"], package_dict
        views = package_dict["retrieval_views"]
        assert set(views) == {
            "anchor_view",
            "relation_keyword_view",
            "evidence_forward_view",
            "evidence_inverse_view",
            "slotless_evidence_view",
            "bm25_field_view",
            "dense_natural_view",
            "dense_safe_view",
            "dense_safe_expansion_view",
            "template_expansion_view",
            "corpus_style_view",
        }, views
        assert expected["slot"] in views["evidence_forward_view"], views
        assert expected["slot"] in views["evidence_inverse_view"], views
        assert "[" not in views["slotless_evidence_view"], views
        assert "[" not in views["bm25_field_view"], views
        assert "[" not in views["dense_natural_view"], views
        assert "[" not in views["dense_safe_view"], views
        assert "[" not in views["dense_safe_expansion_view"], views
        assert "[" not in views["template_expansion_view"], views
        assert "[" not in views["corpus_style_view"], views
        assert "generic relation" not in views["dense_safe_view"].lower(), views
        assert "requested fact" not in views["dense_safe_view"].lower(), views
        noisy_terms = [
            "relevant fact",
            "requested information",
            "associated with query relation",
        ]
        for noisy in noisy_terms:
            assert noisy not in views["corpus_style_view"].lower(), (query, views["corpus_style_view"])
        for view_name, view_text in views.items():
            if view_name == "dense_safe_expansion_view":
                continue
            assert query.lower() in str(view_text).lower(), (query, view_name, view_text)
        combined = " ".join(str(text) for text in views.values())
        assert not REFUSAL_RE.search(combined), combined
        for anchor in expected["anchors"]:
            assert anchor.lower() in combined.lower(), (query, anchor, combined)

    procedure = build_lf_er_package("who performed the first c section in Number_NW0044").to_dict()
    procedure_views = procedure["retrieval_views"]
    assert "singer" not in procedure_views["dense_safe_view"].lower(), procedure_views
    assert "vocals" not in procedure_views["dense_safe_view"].lower(), procedure_views
    assert "surgery" in procedure_views["dense_safe_view"].lower(), procedure_views

    generic = build_lf_er_package("who is Entity_P50845 on days of our lives").to_dict()
    generic_views = generic["retrieval_views"]
    assert generic["relation_frame"]["confidence"] == "low", generic
    assert "profile" not in generic_views["dense_safe_view"].lower(), generic_views
    assert "overview" not in generic_views["dense_safe_view"].lower(), generic_views

    llm_payload = json.dumps(
        {
            "answer_type": "LOCATION",
            "anchors": [
                {"text": "Marie Curie", "role": "person"},
                {"text": "Warsaw", "role": "hallucinated_answer"},
            ],
            "relation": {
                "name": "birthplace_lookup",
                "confidence": "high",
                "safe_cues": ["birthplace", "born in", "Warsaw", "warsaw"],
            },
            "must_keep_terms": ["Marie Curie", "born", "Warsaw", "warsaw"],
            "safe_expansion_terms": ["birthplace", "born in", "Warsaw", "warsaw"],
        }
    )
    llm_package = build_llm_lf_er_package("where was Marie Curie born", llm_payload).to_dict()
    llm_views = llm_package["retrieval_views"]
    assert llm_package["validation"]["ok"], llm_package
    assert "Warsaw" not in " ".join(llm_views.values()), llm_views
    assert "warsaw" not in " ".join(llm_views.values()).lower(), llm_views
    assert "Marie Curie" in " ".join(llm_views.values()), llm_views
    assert "where was Marie Curie born" in llm_views["llm_dense_view"], llm_views
    assert not any("[" in text for text in llm_views.values()), llm_views

    private_payload = json.dumps(
        {
            "answer_type": "DATE",
            "anchors": [{"text": "Work_AA0074", "role": "work_title"}],
            "relation": {
                "name": "release_timing",
                "confidence": "high",
                "safe_cues": ["release date", "premiere", "came out"],
            },
            "must_keep_terms": ["Work_AA0074", "come out"],
            "safe_expansion_terms": ["release date", "premiere", "came out"],
        }
    )
    private_package = build_llm_lf_er_package("when does Work_AA0074 come out", private_payload).to_dict()
    private_views = private_package["retrieval_views"]
    assert private_package["validation"]["ok"], private_package
    assert "placeholder" not in " ".join(private_views.values()).lower(), private_views
    assert "Work_AA0074" in " ".join(private_views.values()), private_views
    assert private_package["metadata"]["relation_class"] == "release_time", private_package
    assert private_package["metadata"]["retrieval_policy"] == "anchor_plus_one_cue", private_package

    llm_v2_payload = json.dumps(
        {
            "answer_type": "LOCATION",
            "anchors": [
                {"text": "Marie Curie", "role": "person", "importance": "primary"},
                {"text": "Curie", "role": "person", "importance": "support"},
                {"text": "Warsaw", "role": "location", "importance": "support"},
            ],
            "query_focus_terms": ["Marie Curie", "born", "Warsaw"],
            "relation_class": "location",
            "relation_confidence": "high",
            "retrieval_policy": "anchor_plus_one_cue",
            "safe_expansion_terms": ["Warsaw", "Nobel Prize"],
        }
    )
    llm_v2_package = build_llm_lf_er_package("where was Marie Curie born", llm_v2_payload, version="v2").to_dict()
    llm_v2_views = llm_v2_package["retrieval_views"]
    assert llm_v2_package["validation"]["ok"], llm_v2_package
    assert "Warsaw" not in " ".join(llm_v2_views.values()), llm_v2_views
    assert "nobel" not in " ".join(llm_v2_views.values()).lower(), llm_v2_views
    assert "Marie Curie" in " ".join(llm_v2_views.values()), llm_v2_views
    assert llm_v2_package["metadata"]["relation_class"] == "location", llm_v2_package
    assert "ignored_freeform_safe_terms_v2" in llm_v2_package["metadata"]["sanitizer_issues"], llm_v2_package

    malformed_package = build_llm_lf_er_package("who is Entity_P50845 on days of our lives", "not json").to_dict()
    assert malformed_package["validation"]["ok"], malformed_package
    assert malformed_package["metadata"]["fallback_used"], malformed_package

    print(json.dumps({"ok": True, "cases": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
