# Phase 1 — Library Quality Gate (adapter vs. base)

Gate-passing: **13/14** tasks (τ = margin ≥ 0.05). Eval = exact-match (classification) / ROUGE-L (generation) on the held-out test split; base vs. adapter on the *same* resident Mistral-7B via PEFT `disable_adapter`.

| task_num | task | rank | metric | base | adapter | margin | gate | s |
|---|---|---|---|---|---|---|---|---|
| task280 | task280_stereoset_classification_stere… | 43 | exact_match | 0.075 | 0.983 | +0.908 | PASS | 11 |
| task1391 | task1391_winogrande_easy_answer_genera… | 43 | exact_match | 0.100 | 0.908 | +0.808 | PASS | 16 |
| task512 | task512_twitter_emotion_classification | 16 | exact_match | 0.075 | 0.875 | +0.800 | PASS | 11 |
| task190 | task190_snli_classification | 43 | exact_match | 0.350 | 0.892 | +0.542 | PASS | 14 |
| task391 | task391_causal_relationship | 43 | exact_match | 0.317 | 0.833 | +0.517 | PASS | 10 |
| task843 | task843_financial_phrasebank_classific… | 16 | exact_match | 0.400 | 0.908 | +0.508 | PASS | 14 |
| task1344 | task1344_glue_entailment_classification | 16 | exact_match | 0.417 | 0.892 | +0.475 | PASS | 18 |
| task620 | task620_ohsumed_medical_subject_headin… | 43 | rougeL | 0.188 | 0.582 | +0.394 | PASS | 34 |
| task442 | task442_com_qa_paraphrase_question_gen… | 43 | rougeL | 0.503 | 0.727 | +0.224 | PASS | 26 |
| task1564 | task1564_triviaqa_answer_generation | 16 | exact_match | 0.182 | 0.364 | +0.182 | PASS | 4 |
| task290 | task290_tellmewhy_question_answerabili… | 43 | exact_match | 0.550 | 0.725 | +0.175 | PASS | 14 |
| task379 | task379_agnews_topic_classification | 16 | exact_match | 0.625 | 0.783 | +0.158 | PASS | 18 |
| task1342 | task1342_amazon_us_reviews_title | 43 | rougeL | 0.070 | 0.144 | +0.074 | PASS | 42 |
| task639 | task639_multi_woz_user_utterance_gener… | 16 | rougeL | 0.090 | 0.051 | -0.040 | FAIL | 9 |
