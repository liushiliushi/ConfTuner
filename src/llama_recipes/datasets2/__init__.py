# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from functools import partial

from llama_recipes.datasets2.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from llama_recipes.datasets2.gsm8k_dataset import get_gsm8k_dataset2, get_gsm8k_dataset_raw
from llama_recipes.datasets2.strategyqa_dataset import get_strategyqa
from llama_recipes.datasets2.trivia_qa import get_trivia_qa, get_trivia_qa_raw
from llama_recipes.datasets2.truthful_qa import get_truthful_qa, get_truthful_qa_raw, get_truthful_qa_reflection
from llama_recipes.datasets2.hotpot_qa import get_hotpot_qa, get_hotpot_qa_raw, get_hotpot_qa_reflection
DATASET_PREPROC = {
    "grammar_dataset": get_grammar_dataset,
    "gsm8k_dataset": get_gsm8k_dataset2,
    "strategy_qa": get_strategyqa,
    "trivia_qa": get_trivia_qa,
    "truthful_qa": get_truthful_qa,
    "hotpot_qa": get_hotpot_qa,
}

DATASET_RAW = {
    "gsm8k_dataset": get_gsm8k_dataset_raw,
    "strategy_qa": get_strategyqa,
    "trivia_qa": get_trivia_qa_raw,
    "truthful_qa": get_truthful_qa_raw,
    "hotpot_qa": get_hotpot_qa_raw,
}

DATASET_REFLECTION = {
    "truthful_qa": get_truthful_qa_reflection,
    "hotpot_qa": get_hotpot_qa_reflection,
}


