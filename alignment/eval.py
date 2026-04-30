from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]


def build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [prompt_template.format(question=ex["question"]) for ex in examples]


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
    ground_truths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate model outputs, score them, and return serializable evaluation artifacts."""
    outputs = vllm_model.generate(list(prompts), eval_sampling_params)
    responses = [o.outputs[0].text for o in outputs]

    reward_infos: list[dict[str, float]] = []
    if ground_truths is not None:
        for resp, gt in zip(responses, ground_truths, strict=True):
            reward_infos.append(reward_fn(resp, gt))
    else:
        for resp in responses:
            reward_infos.append(reward_fn(resp, ""))

    mean_reward = sum(r["reward"] for r in reward_infos) / len(reward_infos)
    mean_format = sum(r.get("format_reward", 0.0) for r in reward_infos) / len(reward_infos)
    mean_answer = sum(r.get("answer_reward", 0.0) for r in reward_infos) / len(reward_infos)

    return {
        "prompts": list(prompts),
        "responses": responses,
        "ground_truths": list(ground_truths) if ground_truths is not None else [],
        "reward_infos": reward_infos,
        "mean_reward": mean_reward,
        "mean_format_reward": mean_format,
        "mean_answer_reward": mean_answer,
    }


def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    """Serialize generations and scores for later analysis."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def run_direct_baseline(output_path: Path) -> None:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    from vllm import LLM, SamplingParams
    from .rewards import answer_tag_reward_fn

    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, DIRECT_PROMPT_TEMPLATE)
    ground_truths = [ex["answer"] for ex in examples]

    llm = LLM(model=DEFAULT_MODEL_NAME)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=answer_tag_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        ground_truths=ground_truths,
    )
    print(
        f"Direct baseline — mean_reward={results['mean_reward']:.3f}  "
        f"format={results['mean_format_reward']:.3f}  "
        f"answer={results['mean_answer_reward']:.3f}"
    )
    write_evaluation_results(results, Path(output_path))


def run_cot_baseline(output_path: Path) -> None:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    from vllm import LLM, SamplingParams
    from .rewards import answer_tag_reward_fn

    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, COT_PROMPT_TEMPLATE)
    ground_truths = [ex["answer"] for ex in examples]

    llm = LLM(model=DEFAULT_MODEL_NAME)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=answer_tag_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        ground_truths=ground_truths,
    )
    print(
        f"CoT baseline — mean_reward={results['mean_reward']:.3f}  "
        f"format={results['mean_format_reward']:.3f}  "
        f"answer={results['mean_answer_reward']:.3f}"
    )
    write_evaluation_results(results, Path(output_path))


def run_self_consistency_baseline(output_path: Path, k: int = 5) -> None:
    """Evaluate the self-consistency baseline from Section 3.2."""
    from vllm import LLM, SamplingParams
    from .rewards import answer_tag_reward_fn, majority_vote_tagged_answers

    examples = load_gsm8k_examples("test")
    prompts = build_prompts(examples, COT_PROMPT_TEMPLATE)
    ground_truths = [ex["answer"] for ex in examples]

    llm = LLM(model=DEFAULT_MODEL_NAME)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Expand: each prompt repeated K times
    repeated_prompts = [p for p in prompts for _ in range(k)]
    outputs = llm.generate(repeated_prompts, sampling_params)
    all_responses = [o.outputs[0].text for o in outputs]

    # Group into chunks of K and take majority vote
    voted_responses: list[str] = []
    grouped_responses: list[list[str]] = []
    for i in range(len(examples)):
        group = all_responses[i * k: (i + 1) * k]
        grouped_responses.append(group)
        voted_responses.append(majority_vote_tagged_answers(group) or "")

    reward_infos = [
        answer_tag_reward_fn(resp, gt)
        for resp, gt in zip(voted_responses, ground_truths, strict=True)
    ]
    mean_reward = sum(r["reward"] for r in reward_infos) / len(reward_infos)
    mean_format = sum(r.get("format_reward", 0.0) for r in reward_infos) / len(reward_infos)
    mean_answer = sum(r.get("answer_reward", 0.0) for r in reward_infos) / len(reward_infos)

    results = {
        "prompts": prompts,
        "grouped_responses": grouped_responses,
        "voted_responses": voted_responses,
        "ground_truths": ground_truths,
        "reward_infos": reward_infos,
        "mean_reward": mean_reward,
        "mean_format_reward": mean_format,
        "mean_answer_reward": mean_answer,
        "k": k,
    }
    print(
        f"Self-consistency (k={k}) — mean_reward={mean_reward:.3f}  "
        f"format={mean_format:.3f}  answer={mean_answer:.3f}"
    )
    write_evaluation_results(results, Path(output_path))


def get_prompt_template(use_cot: bool) -> str:
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE
