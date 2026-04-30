from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from itertools import zip_longest
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels."""
    full_sequences = [
        tokenizer.encode(p, add_special_tokens=False) + tokenizer.encode(o, add_special_tokens=False)
        for p, o in zip(prompt_strs, output_strs, strict=True)
    ]
    max_len = max(len(seq) - 1 for seq in full_sequences)

    input_ids_list, labels_list, mask_list = [], [], []
    for prompt, full_seq in zip(prompt_strs, full_sequences, strict=True):
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        response_len = len(full_seq) - prompt_len
        seq_len = len(full_seq) - 1
        pad = [tokenizer.pad_token_id] * (max_len - seq_len)

        input_ids_list.append(full_seq[:-1] + pad)
        labels_list.append(full_seq[1:] + pad)
        mask_list.append(
            [False] * (prompt_len - 1) + [True] * response_len + [False] * (max_len - seq_len)
        )

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "response_mask": torch.tensor(mask_list, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropies over the vocabulary dimension."""
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(torch.exp(log_probs) * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    attention_mask: Tensor | None = None,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples."""
    kwargs = {} if attention_mask is None else {"attention_mask": attention_mask}
    logits = model(input_ids, **kwargs).logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    out: dict[str, Tensor] = {"log_probs": token_log_probs}
    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)
    return out


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over masked elements and normalize by the provided constant."""
    return (tensor * mask).sum(dim=dim) / normalize_constant


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalized advantages for GRPO."""
    reward_dicts = [
        reward_fn(r, g) for r, g in zip(rollout_responses, repeated_ground_truths, strict=True)
    ]
    raw_rewards = torch.tensor(
        [d["reward"] for d in reward_dicts], dtype=torch.float32
    )
    grouped = raw_rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        normalized = centered / (grouped.std(dim=1, keepdim=True, unbiased=False) + advantage_eps)
    else:
        normalized = centered

    mean_format = sum(d.get("format_reward", 0.0) for d in reward_dicts) / len(reward_dicts)
    mean_answer = sum(d.get("answer_reward", 0.0) for d in reward_dicts) / len(reward_dicts)
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "mean_format_reward": mean_format,
        "mean_answer_reward": mean_answer,
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }
    return normalized.reshape(-1), raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss."""
    ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)
    broadcast_adv = advantages.expand_as(policy_log_probs)
    loss = -torch.minimum(ratios * broadcast_adv, clipped_ratios * broadcast_adv)
    is_clipped = (clipped_ratios != ratios)
    metadata = {"clip_fraction": is_clipped.float().mean()}
    return loss, metadata


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate a single GRPO microbatch loss."""
    per_token_loss, metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    mask_f = response_mask.to(per_token_loss.dtype)
    per_example_loss = (per_token_loss * mask_f).sum(dim=1) / response_mask.sum(dim=1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create serializable generation logs for debugging training runs."""
    return [
        {
            "prompt": p,
            "response": r,
            "ground_truth": g,
            "reward_info": ri,
            "token_entropy": te,
        }
        for p, r, g, ri, te in zip_longest(
            prompts, responses, ground_truths, reward_infos, token_entropies or []
        )
    ]


def train_grpo(
    model,
    tokenizer,
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    reward_fn: Callable[[str, str], dict[str, float]],
    prompt_template: str,
    device: torch.device,
    n_grpo_steps: int = 50,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 32,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 256,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 16,
    cliprange: float = 1.0,
    normalize_by_std: bool = True,
    val_every: int = 5,
    val_size: int = 256,
    vllm_model=None,
) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5."""
    from .prompts import COT_PROMPT_TEMPLATE
    from .rewards import answer_tag_reward_fn

    assert train_batch_size % gradient_accumulation_steps == 0
    assert rollout_batch_size % group_size == 0
    assert train_batch_size >= group_size

    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    history: dict[str, list] = {
        "step": [], "loss": [], "grad_norm": [],
        "mean_reward": [], "mean_format_reward": [], "mean_answer_reward": [],
        "val_reward": [], "val_step": [],
    }

    model.train()
    optimizer.zero_grad()

    for grpo_step in range(1, n_grpo_steps + 1):
        # 1. Sample question batch
        batch_examples = random.sample(train_examples, n_prompts_per_rollout_batch)
        questions = [ex["question"] for ex in batch_examples]
        ground_truths = [ex["answer"] for ex in batch_examples]

        # Build prompts and repeat each G times for the rollout
        prompts = [prompt_template.format(question=q) for q in questions]
        repeated_prompts = [p for p in prompts for _ in range(group_size)]
        repeated_gts = [g for g in ground_truths for _ in range(group_size)]

        # 2. Generate rollouts
        model.eval()
        rollout_responses = _generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=repeated_prompts,
            device=device,
            max_new_tokens=sampling_max_tokens,
            temperature=sampling_temperature,
            min_new_tokens=sampling_min_tokens,
            stop_string="</answer>",
            vllm_model=vllm_model,
        )
        model.train()

        # 3. Compute rewards and advantages
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )

        # 4. Cache old log-probs (no grad, from the policy that generated the rollouts)
        all_tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
        input_ids_all = all_tokenized["input_ids"].to(device)
        labels_all = all_tokenized["labels"].to(device)
        response_mask_all = all_tokenized["response_mask"].to(device)
        # Explicit attention mask so the model can distinguish padding from EOS
        # (needed when pad_token_id == eos_token_id, e.g. Qwen models)
        attention_mask_all = (input_ids_all != tokenizer.pad_token_id).long()

        with torch.no_grad():
            old_lp_out = get_response_log_probs(
                model=model,
                input_ids=input_ids_all,
                labels=labels_all,
                attention_mask=attention_mask_all,
                return_token_entropy=False,
            )
        old_log_probs_all = old_lp_out["log_probs"].detach()

        # 5. Training epochs over the rollout batch
        total_loss = 0.0
        microbatch_count = 0
        for _epoch in range(epochs_per_rollout_batch):
            indices = list(range(rollout_batch_size))
            for mb_start in range(0, rollout_batch_size, micro_train_batch_size):
                mb_idx = indices[mb_start: mb_start + micro_train_batch_size]

                mb_input_ids = input_ids_all[mb_idx]
                mb_labels = labels_all[mb_idx]
                mb_mask = response_mask_all[mb_idx]
                mb_advantages = advantages[mb_idx].unsqueeze(1).to(device)
                mb_old_log_probs = old_log_probs_all[mb_idx]
                mb_attention_mask = attention_mask_all[mb_idx]

                lp_out = get_response_log_probs(
                    model=model,
                    input_ids=mb_input_ids,
                    labels=mb_labels,
                    attention_mask=mb_attention_mask,
                    return_token_entropy=False,
                )
                mb_policy_log_probs = lp_out["log_probs"]

                loss, _meta = grpo_microbatch_train_step(
                    policy_log_probs=mb_policy_log_probs,
                    response_mask=mb_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs,
                    cliprange=cliprange,
                )
                total_loss += loss.item()
                microbatch_count += 1

                if microbatch_count % gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                    optimizer.step()
                    optimizer.zero_grad()

        mean_loss = total_loss / max(microbatch_count, 1)
        history["step"].append(grpo_step)
        history["loss"].append(mean_loss)
        history["grad_norm"].append(grad_norm if microbatch_count > 0 else 0.0)
        history["mean_reward"].append(reward_meta["mean_reward"])
        history["mean_format_reward"].append(reward_meta["mean_format_reward"])
        history["mean_answer_reward"].append(reward_meta["mean_answer_reward"])

        print(
            f"[step {grpo_step:3d}] loss={mean_loss:.4f}  "
            f"reward={reward_meta['mean_reward']:.3f}  "
            f"format={reward_meta['mean_format_reward']:.3f}  "
            f"answer={reward_meta['mean_answer_reward']:.3f}  "
            f"grad_norm={history['grad_norm'][-1]:.3f}"
        )

        # 6. Periodic validation
        if grpo_step % val_every == 0:
            val_reward = _evaluate_val(
                model=model,
                tokenizer=tokenizer,
                val_examples=val_examples[:val_size],
                reward_fn=reward_fn,
                prompt_template=prompt_template,
                device=device,
                max_new_tokens=sampling_max_tokens,
                vllm_model=vllm_model,
            )
            history["val_reward"].append(val_reward)
            history["val_step"].append(grpo_step)
            print(f"  [val  step {grpo_step:3d}] val_reward={val_reward:.3f}")

    return history


def _generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    min_new_tokens: int,
    stop_string: str,
    vllm_model=None,
) -> list[str]:
    """Generate one response per prompt using vLLM if available, else HuggingFace."""
    if vllm_model is not None:
        try:
            from vllm import SamplingParams
            sp = SamplingParams(
                temperature=temperature,
                top_p=1.0,
                min_tokens=min_new_tokens,
                max_tokens=max_new_tokens,
                stop=[stop_string],
                include_stop_str_in_output=True,
            )
            outputs = vllm_model.generate(prompts, sp)
            return [o.outputs[0].text for o in outputs]
        except Exception:
            pass  # fall through to HF generation

    responses = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Truncate at stop string
        if stop_string in text:
            text = text[: text.index(stop_string) + len(stop_string)]
        responses.append(text)
    return responses


def _evaluate_val(
    model,
    tokenizer,
    val_examples: list[dict],
    reward_fn,
    prompt_template: str,
    device: torch.device,
    max_new_tokens: int,
    vllm_model=None,
) -> float:
    prompts = [prompt_template.format(question=ex["question"]) for ex in val_examples]
    ground_truths = [ex["answer"] for ex in val_examples]
    responses = _generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        min_new_tokens=1,
        stop_string="</answer>",
        vllm_model=vllm_model,
    )
    rewards = [reward_fn(r, g)["reward"] for r, g in zip(responses, ground_truths)]
    return sum(rewards) / len(rewards) if rewards else 0.0
