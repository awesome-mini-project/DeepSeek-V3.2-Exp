import os
import json
import time
from argparse import ArgumentParser
from typing import List, Optional, Sequence
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs
import trace as ds_trace


def _resolve_config_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    base = Path(__file__).resolve().parent
    cand = base / p
    if cand.exists():
        return str(cand)
    return p


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    request_ids: Optional[Sequence[int]] = None,
    prefix_infos: Optional[Sequence[ds_trace.RequestPrefixInfo]] = None,
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    tracer = ds_trace.get_tracer()
    if tracer.enabled:
        if request_ids is None:
            request_ids = list(range(len(prompt_tokens)))
        tracer.set_batch(request_ids=request_ids, prefix_infos=prefix_infos)
    for cur_pos in range(min(prompt_lens), total_len):
        if tracer.enabled:
            tracer.set_step_timing(step_idx=prev_pos, step_wall_us=None)
            if tracer.cfg.sync_cuda_for_timing:
                torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if tracer.enabled:
            if tracer.cfg.sync_cuda_for_timing:
                torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            tracer.set_step_timing(step_idx=prev_pos, step_wall_us=(t1 - t0) // 1000)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    trace_enable: bool = False,
    trace_out: str = "",
    kv_block_size: int = 16,
    trace_store_scores: bool = False,
    trace_sample_rate: float = 1.0,
    trace_prefix_key_tokens: int = 256,
    trace_sync_cuda: bool = True,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    config = _resolve_config_path(config)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    if trace_enable or trace_out:
        if not trace_out:
            trace_out = str(os.path.join("outputs", f"trace_{int(time.time() * 1000)}"))
        cfg = ds_trace.TraceConfig(
            enabled=True,
            out_dir=trace_out,
            kv_block_size_tokens=int(kv_block_size),
            store_scores_topk=bool(trace_store_scores),
            store_selected_token_pos=True,
            sample_rate=float(trace_sample_rate),
            rank0_only=True,
            sync_cuda_for_timing=bool(trace_sync_cuda),
            prefix_cache_key_tokens=int(trace_prefix_key_tokens),
        )
        tracer = ds_trace.init_tracer(cfg)
        tracer.set_run_meta(run_name=os.path.basename(trace_out.rstrip("/")), dataset="interactive" if interactive else "file")
        # Expose bytes/token for logical KV fetch estimation.
        bytes_per_token = int((model.layers[0].attn.kv_cache.size(-1) * model.layers[0].attn.kv_cache.element_size()) +
                              (model.layers[0].attn.pe_cache.size(-1) * model.layers[0].attn.pe_cache.element_size()))
        os.environ["DS_TRACE_KV_BYTES_PER_TOKEN"] = str(bytes_per_token)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("load model")
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    print("I'm DeepSeek 👋")

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature, request_ids=[0])
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = f.read().split("\n\n")
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature, request_ids=list(range(len(prompt_tokens))))
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()
    tracer = ds_trace.get_tracer()
    if tracer.enabled:
        tracer.close()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--trace-enable", action="store_true")
    parser.add_argument("--trace-out", type=str, default="")
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument("--trace-store-scores", action="store_true")
    parser.add_argument("--trace-sample-rate", type=float, default=1.0)
    parser.add_argument("--trace-prefix-key-tokens", type=int, default=256)
    parser.add_argument("--trace-no-sync-cuda", action="store_true")
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(
        args.ckpt_path,
        args.config,
        args.input_file,
        args.interactive,
        args.max_new_tokens,
        args.temperature,
        args.trace_enable,
        args.trace_out,
        args.kv_block_size,
        args.trace_store_scores,
        args.trace_sample_rate,
        args.trace_prefix_key_tokens,
        not args.trace_no_sync_cuda,
    )
