"""Interactive, multi-turn text chat with a local Qwen3.5 dense/MoE checkpoint."""

import argparse

from scripts._common import (
    add_model_argument,
    normalize_paths,
    validate_finite,
    validate_gpu_utilization,
    validate_positive,
)

HELP = "/clear: clear conversation; /help: show commands; /exit or /quit: exit"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument("--system", default="", help="optional system prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--decode-graph", action="store_true")
    args = parser.parse_args(argv)
    validate_positive(parser, args, "max_new_tokens", "max_model_len", "max_num_batched_tokens")
    validate_gpu_utilization(parser, args)
    validate_finite(parser, args.temperature, "temperature")
    if args.max_new_tokens > args.max_model_len:
        parser.error("--max-new-tokens must leave room for the chat prompt")
    if not args.model:
        parser.error("model path required: pass model or set NANOVLLM_MODEL")
    return normalize_paths(args, "model")


def run_chat(llm, sampling, *, system_prompt: str = "") -> None:
    initial = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages = list(initial)
    print(HELP, flush=True)
    try:
        while True:
            text = input("You: ").strip()
            if not text:
                continue
            if text in ("/exit", "/quit"):
                break
            if text == "/help":
                print(HELP, flush=True)
                continue
            if text == "/clear":
                messages = list(initial)
                print("Conversation cleared.", flush=True)
                continue

            candidate = [*messages, {"role": "user", "content": text}]
            try:
                # Tokenize the template once; passing text to generate would
                # tokenize it again and could add extra special tokens.
                prompt_ids = llm.tokenizer.apply_chat_template(
                    candidate,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                output = llm.generate([prompt_ids], sampling, use_tqdm=False)[0]
            except ValueError as exc:
                # Request validation happens before enqueueing. A failed model
                # execution can leave active work, in which case exit instead.
                if not llm.is_finished():
                    raise
                print(f"Request rejected: {exc}", flush=True)
                print("Use /clear to reset history, or enter a shorter message.", flush=True)
                continue

            reply = output["text"]
            messages = [*candidate, {"role": "assistant", "content": reply}]
            print(f"Assistant: {reply}\n", flush=True)
    except (EOFError, KeyboardInterrupt):
        print()


def run(args) -> None:
    # Keep --help and CLI validation available without loading CUDA dependencies.
    from nanovllm import LLM, SamplingParams

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        ignore_eos=False,
    )
    llm = LLM(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_visual_tokens_per_request=0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        decode_graph_batch_sizes=(1,) if args.decode_graph else (),
    )
    try:
        run_chat(llm, sampling, system_prompt=args.system)
    finally:
        llm.exit()


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
