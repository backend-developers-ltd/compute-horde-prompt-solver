import argparse
import json
import pathlib
from typing import List, Dict

import torch
import vllm
from vllm import SamplingParams

# Import the set_deterministic function
from deterministic_ml.v1 import set_deterministic


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate responses for prompts using vLLM."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        type=pathlib.Path,
        help="Input files containing prompts",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        type=pathlib.Path,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--model", default="microsoft/Phi-3.5-mini-instruct", help="Model name or path"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.1, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dtype", default="auto",
        choices=("auto", "half", "float16", "bfloat16", "float", "float32"),
        help=(
            "model dtype - setting `float32` helps with deterministic prompts in different batches"
        )
    )
    return parser.parse_args()


def setup_model(model_name: str, dtype: str = "auto") -> vllm.LLM:
    gpu_count = torch.cuda.device_count()
    return vllm.LLM(
        model=model_name,
        tensor_parallel_size=gpu_count,
        max_model_len=6144,
        enforce_eager=True,
        dtype=dtype,
    )


def make_prompt(prompt: str) -> str:
    system_msg = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{{{{ You are a helpful AI assistant }}}}<|eot_id|>"
    user_msg = f"<|start_header_id|>user<|end_header_id|>\n{{{{ {prompt} }}}}<|eot_id|>"
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
    return f"{system_msg}{user_msg}{assistant_start}"


def generate_responses(
    model: vllm.LLM, prompts: List[str], sampling_params: SamplingParams
) -> Dict[str, str]:
    requests = [make_prompt(prompt) for prompt in prompts]
    responses = model.generate(requests, sampling_params, use_tqdm=True)
    return {
        prompt: response.outputs[0].text for prompt, response in zip(prompts, responses)
    }


def process_file(
    model: vllm.LLM,
    input_file: pathlib.Path,
    output_dir: pathlib.Path,
    sampling_params: SamplingParams,
):
    with open(input_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    responses = generate_responses(model, prompts, sampling_params)

    output_file = output_dir / f"{input_file.stem}.json"
    with open(output_file, "w") as f:
        json.dump(responses, f, indent=2)


def main():
    args = parse_arguments()

    # Set deterministic behavior
    set_deterministic(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = setup_model(args.model)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    for input_file in args.input_files:
        process_file(model, input_file, args.output_dir, sampling_params)


if __name__ == "__main__":
    main()
