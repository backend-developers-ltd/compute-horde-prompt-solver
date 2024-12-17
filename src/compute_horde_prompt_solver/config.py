import argparse
import dataclasses
import pathlib
from typing import Optional, List


@dataclasses.dataclass
class Config:
    input_files: List[pathlib.Path]
    output_dir: pathlib.Path
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    dtype: str
    seed: Optional[int]
    server: Optional[bool]
    server_port: int
    mock: bool


def parse_arguments() -> Config:
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
        "--dtype",
        default="auto",
        choices=("auto", "half", "float16", "bfloat16", "float", "float32"),
        help=(
            "model dtype - setting `float32` helps with deterministic prompts in different batches"
        ),
    )

    seed_or_server_group = parser.add_mutually_exclusive_group(required=True)
    seed_or_server_group.add_argument(
        "--seed", type=int, help="Random seed for reproducibility"
    )
    seed_or_server_group.add_argument(
        "--server",
        action="store_true",
        help="Spin up a temporary HTTP server to receive the seed",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="Port for temporary HTTP server",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Don't use an actual model, generate random gibberish based on the input and the seed",
    )
    args = parser.parse_args()

    return Config(
        input_files=args.input_files,
        output_dir=args.output_dir,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype=args.dtype,
        seed=args.seed,
        server=args.server,
        server_port=args.server_port,
        mock=args.mock,
    )
