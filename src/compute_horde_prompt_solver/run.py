from .prompt_solver import CLISolver, GPULLMProvider, MockLLMProvider, HttpSolver
from .config import parse_arguments


def mock_run(input_file: pathlib.Path, output_dir: pathlib.Path):
    with open(input_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    output_file = output_dir / f"{input_file.stem}.json"
    with open(output_file, "w") as f:
        json.dump({prompt: "mock mock mock" for prompt in prompts}, f, indent=2)


def main():
    config = parse_arguments()
    if not config.mock:
        provider = GPULLMProvider(config.model, config.dtype)
    else:
        provider = MockLLMProvider()

    if config.server:
        solver = HttpSolver(provider, config)
    else:
        solver = CLISolver(provider, config)

    solver.run()


if __name__ == "__main__":
    main()
