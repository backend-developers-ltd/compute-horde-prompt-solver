from .prompt_solver import CLISolver, GPULLMProvider, MockLLMProvider, HttpSolver
from .config import parse_arguments


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
