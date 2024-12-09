import abc
import hashlib
import json
import multiprocessing as mp
import pathlib
import queue
import random
import string
from typing import List, Dict

import torch
import vllm
from flask import Flask, Blueprint, jsonify
from vllm import SamplingParams

# Import the set_deterministic function
from deterministic_ml.v1 import set_deterministic

from .config import Config

TIMEOUT = 5 * 60


class BaseLLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate_responses(self, prompts: List[str], sampling_params: SamplingParams) -> Dict[str, str]: ...


class GPULLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str, dtype: str = "auto"):
        self.model_name = model_name
        self.dtype = dtype
        self._model = None

    @property
    def model(self):
        if self._model is None:
            return self._model
        self._model = self.setup_model()
        return self._model

    def setup_model(self) -> vllm.LLM:
        gpu_count = torch.cuda.device_count()
        return vllm.LLM(
            model=self.model_name,
            tensor_parallel_size=gpu_count,
            max_model_len=6144,
            enforce_eager=True,
            dtype=self.dtype,
        )

    def make_prompt(self, prompt: str) -> str:
        system_msg = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{{{{ You are a helpful AI assistant }}}}<|eot_id|>"
        user_msg = f"<|start_header_id|>user<|end_header_id|>\n{{{{ {prompt} }}}}<|eot_id|>"
        assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
        return f"{system_msg}{user_msg}{assistant_start}"

    def generate_responses(
            self, prompts: List[str], sampling_params: SamplingParams
    ) -> Dict[str, str]:
        requests = [self.make_prompt(prompt) for prompt in prompts]
        responses = self.model.generate(requests, sampling_params, use_tqdm=True)
        return {
            prompt: response.outputs[0].text for prompt, response in zip(prompts, responses)
        }


class MockLLMProvider(BaseLLMProvider):
    def generate_responses(self, prompts: List[str], sampling_params: SamplingParams) -> Dict[str, str]:
        result = {}
        for prompt in prompts:
            generator = random.Random(str(sampling_params.seed) + prompt)
            answer_length = generator.randint(100, 200)
            answer = ''.join(generator.choice(string.ascii_letters) for _ in range(answer_length))
            result[prompt] = answer
        return result


def _run_server(
        start_server_event: mp.Event,
        seed_queue: mp.Queue,
        result_queue: mp.Queue,
        ready_to_terminate_event: mp.Event,
        config: Config,
):
    start_server_event.wait()

    app = Flask("compute_horde_prompt_solver")

    @app.route("/health")
    def server_healthcheck():
        return {"status": "ok"}

    @app.route("/execute-job", methods=["POST"])
    def execute_job():
        try:
            from flask import request

            seed_raw = request.json.get("seed")
            seed = int(seed_raw)
            seed_queue.put(seed)
            result = result_queue.get(timeout=TIMEOUT)
            return jsonify(result)
        finally:
            # The seed_queue.put(seed) can fail (request not having int seed etc.),
            # so we always put a None to make sure process is terminated when the view returns.
            seed_queue.put(None)

    @app.route("/terminate")
    def terminate():
        ready_to_terminate_event.set()
        return {"status": "ok"}

    app.run(
        host="0.0.0.0",
        port=config.server_port,
        debug=False,
    )


class BaseSolver(abc.ABC):
    def __init__(
            self,
            provider: BaseLLMProvider,
            config: Config
    ):
        self.provider = provider
        self.config = config

    def process_file(self, input_file, sampling_params):
        with open(input_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

        responses = self.provider.generate_responses(prompts, sampling_params)

        output_file = self.config.output_dir / f"{input_file.stem}.json"
        self.save_output_file(responses, output_file)

    def save_output_file(self, responses: Dict[str, str], output_file: pathlib.Path):
        with open(output_file, "w") as f:
            json.dump(responses, f, indent=2)

    def get_sampling_params(self, seed):
        set_deterministic(seed)

        return SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            seed=seed,
        )

    @abc.abstractmethod
    def run(self): ...


class CLISolver(BaseSolver):

    def run(self):
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        sampling_params = self.get_sampling_params(self.config.seed)

        for input_file in self.config.input_files:
            self.process_file(input_file, sampling_params)


class HttpSolver(BaseSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_server_event = mp.Event()
        self.seed_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.ready_to_terminate_event = mp.Event()
        self.response_hashes: Dict[str, str] = {}

    def save_output_file(self, responses: Dict[str, str], output_file: pathlib.Path):
        response_body = json.dumps(responses, indent=2).encode()
        self.response_hashes[output_file.as_posix()] = hashlib.sha256(response_body).hexdigest()
        pathlib.Path(output_file).write_bytes(response_body)

    def run(self):
        process = mp.Process(
            target=_run_server,
            args=(
                self.start_server_event,
                self.seed_queue,
                self.result_queue,
                self.ready_to_terminate_event,
                self.config,
            )
        )
        process.start()

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_server_event.set()

        try:
            seed = self.seed_queue.get(block=True, timeout=TIMEOUT)
        except queue.Empty:
            seed = None

        if seed is None:
            raise SystemExit("ERROR: provided seed is malformed!")

        sampling_params = self.get_sampling_params(seed)

        try:
            for input_file in self.config.input_files:
                self.process_file(input_file, sampling_params)
            self.result_queue.put(self.response_hashes)
            self.ready_to_terminate_event.wait(timeout=TIMEOUT)
        finally:
            process.terminate()
