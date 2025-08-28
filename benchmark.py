import asyncio
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import requests
import seaborn as sns

from dotenv import load_dotenv

load_dotenv()

# Configuration
LLAMA_CPP_URL = "http://localhost:10000/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single test"""

    test_name: str
    model_config: str
    latency_ms: float
    first_token_latency_ms: float
    tokens_generated: int
    token_speed: float
    throughput_imgs_per_sec: float
    response_text: str
    accuracy_score: float = 0.0
    similarity_reasoning: str = ""
    timestamp: str = ""


class VLMBenchmark:
    def __init__(
        self, server_url: str = LLAMA_CPP_URL, openai_key: str = OPENAI_API_KEY
    ):
        self.server_url = server_url
        self.openai_client = openai.OpenAI(api_key=openai_key)
        self.results = []

    def encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def call_llama_cpp(
        self, image_path: str, prompt: str, stream: bool = True
    ) -> Tuple[str, Dict]:
        """
        Call llama.cpp server with image and prompt
        Returns: (response_text, metrics_dict)
        """
        # Prepare the request following OpenAI's vision API format
        image_base64 = self.encode_image_base64(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ]

        headers = {"Content-Type": "application/json"}

        payload = {
            "model": "llava",  # Adjust model name as needed
            "messages": messages,
            "temperature": 0.1,  # Low temperature for consistency
            "max_tokens": 2048,
            "stream": stream,
        }

        metrics = {
            "start_time": time.perf_counter(),
            "first_token_time": None,
            "end_time": None,
            "tokens_count": 0,
            "response_text": "",
        }

        try:
            if stream:
                # Streaming response for detailed metrics
                response = requests.post(
                    f"{self.server_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    stream=True,
                )
                response.raise_for_status()

                full_response = []
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data_str)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content = delta["content"]
                                        # Only append if content is not None
                                        if content is not None:
                                            full_response.append(content)
                                            metrics["tokens_count"] += 1

                                            # Record first token time
                                            if metrics["first_token_time"] is None:
                                                metrics["first_token_time"] = (
                                                    time.perf_counter()
                                                )
                            except json.JSONDecodeError:
                                continue

                metrics["end_time"] = time.perf_counter()
                metrics["response_text"] = "".join(full_response)

            else:
                # Non-streaming for simpler implementation
                response = requests.post(
                    f"{self.server_url}/chat/completions", headers=headers, json=payload
                )
                response.raise_for_status()

                result = response.json()
                metrics["end_time"] = time.perf_counter()
                metrics["first_token_time"] = metrics[
                    "end_time"
                ]  # Can't measure in non-streaming
                metrics["response_text"] = result["choices"][0]["message"]["content"]
                # Estimate token count (rough approximation)
                metrics["tokens_count"] = len(metrics["response_text"].split()) * 1.3

        except requests.exceptions.RequestException as e:
            print(f"Error calling llama.cpp: {e}")
            raise

        return metrics["response_text"], metrics

    def evaluate_with_chatgpt(
        self, baseline_response: str, optimized_response: str, task_description: str
    ) -> Tuple[float, str]:
        """
        Use ChatGPT to evaluate the similarity between baseline and optimized responses
        Returns: (similarity_score, reasoning)
        """
        evaluation_prompt = f"""
        You are evaluating the quality of two VLM model responses for the same document extraction task.
        
        Task Description: {task_description}
        
        Baseline Response (Ground Truth):
        {baseline_response}
        
        Optimized Model Response:
        {optimized_response}
        
        Please evaluate the similarity and quality of the Optimized Model Response compared to the Baseline Response.
        Consider:
        1. Completeness - Are all important information extracted?
        2. Accuracy - Is the extracted information correct?
        3. Structure - Is the information well-organized?
        4. Missing Elements - What key information is missing?
        
        Provide:
        1. A similarity score from 0.0 to 1.0 (where 1.0 means identical quality)
        2. Brief reasoning for your score
        
        Format your response as JSON:
        {{
            "similarity_score": 0.XX,
            "reasoning": "Brief explanation...",
            "missing_elements": ["element1", "element2"],
            "accuracy_issues": ["issue1", "issue2"]
        }}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Or "gpt-4" for better evaluation
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise document extraction evaluator.",
                    },
                    {"role": "user", "content": evaluation_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            return result["similarity_score"], result["reasoning"]

        except Exception as e:
            print(f"Error evaluating with ChatGPT: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    def run_single_benchmark(
        self, image_path: str, prompt: str, test_name: str, model_config: str
    ) -> BenchmarkResult:
        """Run a single benchmark test"""
        print(f"Running benchmark: {test_name} with {model_config}")

        # Call the model and get metrics
        response_text, metrics = self.call_llama_cpp(image_path, prompt)

        # Calculate metrics
        total_latency = (metrics["end_time"] - metrics["start_time"]) * 1000  # ms
        first_token_latency = (
            (metrics["first_token_time"] - metrics["start_time"]) * 1000
            if metrics["first_token_time"]
            else total_latency
        )
        token_speed = (
            metrics["tokens_count"] / (metrics["end_time"] - metrics["start_time"])
            if metrics["end_time"] != metrics["start_time"]
            else 0
        )
        throughput = 1000 / total_latency  # images per second

        result = BenchmarkResult(
            test_name=test_name,
            model_config=model_config,
            latency_ms=total_latency,
            first_token_latency_ms=first_token_latency,
            tokens_generated=metrics["tokens_count"],
            token_speed=token_speed,
            throughput_imgs_per_sec=throughput,
            response_text=response_text,
            timestamp=datetime.now().isoformat(),
        )

        return result

    def benchmark_suite(self, test_cases: List[Dict], model_configs: List[str]):
        """
        Run complete benchmark suite

        test_cases: List of dicts with keys: 'name', 'image_path', 'prompt', 'task_description'
        model_configs: List of model configuration names (e.g., ['baseline_fp16', 'optimized_int8'])
        """
        results_df = pd.DataFrame()

        # First, collect baseline results
        baseline_results = {}
        baseline_config = model_configs[0]  # Assume first config is baseline

        print("=" * 50)
        print("Collecting Baseline Results...")
        print("=" * 50)

        for test_case in test_cases:
            result = self.run_single_benchmark(
                image_path=test_case["image_path"],
                prompt=test_case["prompt"],
                test_name=test_case["name"],
                model_config=baseline_config,
            )
            baseline_results[test_case["name"]] = result
            self.results.append(result)
            time.sleep(1)  # Small delay between tests

        # Now run optimized configs and compare
        print("Finish running baseline...\n")

        for config in model_configs[1:]:

            confirm = input("Press 'enter' to confirm testing with the next benchmark")

            print("=" * 50)
            print(f"Testing Configuration: {config}")
            print("=" * 50)

            for test_case in test_cases:
                # Run benchmark
                result = self.run_single_benchmark(
                    image_path=test_case["image_path"],
                    prompt=test_case["prompt"],
                    test_name=test_case["name"],
                    model_config=config,
                )

                # Evaluate against baseline
                baseline = baseline_results[test_case["name"]]
                score, reasoning = self.evaluate_with_chatgpt(
                    baseline_response=baseline.response_text,
                    optimized_response=result.response_text,
                    task_description=test_case["task_description"],
                )

                result.accuracy_score = score
                result.similarity_reasoning = reasoning

                self.results.append(result)

                # Print immediate results
                print(f"\nTest: {test_case['name']}")
                print(
                    f"  Latency: {result.latency_ms:.2f}ms (Baseline: {baseline.latency_ms:.2f}ms)"
                )
                print(
                    f"  Token Speed: {result.token_speed:.2f} toks/s (Baseline: {baseline.token_speed:.2f} tok/s)"
                )
                print(
                    f"  Throughput: {result.throughput_imgs_per_sec:.2f} imgs/s (Baseline: {baseline.throughput_imgs_per_sec} imgs/s)"
                )
                print(f"  Accuracy Score: {score:.3f}")
                print(f"  Speedup: {baseline.latency_ms/result.latency_ms:.2f}x")

                time.sleep(1)  # Delay to avoid rate limiting

        # Create results DataFrame
        results_df = pd.DataFrame([asdict(r) for r in self.results])
        return results_df

    def generate_report(
        self, results_df: pd.DataFrame, output_dir: str = "./benchmark_results"
    ):
        """Generate comprehensive benchmark report with visualizations"""
        Path(output_dir).mkdir(exist_ok=True)

        # Save raw results
        results_df.to_csv(f"{output_dir}/benchmark_results.csv", index=False)

        # Create comparison table
        comparison = results_df.pivot_table(
            index="test_name",
            columns="model_config",
            values=[
                "latency_ms",
                "token_speed",
                "accuracy_score",
                "throughput_imgs_per_sec",
            ],
        )

        # Generate visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Latency comparison
        latency_data = results_df.pivot(
            index="test_name", columns="model_config", values="latency_ms"
        )
        latency_data.plot(kind="bar", ax=axes[0, 0])
        axes[0, 0].set_title("Latency Comparison (ms)")
        axes[0, 0].set_ylabel("Latency (ms)")

        # Token speed comparison
        token_speed_data = results_df.pivot(
            index="test_name", columns="model_config", values="token_speed"
        )
        token_speed_data.plot(kind="bar", ax=axes[0, 1])
        axes[0, 1].set_title("Token Generation Speed (tokens/sec)")
        axes[0, 1].set_ylabel("Tokens/sec")

        # Accuracy scores
        if "accuracy_score" in results_df.columns:
            accuracy_data = results_df[
                results_df["model_config"] != results_df["model_config"].iloc[0]
            ]  # Exclude baseline
            accuracy_pivot = accuracy_data.pivot(
                index="test_name", columns="model_config", values="accuracy_score"
            )
            accuracy_pivot.plot(kind="bar", ax=axes[1, 0])
            axes[1, 0].set_title("Accuracy Scores (vs Baseline)")
            axes[1, 0].set_ylabel("Similarity Score")
            axes[1, 0].axhline(y=0.9, color="r", linestyle="--", label="90% threshold")
            axes[1, 0].legend()

        # Speedup calculation
        baseline_config = results_df["model_config"].iloc[0]
        baseline_latencies = results_df[
            results_df["model_config"] == baseline_config
        ].set_index("test_name")["latency_ms"]

        speedup_data = pd.DataFrame()
        for config in results_df["model_config"].unique():
            if config != baseline_config:
                config_latencies = results_df[
                    results_df["model_config"] == config
                ].set_index("test_name")["latency_ms"]
                speedup_data[config] = baseline_latencies / config_latencies

        if not speedup_data.empty:
            speedup_data.plot(kind="bar", ax=axes[1, 1])
            axes[1, 1].set_title("Speedup vs Baseline")
            axes[1, 1].set_ylabel("Speedup Factor")
            axes[1, 1].axhline(y=1, color="gray", linestyle="--")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/benchmark_charts.png", dpi=300)
        plt.show()

        # Generate summary statistics
        summary = {
            "config": [],
            "avg_latency": [],
            "avg_token_speed": [],
            "avg_accuracy": [],
            "avg_throughput": [],
        }

        for config in results_df["model_config"].unique():
            config_data = results_df[results_df["model_config"] == config]
            summary["config"].append(config)
            summary["avg_latency"].append(config_data["latency_ms"].mean())
            summary["avg_token_speed"].append(config_data["token_speed"].mean())
            summary["avg_accuracy"].append(
                config_data["accuracy_score"].mean()
                if "accuracy_score" in config_data
                else 1.0
            )
            summary["avg_throughput"].append(
                config_data["throughput_imgs_per_sec"].mean()
            )

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(f"{output_dir}/summary_statistics.csv", index=False)

        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(summary_df.to_string(index=False))

        return comparison, summary_df
