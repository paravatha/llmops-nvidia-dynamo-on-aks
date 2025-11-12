"""
NVIDIA AIPerf Benchmarking Utilities

This module provides functions for benchmarking LLM deployments using NVIDIA's AIPerf tool.
It supports comprehensive performance testing with features like:
- Realistic load testing using Mooncake dataset with fixed schedule
- Comprehensive metrics (latency, throughput, token rates)
- Multiple protocol support (OpenAI-compatible APIs)
- Detailed analysis with JSON/CSV exports

The module implements two main benchmark strategies:
1. Standard Deployment: Baseline performance with single worker
2. Router Deployment: Performance with KV cache-aware routing
3. Comparative Analysis: Side-by-side metric comparison
"""

import os
import subprocess
import time

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print(
        "Warning: matplotlib or numpy not found. Visualization functions will not work."
    )
    plt = None
    np = None

import glob
import json


def run_aiperf_benchmark(
    deployment_name: str, url_path: str, artifact_dir: str, load_balancer_ip: str
) -> bool:
    """Run AIPerf benchmark using Mooncake dataset with fixed schedule.

    Args:
        deployment_name: Name of the deployment being benchmarked
        url_path: URL path for the deployment endpoint
        artifact_dir: Directory to store benchmark results
        load_balancer_ip: IP address of the load balancer

    Returns:
        bool: True if benchmark completed successfully, False otherwise
    """
    print(f"🏃‍♂️ Running AIPerf benchmark for {deployment_name}...")
    print(f"📊 URL: http://{load_balancer_ip}/{url_path}/v1")
    print(f"💾 Artifact Directory: {artifact_dir}")

    # Verify dataset exists
    dataset_path = "results/dataset.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return False

    # Create artifact directory
    os.makedirs(artifact_dir, exist_ok=True)

    try:
        # Build AIPerf command for Mooncake trace benchmarking
        cmd = [
            "aiperf",
            "profile",
            "-m",
            "Qwen/Qwen3-0.6B",
            "--endpoint-type",
            "chat",
            "--url",
            f"http://{load_balancer_ip}/{url_path}/v1",
            "--input-file",
            dataset_path,
            "--custom-dataset-type",
            "mooncake_trace",
            "--fixed-schedule",
            "--artifact-dir",
            artifact_dir,
            "--streaming",
        ]

        print("🚀 Executing AIPerf benchmark:")
        print(f"   Command: {' '.join(cmd)}")
        print("⏳ This may take several minutes...")

        # Run the benchmark
        start_time = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )  # 10 minute timeout
        end_time = time.time()

        # Show output
        if result.stdout:
            print(f"\n📋 AIPerf Output:\n{result.stdout}")
        if result.stderr and "ERROR" in result.stderr:
            print(f"\n⚠️  AIPerf Errors:\n{result.stderr}")

        if result.returncode == 0:
            print(
                f"\n✅ Benchmark completed for {deployment_name} in {end_time - start_time:.1f}s"
            )

            # Verify results were generated
            results_pattern = f"{artifact_dir}/**/profile_export.jsonl"
            results_files = glob.glob(results_pattern, recursive=True)

            if results_files:
                print(f"📄 Results saved to: {results_files[0]}")
                return True
            else:
                print("⚠️  Benchmark completed but no results file found")
                # Check for any JSON files
                json_files = glob.glob(f"{artifact_dir}/**/*.json", recursive=True)
                if json_files:
                    print(f"📄 Found alternative results: {json_files}")
                    return True
                return False
        else:
            print(
                f"❌ Benchmark failed for {deployment_name} with exit code: {result.returncode}"
            )
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ Benchmark timed out for {deployment_name}")
        return False
    except Exception as e:
        print(f"❌ Error running benchmark for {deployment_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_benchmark_comparison_visualization() -> plt.Figure:
    """Create comprehensive performance comparison charts from AIPerf results.

    This function loads benchmark results from both standard vLLM and KV Cache Router
    deployments, and creates a 2x2 grid of comparison charts showing:
    1. Average Request Latency
    2. P99 Request Latency
    3. Time to First Token
    4. Output Token Throughput

    The function expects results to be in:
    - results/standard-benchmark/**/profile_export.jsonl
    - results/router-benchmark/**/profile_export.jsonl

    Returns:
        matplotlib.figure.Figure: The generated comparison plot figure
        None: If insufficient data is available
    """

    print("📊 Creating Performance Comparison Visualization...")

    def load_aiperf_results(path_pattern):
        """Load and aggregate AIPerf results from JSONL file."""
        files = glob.glob(path_pattern, recursive=True)
        if not files:
            return None

        try:
            with open(files[0], "r") as f:
                # Parse JSONL format
                metrics = {}
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Aggregate metrics
                        for key, value in data.items():
                            if isinstance(value, (int, float)):
                                if key not in metrics:
                                    metrics[key] = []
                                metrics[key].append(value)

                # Calculate statistics
                summary = {}
                for key, values in metrics.items():
                    if values:
                        summary[key] = {
                            "avg": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values),
                            "p99": sorted(values)[int(len(values) * 0.99)]
                            if len(values) > 0
                            else 0,
                        }
                return summary
        except Exception as e:
            print(f"❌ Error loading {path_pattern}: {e}")
            return None

    # Load results from both benchmarks
    standard_results = load_aiperf_results(
        "results/standard-benchmark/**/profile_export.jsonl"
    )
    router_results = load_aiperf_results(
        "results/router-benchmark/**/profile_export.jsonl"
    )

    if not standard_results or not router_results:
        print("⚠️  Insufficient benchmark data for visualization")
        print("   Checking for alternative result formats...")

        # Try to load JSON format instead
        standard_results = load_aiperf_results("results/standard-benchmark/**/*.json")
        router_results = load_aiperf_results("results/router-benchmark/**/*.json")

        if not standard_results or not router_results:
            print("❌ No benchmark data found")
            return None

    # Extract metrics for comparison
    deployments = ["Standard vLLM", "KV Cache Router"]

    # Latency metrics (ms)
    avg_latency = [
        standard_results.get("request_latency", {}).get("avg", 0),
        router_results.get("request_latency", {}).get("avg", 0),
    ]

    p99_latency = [
        standard_results.get("request_latency", {}).get("p99", 0),
        router_results.get("request_latency", {}).get("p99", 0),
    ]

    # Throughput metrics
    request_throughput = [
        standard_results.get("request_throughput", {}).get("avg", 0),
        router_results.get("request_throughput", {}).get("avg", 0),
    ]

    token_throughput = [
        standard_results.get("output_token_throughput", {}).get("avg", 0),
        router_results.get("output_token_throughput", {}).get("avg", 0),
    ]

    time_to_first_token = [
        standard_results.get("time_to_first_token", {}).get("avg", 0),
        router_results.get("time_to_first_token", {}).get("avg", 0),
    ]

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "AIPerf Benchmark Comparison: Mooncake Dataset Results",
        fontsize=16,
        fontweight="bold",
    )

    colors = ["#FF9999", "#99CCFF"]

    # 1. Average Latency Comparison
    bars1 = ax1.bar(deployments, avg_latency, color=colors, alpha=0.8)
    ax1.set_title("Average Request Latency", fontweight="bold")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(axis="y", alpha=0.3)

    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(avg_latency) * 0.01,
            f"{height:.1f}ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Calculate improvement
    if avg_latency[0] > 0 and avg_latency[1] > 0:
        improvement = ((avg_latency[0] - avg_latency[1]) / avg_latency[0]) * 100
        ax1.text(
            0.5,
            max(avg_latency) * 0.7,
            f"{improvement:.1f}% faster",
            ha="center",
            transform=ax1.transData,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        )

    # 2. P99 Latency Comparison
    bars2 = ax2.bar(deployments, p99_latency, color=colors, alpha=0.8)
    ax2.set_title("P99 Request Latency", fontweight="bold")
    ax2.set_ylabel("Latency (ms)")
    ax2.grid(axis="y", alpha=0.3)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(p99_latency) * 0.01,
            f"{height:.1f}ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. TTFT  Comparison
    bars3 = ax3.bar(deployments, time_to_first_token, color=colors, alpha=0.8)
    ax3.set_title("Time to First Token", fontweight="bold")
    ax3.set_ylabel("Latency (ms)")
    ax3.grid(axis="y", alpha=0.3)

    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(time_to_first_token) * 0.01,
            f"{height:.2f}ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 4. Token Throughput Comparison
    bars4 = ax4.bar(deployments, token_throughput, color=colors, alpha=0.8)
    ax4.set_title("Output Token Throughput", fontweight="bold")
    ax4.set_ylabel("Tokens/sec")
    ax4.grid(axis="y", alpha=0.3)

    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(token_throughput) * 0.01,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.tight_layout()

    # Save the plot
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/benchmark_comparison.png", dpi=300, bbox_inches="tight")

    # Print summary
    print("\n📊 Performance Comparison Summary:")
    print(f"   Average Latency: {avg_latency[0]:.1f}ms → {avg_latency[1]:.1f}ms")
    print(f"   P99 Latency: {p99_latency[0]:.1f}ms → {p99_latency[1]:.1f}ms")
    print(f"   TTFT: {time_to_first_token[0]:.2f}ms → {time_to_first_token[1]:.2f}ms")
    print(
        f"   Token Throughput: {token_throughput[0]:.1f} → {token_throughput[1]:.1f} tok/s"
    )

    plt.show()
    return fig


# Backward compatibility aliases
run_genai_perf_benchmark = run_aiperf_benchmark
