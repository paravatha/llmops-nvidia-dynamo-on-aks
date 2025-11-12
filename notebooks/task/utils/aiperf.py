"""
NVIDIA AIPerf Benchmarking Utilities

This module provides functions for benchmarking LLM deployments using NVIDIA's AIPerf tool.
It supports comprehensive performance testing with features like:
- Realistic load testing using Mooncake dataset with fixed schedule
- Comprehensive metrics (latency, throughput, token rates)
- Multiple protocol support (OpenAI-compatible APIs)
- Detailed analysis with JSON/CSV exports

The module implements benchmark strategies for:
1. Aggregated Deployments: Standard and Router configurations
2. Disaggregated Deployments: Standard and Router configurations
3. Comparative Analysis: Side-by-side metric comparison

Reference: https://github.com/ai-dynamo/aiperf
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
    deployment_name: str,
    url_path: str,
    artifact_dir: str,
    load_balancer_ip: str,
    dataset_path: str = "results/dataset.jsonl",
) -> bool:
    """Run AIPerf benchmark using Mooncake dataset with fixed schedule.

    Args:
        deployment_name: Name of the deployment being benchmarked
        url_path: URL path for the deployment endpoint
        artifact_dir: Directory to store benchmark results
        load_balancer_ip: IP address of the load balancer
        dataset_path: Path to the Mooncake trace dataset file

    Returns:
        bool: True if benchmark completed successfully, False otherwise
    """
    print(f"🏃‍♂️ Running AIPerf benchmark for {deployment_name}...")
    print(f"📊 URL: http://{load_balancer_ip}/{url_path}/v1")
    print(f"💾 Artifact Directory: {artifact_dir}")
    print(f"📁 Dataset: {dataset_path}")

    # Verify dataset exists
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
            results_json = os.path.join(artifact_dir, "profile_export_aiperf.json")
            results_jsonl = os.path.join(artifact_dir, "profile_export.jsonl")

            if os.path.exists(results_json):
                print(f"📄 Results saved to: {results_json}")
                return True
            elif os.path.exists(results_jsonl):
                print(f"📄 Results saved to: {results_jsonl}")
                return True
            else:
                print("⚠️  Benchmark completed but no results file found")
                # List all files in artifact dir
                all_files = glob.glob(f"{artifact_dir}/*")
                if all_files:
                    print(f"📂 Files in artifact directory: {all_files}")
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


def create_benchmark_comparison_visualization(
    standard_dir: str,
    router_dir: str,
    deployment_labels: tuple = ("Standard", "Router"),
    output_dir: str = "results/plots",
    output_filename: str = "benchmark_comparison.png",
) -> "plt.Figure":
    """Create comprehensive performance comparison charts from AIPerf results.

    This function loads benchmark results from two deployments and creates a 2x2 grid
    of comparison charts showing:
    1. Average Request Latency
    2. P99 Request Latency
    3. Time to First Token
    4. Output Token Throughput

    Args:
        standard_dir: Directory containing first deployment benchmark results
        router_dir: Directory containing second deployment benchmark results
        deployment_labels: Tuple of labels for the two deployments (default: ("Standard", "Router"))
        output_dir: Directory to save the visualization plots
        output_filename: Name of the output file (default: "benchmark_comparison.png")

    Returns:
        matplotlib.figure.Figure: The generated comparison plot figure
        None: If insufficient data is available

    Example:
        # For aggregated deployments
        create_benchmark_comparison_visualization(
            standard_dir="results/standard-benchmark",
            router_dir="results/router-benchmark",
            deployment_labels=("Standard Agg", "Router Agg"),
            output_filename="agg_comparison.png"
        )

        # For disaggregated deployments
        create_benchmark_comparison_visualization(
            standard_dir="results/standard-benchmark-disagg",
            router_dir="results/router-benchmark-disagg",
            deployment_labels=("Standard Disagg", "Router Disagg"),
            output_filename="disagg_comparison.png"
        )
    """

    print("📊 Creating Performance Comparison Visualization...")
    print(f"   Deployment 1 results: {standard_dir}")
    print(f"   Deployment 2 results: {router_dir}")
    print(f"   Labels: {deployment_labels}")

    def load_aiperf_json(file_path):
        """Load AIPerf JSON results."""
        try:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    print(f"✅ Loaded results from: {file_path}")
                    return data
            else:
                print(f"⚠️  File not found: {file_path}")
                return None
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None

    # Load results from both benchmarks
    standard_json = os.path.join(standard_dir, "profile_export_aiperf.json")
    router_json = os.path.join(router_dir, "profile_export_aiperf.json")

    standard_results = load_aiperf_json(standard_json)
    router_results = load_aiperf_json(router_json)

    if not standard_results or not router_results:
        print("⚠️  Insufficient benchmark data for visualization")
        return None

    # Extract metrics for comparison
    deployments = list(deployment_labels)

    # Helper function to safely get metric values
    def get_metric(results, metric_name, stat="avg"):
        metric = results.get(metric_name, {})
        if isinstance(metric, dict):
            return metric.get(stat, 0)
        return 0

    # Latency metrics (ms) - AIPerf provides these directly
    avg_latency = [
        get_metric(standard_results, "request_latency", "avg"),
        get_metric(router_results, "request_latency", "avg"),
    ]

    p90_latency = [
        get_metric(standard_results, "request_latency", "p90"),
        get_metric(router_results, "request_latency", "p90"),
    ]

    # Time to First Token (ms)
    time_to_first_token = [
        get_metric(standard_results, "time_to_first_token", "avg"),
        get_metric(router_results, "time_to_first_token", "avg"),
    ]

    # Token Throughput (tokens/sec)
    token_throughput = [
        get_metric(standard_results, "output_token_throughput", "avg"),
        get_metric(router_results, "output_token_throughput", "avg"),
    ]

    # Request Throughput (req/sec)
    request_throughput = [
        get_metric(standard_results, "request_throughput", "avg"),
        get_metric(router_results, "request_throughput", "avg"),
    ]

    # Print metrics for debugging
    print("\n📊 Extracted Metrics:")
    print(
        f"   {deployment_labels[0]:<15} - Avg Latency: {avg_latency[0]:.2f}ms, P90: {p90_latency[0]:.2f}ms, TTFT: {time_to_first_token[0]:.2f}ms"
    )
    print(
        f"   {deployment_labels[1]:<15} - Avg Latency: {avg_latency[1]:.2f}ms, P90: {p90_latency[1]:.2f}ms, TTFT: {time_to_first_token[1]:.2f}ms"
    )

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"AIPerf Benchmark Comparison: {deployment_labels[0]} vs {deployment_labels[1]}",
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
        if height > 0:
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

    # 2. P90 Latency Comparison
    bars2 = ax2.bar(deployments, p90_latency, color=colors, alpha=0.8)
    ax2.set_title("P90 Request Latency", fontweight="bold")
    ax2.set_ylabel("Latency (ms)")
    ax2.grid(axis="y", alpha=0.3)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        if height > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(p90_latency) * 0.01,
                f"{height:.1f}ms",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # 3. Time to First Token Comparison
    bars3 = ax3.bar(deployments, time_to_first_token, color=colors, alpha=0.8)
    ax3.set_title("Time to First Token (TTFT)", fontweight="bold")
    ax3.set_ylabel("Latency (ms)")
    ax3.grid(axis="y", alpha=0.3)

    for i, bar in enumerate(bars3):
        height = bar.get_height()
        if height > 0:
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(time_to_first_token) * 0.01,
                f"{height:.1f}ms",
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
        if height > 0:
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
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, output_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n💾 Visualization saved to: {plot_path}")

    # Print detailed summary
    print("\n📊 Performance Comparison Summary:")
    print("=" * 80)
    print(f"{'Metric':<35} {deployment_labels[0]:<20} {deployment_labels[1]:<20}")
    print("=" * 80)
    print(
        f"{'Avg Request Latency (ms)':<35} {avg_latency[0]:<20.2f} {avg_latency[1]:<20.2f}"
    )
    print(
        f"{'P90 Request Latency (ms)':<35} {p90_latency[0]:<20.2f} {p90_latency[1]:<20.2f}"
    )
    print(
        f"{'Time to First Token (ms)':<35} {time_to_first_token[0]:<20.2f} {time_to_first_token[1]:<20.2f}"
    )
    print(
        f"{'Token Throughput (tok/s)':<35} {token_throughput[0]:<20.2f} {token_throughput[1]:<20.2f}"
    )
    print(
        f"{'Request Throughput (req/s)':<35} {request_throughput[0]:<20.2f} {request_throughput[1]:<20.2f}"
    )
    print("=" * 80)

    # Calculate improvements
    if avg_latency[0] > 0 and avg_latency[1] > 0:
        latency_improvement = ((avg_latency[0] - avg_latency[1]) / avg_latency[0]) * 100
        print(f"\n🎯 {deployment_labels[1]} Performance vs {deployment_labels[0]}:")
        print(f"   Latency: {latency_improvement:+.1f}%")

    if token_throughput[0] > 0 and token_throughput[1] > 0:
        throughput_improvement = (
            (token_throughput[1] - token_throughput[0]) / token_throughput[0]
        ) * 100
        print(f"   Throughput: {throughput_improvement:+.1f}%")

    if time_to_first_token[0] > 0 and time_to_first_token[1] > 0:
        ttft_improvement = (
            (time_to_first_token[0] - time_to_first_token[1]) / time_to_first_token[0]
        ) * 100
        print(f"   TTFT: {ttft_improvement:+.1f}%")

    plt.show()
    return fig


# Backward compatibility aliases
run_genai_perf_benchmark = run_aiperf_benchmark
