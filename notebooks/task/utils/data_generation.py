"""
Mooncake Dataset Generation Utilities

This module provides functions for generating synthetic LLM inference workload datasets
using the Mooncake format. The generator creates realistic workloads with features like:

Load Pattern Generation:
- Sinusoidal Load Curves: Simulates natural traffic patterns with peaks and valleys
- Variable Request Rates: Configurable minimum and maximum request rates
- Temporal Distribution: Realistic timing between requests

Token Configuration:
- Input Sequence Length (ISL): Controls the length of input prompts
- Output Sequence Length (OSL): Controls the length of expected responses
- Dual Length Support: Can simulate mixed workloads with different token patterns

Key Parameters:
- time_duration: Total duration of the load test (seconds)
- request_rate_min/max: Minimum and maximum requests per second
- request_rate_period: Period of the sinusoidal load curve (seconds)
- isl1/isl2: Input sequence lengths for workload patterns
- osl1/osl2: Output sequence lengths for workload patterns
"""

import json
import os
import subprocess


def generate_mooncake_dataset(
    time_duration: int = 60,
    request_rate_min: int = 2,
    request_rate_max: int = 8,
    request_rate_period: int = 20,
    isl1: int = 3000,
    osl1: int = 150,
    isl2: int = 3000,
    osl2: int = 150,
    output_file: str = "results/dataset.jsonl",
) -> bool:
    """Generate a synthetic dataset using Mooncake format with sinusoidal load pattern.

    Args:
        time_duration: Total duration of the load test in seconds
        request_rate_min: Minimum requests per second
        request_rate_max: Maximum requests per second
        request_rate_period: Period of the sinusoidal load curve in seconds
        isl1: Input sequence length for first workload pattern
        osl1: Output sequence length for first workload pattern
        isl2: Input sequence length for second workload pattern
        osl2: Output sequence length for second workload pattern
        output_file: Path to save the generated dataset

    Returns:
        bool: True if dataset was generated successfully, False otherwise
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Build command
    cmd = [
        "python",
        "benchmarks/sin_load_generator/sin_synth.py",
        "--time-duration",
        str(time_duration),
        "--request-rate-min",
        str(request_rate_min),
        "--request-rate-max",
        str(request_rate_max),
        "--request-rate-period",
        str(request_rate_period),
        "--isl1",
        str(isl1),
        "--osl1",
        str(osl1),
        "--isl2",
        str(isl2),
        "--osl2",
        str(osl2),
        "--output-file",
        output_file,
    ]

    try:
        # Run generator
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("\n✅ Dataset generation completed!")

        # Verify dataset
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                line_count = sum(1 for line in f)

            print("📄 Dataset created successfully:")
            print(f"   📊 Total requests: {line_count}")
            print(f"   💾 File size: {os.path.getsize(output_file)} bytes")

            # Show sample entries
            print("\n📋 Sample dataset entries:")
            with open(output_file, "r") as f:
                for i, line in enumerate(f):
                    if i < 3:  # Show first 3 entries
                        try:
                            entry = json.loads(line)
                            timestamp = entry.get("timestamp", "N/A")
                            print(f"   {i + 1}. Timestamp: {timestamp:.2f}s")
                        except:
                            print(f"   {i + 1}. Raw entry: {line[:100]}...")
                    else:
                        break
            return True
        else:
            print("❌ Dataset creation failed - file not found")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating dataset: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
