"""
Directory and Path Management Utilities

This module provides constants and functions for managing paths and directories
across the Dynamo lab notebooks.
"""

import os

# Base directories
WORKSPACE_ROOT = "/dli/task"
LAB_ROOT = WORKSPACE_ROOT

# Deployment directories
DEPLOYMENT_DIR = os.path.join(LAB_ROOT, "dynamo-deployments")
CONFIGS_DIR = os.path.join(DEPLOYMENT_DIR, "updated_configs")
PVC_DIR = os.path.join(DEPLOYMENT_DIR, "pvc")
INGRESS_DIR = os.path.join(DEPLOYMENT_DIR, "ingress")

# Results and artifacts
RESULTS_DIR = os.path.join(LAB_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def ensure_dirs_exist():
    """Create all required directories if they don't exist."""
    for directory in [CONFIGS_DIR, PVC_DIR, INGRESS_DIR, RESULTS_DIR, PLOTS_DIR]:
        os.makedirs(directory, exist_ok=True)


def get_deployment_path(deployment_type: str) -> str:
    """Get path to deployment YAML file.

    Args:
        deployment_type: Type of deployment (e.g., 'vllm-agg', 'vllm-disagg')

    Returns:
        str: Path to deployment YAML file
    """
    return os.path.join(DEPLOYMENT_DIR, f"{deployment_type}.yaml")


def get_pvc_path(deployment_type: str) -> str:
    """Get path to PVC YAML file.

    Args:
        deployment_type: Type of deployment (e.g., 'vllm-agg', 'vllm-disagg')

    Returns:
        str: Path to PVC YAML file
    """
    return os.path.join(PVC_DIR, f"{deployment_type}-pvc.yaml")


def get_ingress_path(deployment_type: str) -> str:
    """Get path to ingress YAML file.

    Args:
        deployment_type: Type of deployment (e.g., 'vllm-agg', 'vllm-disagg')

    Returns:
        str: Path to ingress YAML file
    """
    return os.path.join(INGRESS_DIR, f"{deployment_type}-ingress.yaml")


def get_benchmark_path(deployment_type: str) -> str:
    """Get path to benchmark results directory.

    Args:
        deployment_type: Type of deployment (e.g., 'vllm-agg', 'vllm-disagg')

    Returns:
        str: Path to benchmark results directory
    """
    return os.path.join(BENCHMARK_DIR, f"{deployment_type}-benchmark")
