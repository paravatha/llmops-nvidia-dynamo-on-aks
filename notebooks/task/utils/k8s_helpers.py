"""
Kubernetes Helper Functions

This module provides helper functions for working with Kubernetes deployments,
particularly for monitoring pod status and health.
"""

import subprocess
import time
from typing import Dict, List


def wait_for_pods_ready(
    deployment_name: str, namespace: str, timeout_minutes: int = 45
) -> bool:
    """Wait for all pods in a deployment to be ready with detailed status reporting.

    Args:
        deployment_name: Name of the Dynamo deployment
        namespace: Kubernetes namespace where pods are deployed
        timeout_minutes: Maximum time to wait for pods to be ready

    Returns:
        bool: True if all pods are ready, False if timeout occurs
    """
    print(
        f"⏳ Waiting for {deployment_name} pods to be ready (timeout: {timeout_minutes} minutes)..."
    )

    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    while time.time() - start_time < timeout_seconds:
        # Get pod status
        cmd = [
            "kubectl",
            "get",
            "pods",
            "-l",
            f"nvidia.com/dynamo-namespace={deployment_name}",
            "-n",
            namespace,
            "--no-headers",
        ]

        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            pods_status = result.stdout.strip().splitlines()
        except Exception as e:
            print(f"   ❌ Error getting pod status: {e}")
            time.sleep(10)
            continue

        if not pods_status:
            print("   📋 No pods found yet, waiting...")
            time.sleep(10)
            continue

        ready_count = 0
        total_count = len(pods_status)

        print("\n   📊 Pod Status Update:")
        for pod_line in pods_status:
            parts = pod_line.split()
            if len(parts) >= 3:
                pod_name = parts[0]
                ready = parts[1]
                status = parts[2]
                print(f"   • {pod_name}: {status} ({ready})")

                if status == "Running" and "/" in ready:
                    ready_nums = ready.split("/")
                    if ready_nums[0] == ready_nums[1] and int(ready_nums[0]) > 0:
                        ready_count += 1

        if ready_count == total_count and total_count > 0:
            print(f"\n✅ All {total_count} pods are ready!")
            return True

        print(f"   ⏳ {ready_count}/{total_count} pods ready, waiting...")
        time.sleep(15)

    print(f"❌ Timeout waiting for pods to be ready after {timeout_minutes} minutes")
    return False


def get_pod_status(namespace: str, label_selector: str) -> List[Dict[str, str]]:
    """Get status of pods matching a label selector.

    Args:
        namespace: Kubernetes namespace to search in
        label_selector: Label selector to filter pods

    Returns:
        list: List of pod status dictionaries with name, ready state, and status
    """
    try:
        cmd = [
            "kubectl",
            "get",
            "pods",
            "-l",
            label_selector,
            "-n",
            namespace,
            "--no-headers",
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            return []

        pods = []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                pods.append({"name": parts[0], "ready": parts[1], "status": parts[2]})
        return pods
    except Exception:
        return []
