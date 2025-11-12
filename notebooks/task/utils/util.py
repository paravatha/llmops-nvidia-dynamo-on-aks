"""
YAML Configuration Utilities

This module provides helper functions for working with YAML configuration files,
particularly for Dynamo deployments.
"""

import os
from typing import Any, Dict

import yaml


def update_yaml_config(
    source_yaml_path: str, destination_directory: str, replacements: Dict[str, Any]
) -> bool:
    """Update a YAML configuration file with provided replacements.

    Args:
        source_yaml_path: Path to the template YAML file
        destination_directory: Directory to save the configured YAML
        replacements: Dictionary of key-value pairs to replace in the YAML

    Returns:
        bool: True if configuration was created successfully, False otherwise
    """
    try:
        # Read the template YAML
        with open(source_yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # Apply replacements
        for key, value in replacements.items():
            try:
                if "." in key:
                    # Handle nested keys like 'spec.services.Frontend.image'
                    parts = key.split(".")
                    target = config
                    for part in parts[:-1]:
                        target = target[part]
                    target[parts[-1]] = value
                else:
                    config[key] = value
            except Exception as e:
                print(f"❌ Error applying replacement for key '{key}': {e}")
                print(f"   Value type: {type(value)}")
                print(f"   Value: {value}")
                raise

        # Create destination directory
        os.makedirs(destination_directory, exist_ok=True)

        # Write configured YAML
        output_path = os.path.join(
            destination_directory, os.path.basename(source_yaml_path)
        )
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"✅ Configuration file created successfully at {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        import traceback

        traceback.print_exc()
        return False


def image_model_replacement(
    source_yaml_path: str, destination_directory: str, vllm_image: str, model_name: str
) -> bool:
    """Create a configured YAML file with substituted vLLM image and model name.

    This function takes a template YAML file for vLLM deployment and creates a new
    configuration with the specified vLLM image and model name. It updates both
    the Frontend and VllmDecodeWorker containers.

    Args:
        source_yaml_path: Path to the template YAML file
        destination_directory: Directory to save the configured YAML
        vllm_image: Docker image to use for vLLM containers
        model_name: Name of the model to deploy

    Returns:
        bool: True if configuration was created successfully, False otherwise
    """
    try:
        # Read the template YAML to check structure
        print(f"📖 Reading configuration from: {source_yaml_path}")
        with open(source_yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # Check if it's a disaggregated deployment
        services = config.get("spec", {}).get("services", {})
        has_prefill_worker = "VllmPrefillWorker" in services

        print(f"📋 Detected services: {list(services.keys())}")
        print(f"🔍 Has prefill worker: {has_prefill_worker}")

        # Base replacements for all deployments
        replacements = {}

        # Update Frontend image
        replacements["spec.services.Frontend.extraPodSpec.mainContainer.image"] = (
            vllm_image
        )

        # Update VllmDecodeWorker image
        replacements[
            "spec.services.VllmDecodeWorker.extraPodSpec.mainContainer.image"
        ] = vllm_image

        # Update VllmDecodeWorker args
        decode_args = [
            f"python3 -m dynamo.vllm --model {model_name} 2>&1 | tee /tmp/vllm.log"
        ]
        replacements[
            "spec.services.VllmDecodeWorker.extraPodSpec.mainContainer.args"
        ] = decode_args

        # Add prefill worker configuration if present
        if has_prefill_worker:
            print("🔄 Adding prefill worker configuration...")
            prefill_args = [
                f"python3 -m dynamo.vllm --model {model_name} --is-prefill-worker 2>&1 | tee /tmp/vllm.log"
            ]
            replacements[
                "spec.services.VllmPrefillWorker.extraPodSpec.mainContainer.image"
            ] = vllm_image
            replacements[
                "spec.services.VllmPrefillWorker.extraPodSpec.mainContainer.args"
            ] = prefill_args

        print(f"📝 Applying {len(replacements)} replacements...")
        return update_yaml_config(source_yaml_path, destination_directory, replacements)

    except Exception as e:
        print(f"Error in image_model_replacement: {e}")
        import traceback

        traceback.print_exc()
        return False
