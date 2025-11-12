# llmops-nvidia-dynamo-on-aks

## Overview

This repository provides resources and instructions to deploy and manage Large Language Model (LLM) operations on Azure Kubernetes Service (AKS) with NVIDIA GPU Operator and Dynamo. It enables scalable, GPU-accelerated LLM workflows in the cloud.

## Prerequisites

- Azure subscription
- Azure CLI installed
- kubectl installed
- Access to GPU-enabled VM sizes in your Azure region
- Helm (for some deployments)

## Setup Instructions

1. **Provision AKS Cluster with GPU Nodes**
   - Create an AKS cluster with GPU-enabled node pools using Azure CLI or provided scripts.

2. **Install NVIDIA GPU Operator**
   - Deploy the NVIDIA GPU Operator to manage GPU resources on AKS. Use Helm or official manifests.

3. **Deploy Dynamo**
   - Install Dynamo on your AKS cluster using provided manifests or Helm charts.

4. **Configure LLMOps Workflows**
   - Use example YAML files to set up LLM pipelines and GPU workloads.

## Usage

- Monitor GPU resources via AKS and NVIDIA dashboards.
- Run LLM inference and training jobs using Dynamo.
- Scale workloads using AKS features.

## Resources

- [AKS Documentation](https://docs.microsoft.com/en-us/azure/aks/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)
- [Dynamo Documentation](https://github.com/dynamo-org/dynamo)
