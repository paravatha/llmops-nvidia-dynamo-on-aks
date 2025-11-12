import os

from huggingface_hub.errors import HfHubHTTPError


def get_hf_token():
    """Fetch token from environment or ask the user for input."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("No HF_TOKEN found in environment.")
        token = input("Please enter your Hugging Face token: ").strip()
    return token


def check_token_validity(api, token):
    """Check if the Hugging Face token is valid."""
    try:
        user = api.whoami(token=token)
        print("Token is valid.")
        return True
    except Exception as e:
        print("Invalid Hugging Face token.")
        print("Error:", e)
        return False


def check_model_access(api, token, model_name):
    """Check if user has access to a specific model."""
    try:
        model_info = api.model_info(repo_id=model_name, token=token)
        print(f"Access granted to model: {model_name}")
        return True
    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            print("Unauthorized: Invalid or missing token.")
        elif e.response.status_code == 403:
            print(f"Forbidden: You don’t have permission to access {model_name}.")
        elif e.response.status_code == 404:
            print(f"Model {model_name} not found (private or misspelled).")
        else:
            print("Unexpected error:", e)
        return False
