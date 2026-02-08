
import os
from huggingface_hub import HfApi
import sys

def main():
    try:
        api = HfApi()
        repo_id = "SmayanKulkarni/encs-systemic-risk"
        print(f"Checking status for {repo_id}...")
        
        # Get space runtime info
        # Note: This might require a token with read permissions if the space is private.
        # But since we can upload, we likely have a token in cache or env.
        runtime = api.get_space_runtime(repo_id)
        
        print(f"Stage: {runtime.stage}")
        print(f"Hardware: {runtime.hardware}")
        if runtime.stage == "BUILDING":
            print("Status: Building")
        elif runtime.stage == "RUNNING":
            print("Status: Running")
        elif runtime.stage == "APP_STARTING":
            print("Status: App Starting")
        else:
            print(f"Status: {runtime.stage}")
            
    except Exception as e:
        print(f"Error checking status: {e}")

if __name__ == "__main__":
    main()
