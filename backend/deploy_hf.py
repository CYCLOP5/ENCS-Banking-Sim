
import logging
import sys
from huggingface_hub import upload_folder

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("huggingface_hub")
logger.setLevel(logging.INFO)

print("Starting upload...")

try:
    upload_folder(
        folder_path='/home/smayan/Datathon/lowertaperfade-datahack-2026/backend',
        repo_id='SmayanKulkarni/encs-systemic-risk',
        repo_type='space',
        ignore_patterns=['.git', '.git/*', '*.db', '__pycache__', 'data/output/*.db']
    )
    print("Upload successful!")
except Exception as e:
    print(f"Upload failed: {e}")
    sys.exit(1)
