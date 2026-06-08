import os
import re
import gdown

# Environment configuration
## Define the input URL and the local target directory
FOLDER_URL = "https://drive.google.com/drive/folders/1gF4fc0dBgXm5cQ41rYSYj7wxtw13G04I?usp=sharing"
TARGET_DIR = "./data"

# Resource identifier extraction
## Use a regular expression to capture the folder ID from the URL
match = re.search(r"folders/([a-zA-Z0-9-_]+)", FOLDER_URL)
if not match:
    raise ValueError("Invalid Google Drive URL format.")

folder_id = match.group(1)

# Local structure validation and preparation
## Create the base directory if it does not exist to prevent overwriting
if not os.path.exists(TARGET_DIR):
    print(f"Creating directory: {TARGET_DIR}")
    os.makedirs(TARGET_DIR)
else:
    print(f"Directory {TARGET_DIR} already exists. Proceeding with download.")

# Download execution
## Invoke recursive download using gdown
## The remaining_ok=True flag prevents deletion of existing local files
print(f"Starting download for folder ID: {folder_id}")
gdown.download_folder(
    id=folder_id,
    output=TARGET_DIR,
    quiet=False,
    remaining_ok=True
)
print("Download process completed.")