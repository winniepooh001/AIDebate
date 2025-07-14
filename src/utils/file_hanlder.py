import time
from pathlib import Path
import os
import glob

def delete_files_older_than(days: int, folder: str):
    """
    Deletes all files in the given folder older than `days`.

    Args:
        days (int): Number of days before a file is considered old.
        folder (str): Path to the folder to clean up.
    """
    now = time.time()
    cutoff = now - (days * 86400)  # 86400 seconds in a day

    folder_path = Path(folder).resolve()

    if not folder_path.exists() or not folder_path.is_dir():
        print(f"[WARNING] Folder not found: {folder_path}")
        return

    for file in folder_path.iterdir():
        if file.is_file():
            if file.stat().st_mtime < cutoff:
                print(f"Deleting old file: {file}")
                file.unlink()




def find_latest_file(folder_path, pattern="*"):
    # Get full paths of all files matching the pattern
    files = find_all_files(folder_path, pattern)

    # Find the file with the latest modification time
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def find_all_files(folder_path, pattern="*"):
    if "*" not in pattern:
        pattern = f"*{pattern}"

    files = glob.glob(os.path.join(folder_path, pattern))
    if not files:
        return []  # No files found

    return files
