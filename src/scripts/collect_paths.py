import os
from typing import List

def save_filepaths(root_dir: str) -> List[str]:
    files_paths = []
    for file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file)
        if os.path.isfile(file_path):
            files_paths.append(file_path)
    return files_paths
