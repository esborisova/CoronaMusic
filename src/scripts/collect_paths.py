import os
from typing import List

def save_filepaths(root_dir: str) -> List[str]:
    files_paths = []
    for root in root_dir:
        for file in os.listdir(root):
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                files_paths.append(file_path)
    return files_paths
