# file_structure.py
import os

SKIP_DIRS = {"venv", "__pycache__", ".git"}
SKIP_FILES = {".DS_Store"}

def print_structure(path=".", indent=""):
    for item in sorted(os.listdir(path)):
        if item in SKIP_FILES:
            continue

        item_path = os.path.join(path, item)

        if os.path.isdir(item_path):
            if item in SKIP_DIRS:
                continue
            print(indent + item + "/")
            print_structure(item_path, indent + "    ")
        else:
            print(indent + item)

print_structure(".")