"""Creates a list of files and directories."""
import logging
from pathlib import Path

# logging string
logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")


project_name = "cnn_classifier"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html",
    "pyproject.toml",
]


for file_path in list_of_files:
    filepath = Path(file_path)
    filedir, filename = filepath.parent, filepath.name

    if file_path.endswith("/"):
        filepath.mkdir(parents=True, exist_ok=True)
        logging.info("Created directory: %s", filepath)
        continue

    # Ensure parent directories exist
    if filedir and not filedir.exists():
        filedir.mkdir(parents=True, exist_ok=True)
        logging.info("Created directory: %s", filedir)

    if not filepath.exists():
        with filepath.open("w"):
            logging.info("Creating empty file: %s", filename)
    else:
        logging.info("%s already exists", filepath)
