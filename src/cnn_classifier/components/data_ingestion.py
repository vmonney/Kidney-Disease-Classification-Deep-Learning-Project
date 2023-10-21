"""DataIngestion class, which download and extract data for the CNN classifier."""

import shutil
import zipfile
from pathlib import Path

import kaggle

from cnn_classifier import logger
from cnn_classifier.entity.config_entity import DataIngestionConfig
from cnn_classifier.utils.common import get_size, load_json

kaggle_config_path = Path.home() / ".kaggle" / "kaggle.json"
config = load_json(kaggle_config_path)
kaggle_username = config.username
kaggle_key = config.key


class DataIngestion:
    """Class for downloading and extracting data for the CNN classifier."""

    def __init__(self, config: DataIngestionConfig) -> None:
        """Initialize the DataIngestion class with a configuration object."""
        self.config = config

    def download_file(self) -> None:
        """Download file from URL if not exist locally."""
        if not Path(self.config.local_data_file).exists():
            dataset_path = self.config.source_url
            destination_folder = self.config.root_dir
            kaggle.api.dataset_download_files(
                dataset_path,
                path=destination_folder,
                unzip=False,
            )

            logger.info("zip data downloaded!")
        else:
            logger.info(
                f"zip data already exists with size: "
                f"{get_size(Path(self.config.local_data_file))}",
            )

    def remove_redundant_directory(self) -> None:
        """Remove the redundant nested directory if same names as parent."""
        base_path = Path(self.config.unzip_dir)
        nested_path = (
            base_path
            / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
            / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        )

        if nested_path.exists() and nested_path.name == nested_path.parent.name:
            # Move all contents of the nested directory to its parent directory
            for item in nested_path.iterdir():
                shutil.move(str(item), str(nested_path.parent))

            # Remove the now-empty nested directory
            nested_path.rmdir()

            logger.info("Redundant nested directory removed!")

    def extract_zip_file(self) -> None:
        """Extract the zip file into the data directory.

        Returns
        -------
          None
        """
        unzip_path = self.config.unzip_dir
        Path(unzip_path).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        self.remove_redundant_directory()
