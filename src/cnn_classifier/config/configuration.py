"""Configuration settings."""
from pathlib import Path

from cnn_classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnn_classifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
)
from cnn_classifier.utils.common import create_directories, read_yaml


class ConfigurationManager:
    """Class for managing configuration settings."""

    def __init__(
        self,
        config_filepath: str = CONFIG_FILE_PATH,
        params_filepath: str = PARAMS_FILE_PATH,
    ) -> None:
        """Initialize the ConfigurationManager class.

        Args:
        ----
            config_filepath: The path to the configuration file.
            params_filepath: The path to the parameters file.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Return the data ingestion configuration."""
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """Return the prepare base model configuration."""
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )

    def get_training_config(self) -> TrainingConfig:
        """Return the training configuration."""
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = (
            Path(self.config.data_ingestion.unzip_dir)
            / "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        )
        create_directories([Path(training.root_dir)])

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
        )
