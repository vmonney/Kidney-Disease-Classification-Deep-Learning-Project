"""Train the model.

This file is used to train the model. It uses the
configuration file to get the parameters.
"""
from cnn_classifier import logger
from cnn_classifier.components.model_training import Training
from cnn_classifier.config.configuration import ConfigurationManager

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    """ModelTrainingPipeline class."""

    def __init__(self) -> None:
        """__init__ method."""

    def main(self) -> None:
        """Train the model."""
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise
