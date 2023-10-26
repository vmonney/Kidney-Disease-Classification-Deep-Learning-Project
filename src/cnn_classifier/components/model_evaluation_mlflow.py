"""Module for evaluating a trained model."""

from urllib.parse import urlparse

import mlflow
import Pathlib as Path
import tensorflow as tf

from cnn_classifier.entity.config_entity import EvaluationConfig
from cnn_classifier.utils.common import save_json


class Evaluation:
    """Class for evaluating a trained model."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize the Evaluation class.

        Args:
        ----
          config (EvaluationConfig): An instance of EvaluationConfig
          containing the configuration for evaluation.
        """
        self.config = config

    def _valid_generator(self) -> None:
        datagenerator_kwargs = {"rescale": 1.0 / 255, "validation_split": 0.30}

        dataflow_kwargs = {
            "target_size": self.config.params_image_size[:-1],
            "batch_size": self.config.params_batch_size,
            "interpolation": "bilinear",
        }

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs,
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """Load model."""
        return tf.keras.models.load_model(path)

    def evaluation(self) -> None:
        """Evaluate the trained model."""
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self) -> None:
        """Save the evaluation score to a JSON file."""
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self) -> None:
        """Log evaluation metrics and model into MLflow."""
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry,
                # which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name="VGG16Model",
                )
            else:
                mlflow.keras.log_model(self.model, "model")
