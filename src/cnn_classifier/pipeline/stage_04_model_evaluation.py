"""Evaluation pipeline for the CNN classifier."""

from cnn_classifier import logger
from cnn_classifier.components.evaluation import Evaluation
from cnn_classifier.config.configuration import ConfigurationManager

STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    """Pipeline for evaluating the performance of the CNN classifier."""

    def __init__(self) -> None:
        """Initialize the EvaluationPipeline class."""

    def main(self) -> None:
        """Run the evaluation pipeline."""
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise
