"""Main Script.

This module contains the main script for the
Kidney Classification project.

The script imports the following classes from their respective modules:
    - DataIngestionTrainingPipeline from cnn_classifier.pipeline.stage_01_data_ingestion
    - ProcessDataPipeline from cnn_classifier.pipeline.stage_02_data_processing
    - PrepareBaseModelTrainingPipeline from cnn_classifier.pipeline.
    stage_03_prepare_base_model
    - ModelTrainingPipeline from cnn_classifier.pipeline.stage_04_training
    - EvaluationPipeline from cnn_classifier.pipeline.stage_05_evaluation

The script runs the main method of each class in the following order:
    1. DataIngestionTrainingPipeline
    2. ProcessDataPipeline
    3. PrepareBaseModelTrainingPipeline
    4. ModelTrainingPipeline
    5. EvaluationPipeline
"""

from cnn_classifier import logger
from cnn_classifier.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from cnn_classifier.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)
from cnn_classifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnn_classifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise


STAGE_NAME = "Prepare base model"

try:
    logger.info("*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise

STAGE_NAME = "Training"
try:
    logger.info("*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise


STAGE_NAME = "Evaluation stage"
try:
    logger.info("*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evaluation = EvaluationPipeline()
    model_evaluation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise
