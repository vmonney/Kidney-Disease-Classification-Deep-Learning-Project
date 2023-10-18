"""cnn_classifier package.

The module sets up logging to a file and to the console, and provides a
logger object for use throughout the package.
"""

import logging
import sys
from pathlib import Path

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"

log_dir = Path("logs")
log_filepath = Path(log_dir) / "running_logs.log"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("cnnClassifierLogger")
