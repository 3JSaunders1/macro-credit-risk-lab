"""
utils/logger.py
---------------
Simple production-style logger used across pipeline.
"""

import logging


def get_logger(name: str):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger