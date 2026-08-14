import logging

from cognexa.core.logging import get_logger, setup_logging


def test_get_logger():
    logger = get_logger("cognexa.test")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "cognexa.test"


def test_setup_logging():
    setup_logging()

    logger = get_logger("cognexa.test")

    assert logger.level == logging.NOTSET