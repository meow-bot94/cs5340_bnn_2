import logging

from src.logger.logger_initializer import LoggerInitializer


def test_logger():
    LoggerInitializer().init()

    logger = logging.getLogger(__name__)
    print(f'{__name__=}')
    logger.info('info only')
    logger.warning('warning only')
    logger.error('error only')
    assert True
