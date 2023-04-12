import logging
from typing import Union


class EpochScorePrinter:
    logger = logging.getLogger(__name__)

    @classmethod
    def print(
            cls,
            epoch: int,
            total_epoch: int,
            verbose=True,
            **kwargs,
    ):
        if not verbose:
            return
        msg = f'[Epoch {epoch:d}/{total_epoch:d}]: '
        extra_msg = list(cls._format_key_value(k, v) for k, v in kwargs.items())
        overall_msg = ''.join([msg, *extra_msg])
        cls.logger.info(overall_msg)

    @classmethod
    def _format_key_value(cls, key: str, value: Union[str, float, int]) -> str:
        if isinstance(value, str):
            return f'{key}: {value} '
        if isinstance(value, int):
            return f'{key}: {value:d} '
        if isinstance(value, float):
            return f'{key}: {value:.3f} '
        else:
            return f'{key}: {value} '