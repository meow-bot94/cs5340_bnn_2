from abc import ABC
from typing import Dict, Any


class Monostate(ABC):
    __monostate = dict()

    def __init__(self):
        if not self.__class__.__monostate:
            self.__class__.__monostate = self.__dict__
            self.__dict__.update(self.monostate_defaults)
        else:
            self.__dict__ = self.__class__.__monostate

    @property
    def monostate_defaults(self) -> Dict[str, Any]:
        return dict()
