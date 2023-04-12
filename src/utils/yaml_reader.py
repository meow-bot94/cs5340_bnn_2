import string
from pathlib import Path
from typing import Callable, Dict, Union

import yaml


class YamlReader:
    @staticmethod
    def read_yaml_config(
            config_path: Union[Path, str],
            substitution: Dict[str, str] = None,
    ):
        if substitution is None:
            return YamlReader._read_yaml_config(config_path)
        string_constr = YamlReader._generate_string_constructor(substitution)
        return YamlReader._read_yaml_config_with_constructor(config_path, string_constr)

    @staticmethod
    def get_safe_loader():
        class WrappedSafeLoader(yaml.SafeLoader):
            pass
        return WrappedSafeLoader

    @staticmethod
    def _read_yaml_config_with_constructor(
            config_path: Union[Path, str],
            constructor: Callable,
    ):
        loader = YamlReader.get_safe_loader()
        loader.add_constructor('tag:yaml.org,2002:str', constructor)

        token_re = string.Template.pattern
        loader.add_implicit_resolver('tag:yaml.org,2002:str', token_re, None)

        with open(config_path, 'rt') as f:
            config = yaml.load(f.read(), Loader=loader)
        return config

    @staticmethod
    def _generate_string_constructor(substitution: Dict[str, str]) -> Callable:
        # Reference: https://dustinoprea.com/2022/04/01/python-substitute-values-into-yaml-during-parse/
        def string_constructor(loader, node):
            t = string.Template(node.value)
            value = t.substitute(substitution)
            return value
        return string_constructor

    @staticmethod
    def _read_yaml_config(config_path: Union[Path, str]):
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        return config
