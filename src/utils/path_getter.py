import inspect
import pathlib


class PathGetter:

    @staticmethod
    def get_source_directory() -> pathlib.Path:
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        dir_path = pathlib.Path(filename).absolute().parent.parent
        return dir_path

    @staticmethod
    def get_root_directory() -> pathlib.Path:
        source_dir = PathGetter.get_source_directory()
        return source_dir.parent

    @staticmethod
    def get_config_directory() -> pathlib.Path:
        root_dir = PathGetter.get_root_directory()
        return root_dir / 'config'

    @staticmethod
    def get_log_directory() -> pathlib.Path:
        root_dir = PathGetter.get_root_directory()
        return root_dir / 'logs'

    @staticmethod
    def get_model_directory() -> pathlib.Path:
        source_dir = PathGetter.get_source_directory()
        return source_dir / 'models'

    @staticmethod
    def get_assets_directory() -> pathlib.Path:
        root_dir = PathGetter.get_root_directory()
        return root_dir / 'assets'

    @staticmethod
    def get_data_directory() -> pathlib.Path:
        root_dir = PathGetter.get_root_directory()
        return root_dir / 'data'
