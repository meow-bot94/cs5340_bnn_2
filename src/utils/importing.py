import importlib
import importlib.util


def import_py(name, path):
    if path is None:
        return
    if name is None:
        return
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
