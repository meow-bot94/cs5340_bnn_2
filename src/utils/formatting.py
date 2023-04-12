from typing import Union


def snake_to_pascal_case(app_name):
    return app_name.title().replace('_', '')


def infer_json_boolean(json_str: Union[str, bool]):
    if json_str in (True, False):
        return json_str
    json_str_lower = json_str.lower()
    return {
        'true': True,
        'false': False,
    }[json_str_lower]
