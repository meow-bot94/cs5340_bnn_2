from pathlib import Path

from src.data_loader.dataset_getter import DatasetGetter


def test_dataset_getter():
    temp_image_path = Path('/home/hdfsf10n/menquan/bayesian_test/data')
    batch_size = 12
    dataset = DatasetGetter(temp_image_path, batch_size).get()
    print(dataset)
    assert True
