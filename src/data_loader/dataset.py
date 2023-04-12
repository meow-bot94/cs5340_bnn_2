from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.preprocessing.transform import image_transforms


@dataclass
class Dataset:
    root_dir: Path
    batch_size: int

    train_dataset: ImageFolder = field(init=False)
    test_dataset: ImageFolder = field(init=False)
    validate_dataset: ImageFolder = field(init=False)

    train_dataloader: DataLoader = field(init=False)
    test_dataloader: DataLoader = field(init=False)
    validate_dataloader: DataLoader = field(init=False)

    train_datasize: int = field(init=False)
    test_datasize: int = field(init=False)
    validate_datasize: int = field(init=False)

    class_to_idx: Dict[str, int] = field(init=False)
    idx_to_class: Dict[int, str] = field(init=False)
    num_classes: int = field(init=False)

    @property
    def num_workers(self) -> int:
        return 4

    @property
    def shuffle(self) -> bool:
        return True

    @property
    def data_categories(self) -> List[str]:
        return ['train', 'test', 'validate']

    def _init_image_folders(self):
        for category in self.data_categories:
            dataset = ImageFolder(
                root=(self.root_dir / category).as_posix(),
                transform=image_transforms[category],
            )
            attr_name = f'{category}_dataset'
            setattr(self, attr_name, dataset)

    def _init_data_loaders(self):
        for category in self.data_categories:
            attr_name = f'{category}_dataloader'
            data_loader = DataLoader(
                getattr(self, f'{category}_dataset'),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
            )
            setattr(self, attr_name, data_loader)

    def _calc_dataset_sizes(self):
        for category in self.data_categories:
            attr_name = f'{category}_datasize'
            loader = getattr(self, f'{category}_dataloader')
            datasize = len(loader.dataset)
            setattr(self, attr_name, datasize)

    def _init_class_mapping(self):
        self.class_to_idx = self.train_dataset.class_to_idx
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.num_classes = len(self.idx_to_class)

    def __post_init__(self):
        self._init_image_folders()
        self._init_data_loaders()
        self._calc_dataset_sizes()
        self._init_class_mapping()
