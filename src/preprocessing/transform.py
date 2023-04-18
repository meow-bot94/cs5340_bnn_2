from torchvision import transforms


# normalize = transforms.Normalize(
#     [0.6484, 0.6158, 0.5781],
#     [0.1940, 0.1916, 0.2030]
# )
normalize = transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
)  # Imagenet standards

auto_augment = transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)

fast_transform = transforms.Compose([
    auto_augment,
    transforms.ToTensor(),
])

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
        # fast_transform,
        transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.Resize((224, 224), antialias=True),
            # transforms.RandomResizedCrop(size=224),
            # transforms.RandomRotation(degrees=15),
            # transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406],
            #                      [0.229, 0.224, 0.225])  # Imagenet standards
            normalize,
        ]),
    'test':
        # fast_transform,
        transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            # transforms.Resize(size=256),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            normalize
        ]),
    'validate':
        # fast_transform,
        transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            # transforms.Resize(size=256),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            normalize
        ]),
}
