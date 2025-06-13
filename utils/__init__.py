import torchvision
import torchvision.transforms as transforms


imagenet_reverse_transform = transforms.Normalize(
    mean=[-m/s for m, s in zip([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])],
    std=[1/s for s in [0.229, 0.224, 0.225]]
)


cifar100_reverse_transform = transforms.Normalize(
    mean=[-m / s for m, s in zip([0.509, 0.487, 0.442], [0.202, 0.200, 0.204])],
    std=[1 / s for s in [0.202, 0.200, 0.204]]
)
