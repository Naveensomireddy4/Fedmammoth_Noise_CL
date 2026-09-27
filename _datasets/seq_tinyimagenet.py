import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import sys

from _datasets import register_dataset
from _datasets._utils import BaseDataset
from utils.global_consts import DATASET_PATH
from kornia import augmentation as K

TRANSFORMS = {
    "default_train" : K.AugmentationSequential(
        K.RandomResizedCrop(size=(224, 224), resample="bicubic"),
        K.RandomHorizontalFlip(),
        K.Normalize(mean=(0.4802, 0.4480, 0.3975), std=(0.2770, 0.2691, 0.2821)),
    ),
    "default_test" : K.AugmentationSequential(
        K.Resize(size=(256, 256), resample="bicubic"),
        K.CenterCrop(size=(224, 224)),
        K.Normalize(mean=(0.4802, 0.4480, 0.3975), std=(0.2770, 0.2691, 0.2821)),
    ),
}


class MyTinyImageNet(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: transforms = None,
        download: bool = True,
    ) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

        if not os.path.exists(os.path.join(self.root, "TinyImageNet", "processed", "x_train_01.npy")) and self.download:
            import subprocess
            import zipfile
            from PIL import Image as PILImage

            raw_zip = os.path.join(self.root, "tiny-imagenet-200.zip")
            raw_dir = os.path.join(self.root, "tiny-imagenet-200")
            out_dir = os.path.join(self.root, "TinyImageNet", "processed")
            os.makedirs(out_dir, exist_ok=True)

            # Download
            if not os.path.exists(raw_zip):
                print("Downloading TinyImageNet...", file=sys.stderr)
                subprocess.run(
                    ["wget", "-q", "--show-progress", "http://cs231n.stanford.edu/tiny-imagenet-200.zip", "-O", raw_zip],
                    check=True,
                )

            # Unzip
            if not os.path.exists(raw_dir):
                print("Unzipping TinyImageNet...", file=sys.stderr)
                with zipfile.ZipFile(raw_zip, "r") as z:
                    z.extractall(self.root)

            # Preprocess into .npy chunks
            print("Preprocessing TinyImageNet into .npy chunks...", file=sys.stderr)

            classes = sorted(os.listdir(os.path.join(raw_dir, "train")))
            class_to_idx = {c: i for i, c in enumerate(classes)}

            # Train
            train_imgs, train_labels = [], []
            for cls in classes:
                img_dir = os.path.join(raw_dir, "train", cls, "images")
                for fname in sorted(os.listdir(img_dir)):
                    img = PILImage.open(os.path.join(img_dir, fname)).convert("RGB")
                    train_imgs.append(np.array(img, dtype=np.float32) / 255.0)
                    train_labels.append(class_to_idx[cls])
            train_imgs = np.array(train_imgs)
            train_labels = np.array(train_labels)
            for i, (x, y) in enumerate(zip(np.array_split(train_imgs, 20), np.array_split(train_labels, 20))):
                np.save(os.path.join(out_dir, f"x_train_{i+1:02d}.npy"), x)
                np.save(os.path.join(out_dir, f"y_train_{i+1:02d}.npy"), y)

            # Val
            val_imgs, val_labels = [], []
            with open(os.path.join(raw_dir, "val", "val_annotations.txt")) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    fname, cls = parts[0], parts[1]
                    img = PILImage.open(os.path.join(raw_dir, "val", "images", fname)).convert("RGB")
                    val_imgs.append(np.array(img, dtype=np.float32) / 255.0)
                    val_labels.append(class_to_idx[cls])
            val_imgs = np.array(val_imgs)
            val_labels = np.array(val_labels)
            for i, (x, y) in enumerate(zip(np.array_split(val_imgs, 20), np.array_split(val_labels, 20))):
                np.save(os.path.join(out_dir, f"x_val_{i+1:02d}.npy"), x)
                np.save(os.path.join(out_dir, f"y_val_{i+1:02d}.npy"), y)

            print("Preprocessing done.", file=sys.stderr)

        self.data = []
        for num in range(20):
            self.data.append(
                np.load(
                    os.path.join(
                        self.root + "/TinyImageNet",
                        "processed/x_%s_%02d.npy" % ("train" if self.train else "val", num + 1),
                    )
                )
            )
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(
                np.load(
                    os.path.join(
                        self.root + "/TinyImageNet",
                        "processed/y_%s_%02d.npy" % ("train" if self.train else "val", num + 1),
                    )
                )
            )
        self.targets = np.concatenate(np.array(self.targets)).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this to have a PIL Image for the transforms
        # img = Image.fromarray(np.uint8(255 * img))
        # Altermatively, we can use the following line to convert the image to PIL format
        # transform = transforms.Compose([transforms.ToPILImage(), self.TRANSFORM])

        if self.transform is not None:
            img = self.transform(img)

        return img, target


@register_dataset("seq-tinyimagenet")
class SequentialTinyImageNet(BaseDataset):
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10

    MEAN, STD = (0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)
    """
    TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    """
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    TRAIN_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(224, 224), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    BASE_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(size=(224, 224), interpolation=3),
            transforms.ToTensor(),
        ]
    )

    INPUT_SHAPE = (224, 224, 3)

    def train_transform(self, x):
        return TRANSFORMS[self.train_transf](x)

    def test_transform(self, x):
        return TRANSFORMS[self.test_transf](x)
    
    def __init__(
        self,
        num_clients: int,
        batch_size: int,
        train_transform: str = "default_train",
        test_transform: str = "default_test",
        partition_mode: str = "distribution",
        distribution_alpha: float = 0.05,
        class_quantity: int = 4,
    ):
        super().__init__(
            num_clients,
            batch_size,
            partition_mode,
            distribution_alpha,
            class_quantity,
        )
        self.train_transf = train_transform
        self.test_transf = test_transform

        for split in ["train", "test"]:
            dataset = MyTinyImageNet(
                DATASET_PATH,
                train=True if split == "train" else False,
                download=True,
                transform=getattr(self, f"{split.upper()}_TRANSFORM"),
            )
            dataset.classes = [i for i in range(200)]                               # Added for LIVAR compatibility
            dataset.class_to_idx = {cl: i for i, cl in enumerate(dataset.classes)}  # Added for LIVAR compatibility
            setattr(self, f"{split}_dataset", dataset)

        self._split_fcil(
            num_clients,
            partition_mode,
            distribution_alpha,
            class_quantity,
        )

        for split in ["train", "test"]:
            getattr(self, f"{split}_dataset").data = None
            getattr(self, f"{split}_dataset").targets = None


@register_dataset("joint-tinyimagenet")
class JointTinyImageNet(SequentialTinyImageNet):
    N_CLASSES_PER_TASK = 200
    N_TASKS = 1

