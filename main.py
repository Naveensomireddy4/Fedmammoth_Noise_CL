import random
from typing import Tuple
import numpy as np
import setproctitle
from argparse import ArgumentParser
from inspect import signature
import os
import getpass
import lightning as L
import torch
import json
from torch.utils.data import Dataset
from types import SimpleNamespace

from _models._utils import BaseModel
from utils.training import train
from utils.args import add_args
from _models import model_factory
from _networks import network_factory
from _datasets import dataset_factory
from datetime import datetime


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GlobalLabelModWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes_per_task):
        self.dataset = dataset
        self.num_classes_per_task = num_classes_per_task
        print(f" inside GlobalLabelModWrapper  num_classes_per_task = {self.num_classes_per_task} ")

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y % self.num_classes_per_task

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)



def get_artifacts(args: dict, fabric) -> Tuple[BaseModel, Dataset]:
    NetworkClass = network_factory(args["network"])
    DatasetClass = dataset_factory(args["dataset"])
    ModelClass = model_factory(args["model"])

    args["input_shape"] = DatasetClass.INPUT_SHAPE
    if isinstance(DatasetClass.N_CLASSES_PER_TASK, list):
        args["num_classes"] = sum(DatasetClass.N_CLASSES_PER_TASK)
    else:
        args["num_classes"] = DatasetClass.N_CLASSES_PER_TASK * DatasetClass.N_TASKS
    args["num_tasks"] = DatasetClass.N_TASKS
    args["num_classes_per_task"]=DatasetClass.N_CLASSES_PER_TASK

    network_signature = list(signature(NetworkClass.__init__).parameters.keys())[1:]
    dataset_signature = list(signature(DatasetClass.__init__).parameters.keys())[1:]
    model_signature = list(signature(ModelClass.__init__).parameters.keys())[3:]
    # TODO Questo è un po' pericoloso, dobbiamo ricordarci sempre di mettere i primi 3 argomenti fissi e dopo i nostri argomenti, che ci sta eh

    dataset = DatasetClass(**{key: args[key] for key in dataset_signature})
    network = NetworkClass(**{key: args[key] for key in network_signature})

    # server_model = ModelClass(fabric, network, **{key: args[key] for key in model_signature})
    model_kwargs = {}
    for key in model_signature:
        if key == "args":
            model_kwargs["args"] = SimpleNamespace(**args)
        else:
            model_kwargs[key] = args[key]
    server_model = ModelClass(fabric, network, **model_kwargs)

    client_models = []
    active_clients = int(round(args["num_clients"] * args["participation_rate"]))

    for _ in range(active_clients):
        net = NetworkClass(**{key: args[key] for key in network_signature})

        client_models.append(
            ModelClass(fabric, net, **model_kwargs).to("cpu")
        )

    return server_model, client_models, dataset


def main(args: dict, output_folders_root: str, nickname: str) -> None:
    set_random_seed(args["random_seed"])

    setproctitle.setproctitle(f"{getpass.getuser()}_{nickname}")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    # command = " ".join(sys.argv)
    output_folder = os.path.join(output_folders_root, f"{timestamp}_{nickname}")
    if not args["debug_mode"]:
        os.makedirs(output_folder)

    device = args["device"]
    if "cuda" in device:
        device, index = device.split(":")
    torch.set_float32_matmul_precision("medium")
    fabric = L.Fabric(
        accelerator=device,
        devices=1 if device == "cpu" else [int(index)],
        strategy="dp",
        precision=args["precision"],
    )
    fabric.launch()

    server_model, client_models, dataset = get_artifacts(args, fabric)
    
    #dataset = GlobalLabelModWrapper(dataset, dataset.N_CLASSES_PER_TASK)
    for task in range(dataset.N_TASKS):
        unique_labels = set()

        train_loaders, test_loaders = dataset.get_cur_dataloaders(task)

        # Check train loaders
        for loader in train_loaders:
            for _, labels in loader:
                unique_labels.update(labels.cpu().numpy().tolist())

        # Check test loaders (optional but recommended)
        for loader in test_loaders:
            for _, labels in loader:
                unique_labels.update(labels.cpu().numpy().tolist())

        print(f"\nTask {task}:")
        print("Unique labels:", sorted(unique_labels))
        print("Total unique labels:", len(unique_labels))




    if not args["debug_mode"]:
        try:
            with open(os.path.join(output_folder, "config.json"), "w") as f:
                json.dump(args, f, indent=4)
        except Exception as e:
            print(f"Error while saving config: {e}, won't be saving it.")

    train(fabric, server_model, client_models, dataset, args, output_folder)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="fed-mammoth",
        allow_abbrev=False,
        conflict_handler="resolve",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--network", type=str, required=True)

    parser.add_argument(
        "--nickname", type=str, required=False, default="Moscow"
    )  # TODO: Change this to something more appropriate

    args = parser.parse_known_args()[0]

    args.nickname = str(args.nickname + "-" + args.model + "_" + args.dataset + "_" + args.network)

    add_args(parser, args.model, args.network, args.dataset)

    args = {**vars(parser.parse_args()), **vars(args)}

    print(
        """
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣿⣿⣿⣧⡀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⠀⠀⢀⣠⣴⡈⣡⣄⠉⣁⣉⢀⠀⠀⠀⠀⠀
    ⠀⠀⠀⢀⣠⣤⣼⣿⣿⣾⣿⣿⣿⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀
    ⠀⠀⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⣹⣿⣿⠀⠀⠀⠀
    ⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⢁⡀⠻⣿⣿⣿⡆⠀⡞⠀
    ⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠘⢿⣦⡈⠛⠿⠃⣼⠃⠀
    ⠀⢸⡿⣿⣿⣿⣿⣿⣿⠿⣿⣿⡿⠿⣿⣿⣿⣿⡟⠛⠁⠀⠙⠻⠷⠶⠟⠁⠀⠀
    ⠀⠘⠁⢸⣿⣿⣿⣿⠃⠰⠛⠛⠋⠀⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⣶⣶⠀⠀⠀⠀
    ⠀⠀⠀⢸⣿⣿⣿⡇⢰⣶⣿⠀⢸⡆⢸⣿⣿⣿⣟⠀⣀⣀⣠⣼⣿⡟⠀⠀⠀⠀
    ⠀⠀⠀⢸⣿⣿⣿⡇⢸⣿⣿⠀⢸⣧⠈⣿⣿⣿⡿⠀⢿⣿⠿⠛⠉⠀⠀⠀⠀⠀
    ⠀⠀⠀⢺⣿⣿⣿⠀⢸⣿⣿⠀⢸⣿⠀⢻⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠈⠉⠉⠉⠀⠈⠉⠉⠀⠈⠉⠁⠈⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    """
    )

    main(args, args["output_folders_root"], args["nickname"])
