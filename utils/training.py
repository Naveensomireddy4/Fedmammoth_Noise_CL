import os
import torch
import wandb
from time import time
from typing import List

from _datasets._utils import BaseDataset
from utils.global_consts import LOG_LOSS_INTERVAL
from _models._utils import BaseModel
from utils.status import progress_bar
from utils.tools import get_time_str

import numpy as np
import matplotlib.pyplot as plt

def compute_forgetting(accuracies: List[List[float]]) -> List[float]:
    forgetting = []
    tasks = len(accuracies)
    last_accs = accuracies[-1]
    for i in range(tasks):
        forgetting.append(accuracies[i][i] - last_accs[i])
    return forgetting




def accs_and_forgetting_matrix(accuracies: List[List[float]], forgetting: List[float], output_folder: str = None) -> None:
    if output_folder is None:
        output_folder = os.getcwd()
        print("Output folder not specified, saving in current directory:", output_folder)
    tasks = len(accuracies)
    #fill list of accuracies with zeros
    for acc in accuracies:
        while len(acc) < tasks:
            acc.append(0.0)
    accs = np.array(accuracies)
    fig, ax = plt.subplots()
    ax.matshow(accs, cmap='viridis')
    for i in range(tasks):
        for j in range(tasks):
            ax.text(j, i, f'{accs[i, j]:.2f}', ha='center', va='center', color='black')
    plt.title("Task accuracies")
    plt.xlabel("Task")
    plt.ylabel("Task")
    acc_path = os.path.join(output_folder, "task_accuracies.png")
    plt.savefig(acc_path)
    plt.close()

    #now we plot forgetting as a vector, similar to what we did with accuracies
    fig, ax = plt.subplots()
    forgetting.append(round(sum(forgetting) / len(forgetting), 2))
    ax.matshow(np.array([forgetting]), cmap='viridis')
    for i in range(tasks + 1):
        ax.text(i, 0, f'{forgetting[i]:.2f}', ha='center', va='center', color='black')
    plt.title("Forgetting")
    plt.xlabel("Task")
    plt.ylabel("Forgetting")
    forg_path = os.path.join(output_folder, "forgetting.png")
    plt.savefig(forg_path)
    return [acc_path, forg_path]

def evaluate(fabric, task, model: BaseModel, dataset: BaseDataset, return_responses=False):
    correct, total = 0, 0
    task_accuracies = []

    training_status = model.training
    model.eval()

    labels_tensor = torch.tensor([], device=model.device)
    responses_tensor = torch.tensor([], device=model.device)

    with torch.no_grad():
        for t in range(task + 1):
            task_correct, task_total = 0, 0

            if dataset.IS_TEXT:
                test_loaders = dataset.get_cur_dataloaders_oos(t)[1]
            else:
                test_loaders = dataset.get_cur_dataloaders(t)[1]

            for test_loader in test_loaders:
                test_loader = fabric.setup_dataloaders(test_loader)
                #print(f"task:{t} size of testloader:{test_loader}")
                for inputs, labels in test_loader:
                    # print(f"labels:{labels[:10].cpu().tolist()}")
                   # print(f"task_id: {t}")
                    labels = labels
                    # print(f"labels:{labels[:10].cpu().tolist()}")
                    #inputs = dataset.test_transform(inputs)
                    outputs = model(inputs, task_id=t,str="test")
                    labels = labels 
                    preds = outputs.argmax(dim=1)

                    task_correct += (preds == labels).sum().item()
                    task_total += labels.size(0)

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    labels_tensor = torch.cat((labels_tensor, labels))
                    responses_tensor = torch.cat((responses_tensor, preds))
                    # print("Actual vs Predicted:")
                    # for actual, pred in zip(labels[:10].cpu().tolist(), preds[:10].cpu().tolist()):
                    #     print(f"{actual} -> {pred}")

            acc = round(task_correct / task_total * 100, 2) if task_total > 0 else 0.0
            task_accuracies.append(acc)
            #print(f" for task{t} total correct are {task_correct} out of {total}")
   
    model.train(training_status)
    print(
        f" [Global] Mean accuracy up to task {task + 1}:",
        round(correct / total * 100, 2),
        "%",
        "Task accuracies:",
        task_accuracies,
    )
    res = [round(correct / total * 100, 2), task_accuracies]
    return res if not return_responses else [res, labels_tensor, responses_tensor]


def evaluate_client(fabric, task, model: BaseModel, dataset: BaseDataset, idx: int):
    correct, total = 0, 0
    task_accuracies = []
    training_status = model.training
    model.eval()
    start_class = 0
    if isinstance(dataset.N_CLASSES_PER_TASK, list):
        end_class = sum(dataset.N_CLASSES_PER_TASK[: task + 1])
    else:
        end_class = (task + 1) * dataset.N_CLASSES_PER_TASK
    with torch.no_grad():
        for t in range(task + 1):
            task_correct, task_total = 0, 0
            if dataset.IS_TEXT:
                test_loader = dataset.get_cur_dataloaders_oos(t)[1][idx]
            else:
                test_loader = dataset.get_cur_dataloaders(t)[1][idx]
            test_loader = fabric.setup_dataloaders(test_loader)
            for inputs, labels in test_loader:
                # print(f"labels:{labels[:10].cpu().tolist()}")
#                print(f"task_id: {t}")
                labels = labels
                # print(f"labels:{labels[:10].cpu().tolist()}")
                #inputs = dataset.test_transform(inputs) 
                outputs = model(inputs,task_id=t,str="test")
                pred = torch.max(outputs, dim=1)[1]
                task_correct += (pred == labels).sum().item()
                task_total += labels.shape[0]
                correct += (pred == labels).sum().item()
                total += labels.shape[0]
                #print("Actual vs Predicted:")
                # for actual, pred in zip(labels[:10].cpu().tolist(), pred[:10].cpu().tolist()):
                #     print(f"{actual} -> {pred}")
            task_accuracies.append(round(task_correct / task_total * 100, 2))
            #print(f" for task{t} total correct are {task_correct} out of {total}")

    model.train(training_status)
    print(
        f" [Client {idx}] Mean accuracy up to task {task + 1}:",
        round(correct / total * 100, 2),
        "%",
        "Task accuracies:",
        task_accuracies,
    )
    res = [round(correct / total * 100, 2), task_accuracies]
    return res


def evaluate_client_transfer(fabric, task, model: BaseModel, dataset: BaseDataset, idx: int):
    correct, total = 0, 0
    task_accuracies = []
    training_status = model.training
    model.eval()
    start_class = 0
    if isinstance(dataset.N_CLASSES_PER_TASK, list):
        end_class = sum(dataset.N_CLASSES_PER_TASK[: task + 1])
    else:
        end_class = (task + 1) * dataset.N_CLASSES_PER_TASK
    with torch.no_grad():
        for t in range(task + 1):
            task_correct, task_total = 0, 0
            test_loaders = dataset.get_cur_dataloaders(t)[1]
            for i, test_loader in enumerate(test_loaders):
                if i == idx:
                    continue
                test_loader = fabric.setup_dataloaders(test_loader)
                for inputs, labels in test_loader:
                    #print(f"labels:{labels[:10].cpu().tolist()}")
                    #print(f"task_id: {task}")
                    labels = labels
                    # print(f"labels:{labels[:10].cpu().tolist()}")
                    #inputs = dataset.test_transform(inputs)
                    outputs = model(inputs,t)
                    pred = torch.max(outputs, dim=1)[1]
                    task_correct += (pred == labels).sum().item()
                    task_total += labels.shape[0]
                    correct += (pred == labels).sum().item()
                    total += labels.shape[0]
                task_accuracies.append(round(task_correct / task_total * 100, 2))

    model.train(training_status)
    print(
        f"Mean accuracy up to task {task + 1}:",
        round(correct / total * 100, 2),
        "%",
        "Task accuracies:",
        task_accuracies,
    )
    res = [round(correct / total * 100, 2), task_accuracies]
    return res

def get_unique_labels_cifar100(dataloader):
    """
    Extract unique labels from a CIFAR-100 dataloader.
    Assumes batch = (images, labels)
    """
    labels = set()

    for _, y in dataloader:
        labels.update(y.detach().cpu().tolist())

    return sorted(labels)
def print_unique_labels_cifar100_all_tasks_clients(
    dataset,
    fabric,
    args,
):
    """
    Prints unique CIFAR-100 labels for:
    - every task
    - every client
    - train & test loaders
    """

    assert dataset.N_CLASSES == 100, "This function is intended for CIFAR-100"

    for task in range(dataset.N_TASKS):
        print(f"\n================ TASK {task} =================")

        train_loaders, test_loaders = dataset.get_cur_dataloaders(task)

        task_train_labels = set()
        task_test_labels = set()

        for client_idx in range(args["num_clients"]):
            train_loader = fabric.setup_dataloaders(train_loaders[client_idx])
            test_loader = fabric.setup_dataloaders(test_loaders[client_idx])

            train_labels = get_unique_labels_cifar100(train_loader)
            test_labels = get_unique_labels_cifar100(test_loader)

            task_train_labels.update(train_labels)
            task_test_labels.update(test_labels)

            print(
                f"Client {client_idx:02d} | "
                f"Train labels: {train_labels} | "
                f"Test labels: {test_labels}"
            )

        print("\n--- TASK SUMMARY ---")
        print(f"All train labels in task {task}: {sorted(task_train_labels)}")
        print(f"All test  labels in task {task}: {sorted(task_test_labels)}")


def train(
    fabric,
    server_model: BaseModel,
    client_models: List[BaseModel],
    dataset: BaseDataset,
    args: dict,
    output_folder: str,
) -> None:

    accuracies_each_task = []

    if args["wandb"]:
        name = (
            f"{args['nickname']}_{args['dataset']}_{args['model']}"
            f"_rnds{args['num_comm_rounds']}_clnts{args['num_clients']}"
            f"_epchs{args['num_epochs']}_bs{args['batch_size']}_lr{args['lr']}"
        )
        wandb.init(
            project=args["wandb_project"],
            entity=args["wandb_entity"],
            config=args,
            name=name,
        )

    # ---------------------------------------------------------
    # CHECKPOINT
    # ---------------------------------------------------------
    if args["checkpoint"]:
        start_task, start_comm_round = server_model.load_checkpoint(args["checkpoint"])
        print(f"Loaded checkpoint at {args['checkpoint']}")
    else:
        start_task, start_comm_round = 0, 0

    server_model.train()
    for cm in client_models:
        cm.train()

    if not args["debug_mode"]:
        os.makedirs(output_folder, exist_ok=True)

    start_time = time()

    # =========================================================
    # TASK LOOP
    # =========================================================
    #print_unique_labels_cifar100_all_tasks_clients( dataset=dataset,fabric=fabric,args=args,)
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
    for task in range(dataset.N_TASKS):
        if task < start_task:
            continue

        print(f"\n========== TASK {task} ==========")

        # Load data
        if dataset.IS_TEXT:
            train_loaders, test_loaders = dataset.get_cur_dataloaders_oos(task)
        else:
            train_loaders, test_loaders = dataset.get_cur_dataloaders(task)
            
        # count_unique_samples(train_loader, "Train")
        # count_unique_samples(test_loader, "Test")

        last_task_time = time()

        n_classes = (
            dataset.N_CLASSES_PER_TASK[task]
            if isinstance(dataset.N_CLASSES_PER_TASK, list)
            else dataset.N_CLASSES_PER_TASK
        )

        server_model.begin_task(task_id=task)

        # -----------------------------------------------------
        # CLIENT SAMPLING
        # -----------------------------------------------------
        active_clients = torch.randperm(args["num_clients"])[
            : int(args["num_clients"] * args["participation_rate"])
        ].tolist()

        # -----------------------------------------------------
        # INIT CLIENT TASK
        # -----------------------------------------------------
        for client_idx in active_clients:
            cm = client_models[client_idx]
            cm.augment = dataset.train_transform
            cm.test_transform = dataset.test_transform
            cm.begin_task(task_id=task)

        # =====================================================
        # COMMUNICATION ROUNDS
        # =====================================================
        for comm_round in range(args["num_comm_rounds"]):
            if comm_round < start_comm_round:
                continue

            print(f"\n--- Task {task} , Comm round {comm_round} ---")

            server_model.begin_round_server()
            server_info = server_model.get_server_info()
            clients_info = []

            last_round_time = time()

            # -------------------------------------------------
            # CLIENT LOOP
            # -------------------------------------------------
            for client_idx in active_clients:
                model = client_models[client_idx]
                model.to(model.device)

                train_loader = fabric.setup_dataloaders(train_loaders[client_idx])
                test_loader = fabric.setup_dataloaders(test_loaders[client_idx])

                model.begin_round_client(train_loader, server_info,task)
                model.train()
                #print(f"client {client_idx}:")
                # ---------------- TRAIN ----------------
                for epoch in range(args["num_epochs"]):
                    running_loss = 0.0
                    correct = 0
                    total = 0

                    for i, (inputs, labels) in enumerate(train_loader):
                        labels = labels   # if you're remapping classes

                        batch_loss, preds = model.observe(inputs, labels, task_id=task)

                        # running_loss += batch_loss * inputs.size(0)   # sum of batch losses
                        # correct += (preds == labels).sum().item()    # number of correct predictions
                        # total += labels.size(0)                      # total samples



                        # if i % LOG_LOSS_INTERVAL == 0:
                        #     progress_bar(
                        #         task + 1,
                        #         dataset.N_TASKS,
                        #         comm_round + 1,
                        #         args["num_comm_rounds"],
                        #         client_idx,
                        #         epoch + 1,
                        #         args["num_epochs"],
                        #         batch_loss,
                        #     )

                        if args["wandb"]:
                            wandb.log({"train_loss": batch_loss})
                    # epoch_loss = running_loss / total
                    # epoch_acc = correct / total

                    # print(f"[Client {client_idx}]  Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

                    model.end_epoch()

                model.end_round_client(train_loader,task)

                # ---------------- EVAL ----------------
                
                #acc = evaluate_client(fabric, task, model, dataset, client_idx)
                #print(f"[Client {client_idx}] Local acc: {acc[0]}")

                if args["test_local_transfer"]:
                    acc = evaluate_client_transfer(fabric, task, model, dataset, client_idx)
                    print(f"[Client {client_idx}] Transfer acc: {acc[0]}")

                model.to("cpu")
                clients_info.append(model.get_client_info(train_loader))
                torch.cuda.empty_cache()

            print("Round time:", get_time_str(time() - last_round_time))

            # -------------------------------------------------
            # AGGREGATION
            # -------------------------------------------------
            server_model.end_round_server(clients_info,task)
#             assert all(
#     torch.allclose(a.cpu(), b.cpu(), atol=1e-6)
#     for a, b in zip(
#         client_models[0].network.state_dict().values(),
#         server_model.network.state_dict().values()
#     )
# ), "❌ Global model != client model with 1 client"


            server_model.to(server_model.device)

            accuracy = evaluate(fabric, task, server_model, dataset)
            print(f"[GLOBAL] Accuracy after round {comm_round}: {accuracy[0]}")

            if args["wandb"]:
                log_dict = {"Mean_accuracy": accuracy[0], "task": task + 1}
                for i, acc in enumerate(accuracy[1]):
                    log_dict[f"Task_{i+1}_accuracy"] = acc
                wandb.log(log_dict)

            torch.cuda.empty_cache()

        # =====================================================
        # END TASK
        # =====================================================
        client_info = []
        server_info = server_model.get_server_info()

        for client_idx in active_clients:
            model = client_models[client_idx]
            model.to(model.device)
            train_loader = fabric.setup_dataloaders(train_loaders[client_idx])
            client_info.append(model.end_task_client(train_loader, server_info,task_id=task))
            model.to("cpu")
            torch.cuda.empty_cache()

        server_model.end_task_server(client_info)
        server_model.to(server_model.device)

        accuracy = evaluate(fabric, task, server_model, dataset)
        accuracies_each_task.append(accuracy[1])

        print(f"Task {task + 1} time:", get_time_str(time() - last_task_time))
        print(f"[DEBUG] Accuracy after task {task}: {accuracy[0]}")
        print("================================\n")

    # =========================================================
    # END TRAINING
    # =========================================================
    print("\nTotal training time:", get_time_str(time() - start_time))

    for cm in client_models:
        cm.end_training()
    server_model.end_training()

    forgetting = compute_forgetting(accuracies_each_task)
    paths = accs_and_forgetting_matrix(accuracies_each_task, forgetting, output_folder)

    if args["wandb"]:
        wandb.log(
            {
                "Accuracies Matrix": wandb.Image(paths[0]),
                "Forgetting Vector": wandb.Image(paths[1]),
                "task": dataset.N_TASKS,
            }
        )
        wandb.finish()
