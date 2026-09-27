from sympy import python
import torch
import copy
import math
from torch import nn
from torch.nn import functional as F
from typing import List
from torch.utils.data import DataLoader
from collections import defaultdict
import copy
import torch
from _models import register_model
from _models._utils import BaseModel
from utils.tools import str_to_bool
from _networks.vit_prompt_hgp import VitHGP
from _networks.vit import VisionTransformer
from ._utils_memory import Memory
@register_model("bi")
class BI(BaseModel):
    """
    FedAvg adapted for Task-Incremental Learning (Task-IL)
    Assumes:
      - network(x, task_id=...) selects correct head
      - all heads exist from initialization
    """

    def __init__(
        self,
        fabric,
        network: nn.Module,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        linear_probe: str_to_bool = False,       
        snr:None = 0,
        slca: str_to_bool = False,
        args = None
    ) -> None:

        self.lr = lr
        self.wd = wd_reg
        self.avg_type = avg_type
        self.snr=snr
        # self.do_linear_probe = linear_probe
        self.do_linear_probe = None
        self.done_linear_probe = False
        self.slca = slca
        self.args=args
        self.memory =Memory(args)

        # -------- Parameter selection logic (unchanged) --------
        if isinstance(network, VitHGP):
            for n, p in network.named_parameters():
                if "prompt" in n or "last" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            params = [
                {"params": network.last.parameters()},
                {"params": network.prompt.parameters()},
            ]

            super().__init__(
                fabric, network, device, optimizer, lr, wd_reg, params=params
            )

        elif isinstance(network, VisionTransformer):
            params_backbone = []
            params_head = []

            for n, p in network.named_parameters():
                p.requires_grad = True
                if "head" in n or "classifiers" in n:
                    params_head.append(p)
                else:
                    params_backbone.append(p)

            if slca:
                params = [
                    {"params": params_backbone, "lr": lr / 100},
                    {"params": params_head},
                ]
            else:
                params = [
                    {"params": params_head},
                    {"params": params_backbone},
                ]

            super().__init__(
                fabric, network, device, optimizer, lr, wd_reg, params=params
            )

        else:
            super().__init__(fabric, network, device, optimizer, lr, wd_reg,snr)

    # ------------------------------------------------------------------
    # TASK-IL OBSERVE
    # # ------------------------------------------------------------------
    

    def observe(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
        update: bool = True,
    ) -> float:

        self.optimizer.zero_grad()
        num_classes_per_task = 20
        self.task_id = task_id

        # local labels for training
        local_labels = labels

        # global labels for memory
        memory_labels = labels + (task_id * num_classes_per_task)

        # task-specific global class range
        current_classes = list(
            range(
                task_id * num_classes_per_task,
                (task_id + 1) * num_classes_per_task
            )
        )

        # ---------------- TASK 0 ----------------
        if task_id == 0:
            samples = inputs

            with self.fabric.autocast():
                outputs = self.network(inputs, task_id=task_id)
                batch_loss = self.loss(outputs, local_labels)

            if update:
                self.fabric.backward(batch_loss)
                self.optimizer.step()

            # memory update uses GLOBAL labels
            if self.args.update_strategy == 'reservoir':
                self.memory.reservoir_update(
                    samples,
                    memory_labels,
                    self.task_id
                )

            if self.args.update_strategy == 'balanced':
                self.memory.class_balanced_update(
                    samples,
                    memory_labels,
                    self.task_id,
                    self.network,
                    current_classes
                )

            if self.args.update_strategy == 'uncertainty':
                self.memory.uncertainty_update(
                    samples,
                    memory_labels,
                    self.task_id,
                    self.network
                )

        # ---------------- TASK > 0 ----------------
        else:
            samples = inputs

            # replay sampling
            if self.args.sampling_strategy == 'uncertainty':
                mem_x, mem_y, mem_t = self.memory.uncertainty_sampling(
                    self.last_local_model,
                    exclude_task=self.task_id,
                    subsample_size=self.args.subsample_size
                )

            if self.args.sampling_strategy == 'random':
                mem_x, mem_y, mem_t = self.memory.random_sampling(
                    self.args.batch_size,
                    exclude_task=self.task_id
                )

            if self.args.sampling_strategy == 'balanced_random':
                mem_x, mem_y, mem_t = self.memory.balanced_random_sampling(
                    self.args.batch_size,
                    exclude_task=self.task_id
                )

            # fallback if replay empty
            if mem_x is None:
                mem_x = torch.empty(
                    0, *samples.shape[1:], device=self.args.device
                )
                mem_y = torch.empty(
                    0, dtype=torch.long, device=self.args.device
                )
                mem_t = torch.empty(
                    0, dtype=torch.long, device=self.args.device
                )
            else:
                mem_x = mem_x.to(self.args.device)
                mem_y = mem_y.to(self.args.device)
                mem_t = mem_t.to(self.args.device)

            # convert global replay labels back to local labels
            mem_y_local = mem_y % num_classes_per_task

            # current task ids
            cur_t = torch.full(
                (samples.size(0),),
                self.task_id,
                device=self.args.device
            )

            # combine current + replay
            input_x = torch.cat([samples, mem_x])
            input_y = torch.cat([local_labels, mem_y_local])
            input_t = torch.cat([cur_t, mem_t])

            outputs = torch.zeros(
                input_x.size(0),
                self.args.num_classes_per_task,
                device=self.args.device
            )

            with self.fabric.autocast():
                for t in input_t.unique():
                    idx = (input_t == t)

                    outputs[idx] = self.network(
                        input_x[idx],
                        task_id=int(t.item())
                    )

                batch_loss = self.loss(outputs, input_y)

            if update:
                self.fabric.backward(batch_loss)
                self.optimizer.step()

            # memory update uses GLOBAL labels
            if self.args.update_strategy == 'reservoir':
                self.memory.reservoir_update(
                    samples,
                    memory_labels,
                    self.task_id
                )

            if self.args.update_strategy == 'balanced':
                self.memory.class_balanced_update(
                    samples,
                    memory_labels,
                    self.task_id,
                    self.last_local_model,
                    current_classes
                )

            if self.args.update_strategy == 'uncertainty':
                self.memory.uncertainty_update(
                    samples,
                    memory_labels,
                    self.task_id,
                    self.last_local_model
                )

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            return batch_loss.item(), preds

        
    # def observe(
    #     self,
    #     inputs: torch.Tensor,
    #     labels: torch.Tensor,
    #     task_id: int,
    #     update: bool = True,
    # ) -> float:
        
    #     self.optimizer.zero_grad()
    #     num_classes_per_task = 20
    #     self.task_id= task_id
        
    #     if task_id ==0 :
    #         samples, labels = inputs, labels
    #         current_classes = list(range(num_classes_per_task))# no of classes per task
    #         with self.fabric.autocast():
    #            outputs = self.network(inputs, task_id=task_id)
    #            batch_loss = self.loss(outputs, labels)
    #         if update:
    #             self.fabric.backward(batch_loss)
    #             self.optimizer.step()   

    #         if self.args.update_strategy == 'reservoir':
    #             self.memory.reservoir_update(samples, labels, self.task_id)
    #         if self.args.update_strategy == 'balanced':
    #             self.memory.class_balanced_update(samples, labels, self.task_id, self.network, current_classes)
    #         if self.args.update_strategy == 'uncertainty':
    #             self.memory.uncertainty_update(samples, labels, self.task_id, self.network)
                
    #     else:
    #         samples, labels = inputs, labels
    #         current_classes = list(range(num_classes_per_task)) # no of classes per task

    #         if self.args.sampling_strategy == 'uncertainty':
    #             mem_x, mem_y, mem_t = self.memory.uncertainty_sampling(self.last_local_model, exclude_task=self.task_id,
    #                                                                 subsample_size=self.args.subsample_size)
    #         if self.args.sampling_strategy == 'random':
    #             mem_x, mem_y, mem_t = self.memory.random_sampling(self.args.batch_size, exclude_task=self.task_id)
    #         if self.args.sampling_strategy == 'balanced_random':
    #             mem_x, mem_y, mem_t = self.memory.balanced_random_sampling(self.args.batch_size, exclude_task=self.task_id)

    #         mem_x, mem_y, mem_t = (
    #             mem_x.to(self.args.device),
    #             mem_y.to(self.args.device),
    #             mem_t.to(self.args.device)
    #         )

    #         # current batch belongs to current task
    #         cur_t = torch.full(
    #             (samples.size(0),),
    #             self.task_id,
    #             device=self.args.device
    #         )

    #         input_x = torch.cat([samples, mem_x])
    #         input_y = torch.cat([labels, mem_y])
    #         input_t = torch.cat([cur_t, mem_t])

    #         outputs = torch.zeros(
    #             input_x.size(0),
    #             self.args.num_classes_per_task,
    #             device=self.args.device
    #         )

    #         with self.fabric.autocast():
    #             for t in input_t.unique():
    #                 idx = (input_t == t)
    #                 outputs[idx] = self.network(
    #                     input_x[idx],
    #                     task_id=int(t.item())
    #                 )

    #             batch_loss = self.loss(outputs, input_y)
    #         if update:
    #             self.fabric.backward(batch_loss)
    #             self.optimizer.step() 

    #         if self.args.update_strategy == 'reservoir':
    #             self.memory.reservoir_update(samples, labels, self.task_id)
    #         if self.args.update_strategy == 'balanced':
    #             self.memory.class_balanced_update(samples, labels, self.task_id, self.last_local_model, current_classes)

    #     with torch.no_grad():
    #         preds = torch.argmax(outputs, dim=1)
    #         # print(f"True labels:      {labels[:10].cpu().tolist()}")
    #         # print(f"Predicted labels: {preds[:10].cpu().tolist()}")    

    #         return batch_loss.item(),preds
            

    # ------------------------------------------------------------------
    # TASK MANAGEMENT
    # ------------------------------------------------------------------
    def begin_task(self ,task_id):
        self.current_task = task_id
        if self.do_linear_probe:
            self.done_linear_probe = False

    def linear_probe(self, dataloader: DataLoader,task_id):
        print("*******************In linear_probe **************** ")
        for epoch in range(5):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    # inputs = self.augment(inputs)
                    pre_logits = self.network(inputs, pen=True, train=False)
                outputs = self.network.last(pre_logits,task_id)
                labels = labels 
                loss = F.cross_entropy(outputs, labels)
                self.optimizer.zero_grad()
                self.fabric.backward(loss)
                self.optimizer.step()
                


    # def end_round_server(self, client_info,task=0):
    #     if len(client_info) == 0:
    #         return

    #     total_samples = sum(c["num_train_samples"] for c in client_info)

    #     avg_state = {}

    #     for key in client_info[0]["state_dict"].keys():
    #         avg_state[key] = sum(
    #             (c["state_dict"][key] * (c["num_train_samples"] / total_samples))
    #             for c in client_info
    #         )

    #     self.network.load_state_dict(avg_state, strict=True)
    
    def begin_round_server(self, client_info=None):
        self.last_global_model = copy.deepcopy(self.network)


    def end_round_server(self, client_info, task=0):
        if len(client_info) == 0:
            return

        total_samples = sum(c["num_train_samples"] for c in client_info)

        avg_state = {}
        for key in client_info[0]["state_dict"].keys():
            avg_state[key] = sum(
                c["state_dict"][key] * (c["num_train_samples"] / total_samples)
                for c in client_info
            )

        # Previous global model
        old_global = {
            k: v.detach().cpu()
            for k, v in self.last_global_model.state_dict().items()
        }

        # Average previous global and new global
        for key in avg_state:
            avg_state[key] = (old_global[key] + avg_state[key]) / 2.0

        self.network.load_state_dict(avg_state, strict=True)
        
    def end_round_client(self, task_loader,task=0):
        self.last_local_model = copy.deepcopy(self.network)


    def begin_round_client(self, dataloader: DataLoader, server_info: dict,task=0):
        #self.network.set_params(server_info["params"])
        
        self.network.load_state_dict(server_info["state_dict"], strict=True)
        if self.do_linear_probe and not self.done_linear_probe:
            optimizer = self.optimizer_class(self.network.last.parameters(), lr=self.lr, weight_decay=self.wd)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
            self.linear_probe(dataloader,task_id = task)
            self.done_linear_probe = True
        # restore correct optimizer
        params = [{"params": self.network.parameters()}]
        optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd)
        self.optimizer = self.fabric.setup_optimizers(optimizer)

    # def get_client_info(self, dataloader: DataLoader):
    #     return {
    #         "params": self.network.get_params(),
    #         "num_train_samples": len(dataloader.dataset.data),
    #     }
    def get_client_info(self, dataloader):
        return {
            "state_dict": {k: v.detach().cpu() for k, v in self.network.state_dict().items()},
            "num_train_samples": len(dataloader.dataset),
        }


    # def get_server_info(self):
    #     return {"params": self.network.get_params()}
    def get_server_info(self):
        return {"state_dict": {k: v.detach().cpu() for k, v in self.network.state_dict().items()}}
    
    
