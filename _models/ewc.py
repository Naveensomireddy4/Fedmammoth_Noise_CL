from copy import deepcopy
from numpy import copy
import torch
import math
from torch import nn
from torch.nn import functional as F
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
from utils.tools import str_to_bool
from _networks.vit_prompt_hgp import VitHGP
from torch.optim import SGD
from _networks.vit import VisionTransformer
import copy

@register_model("ewc")
class EWC(BaseModel):

    def __init__(
        self,
        fabric,
        network: nn.Module,
        device: str,
        num_comm_rounds: int,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        linear_probe: str_to_bool = False,
        snr:None = 0,
        slca: str_to_bool = False,
        ewc_lambda: float = 100,
        ewc_gamma: float = 0.95,
    ) -> None:
        if type(network) == VitHGP:
            for n, p in network.named_parameters():
                if "prompt" or "last" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            params = [{"params": network.last.parameters()}, {"params": network.prompt.parameters()}]
            super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        elif type(network) == VisionTransformer:
            params_backbone = []
            params_head = []
            for n, p in network.named_parameters():
                p.requires_grad = True
                if "head" not in n:
                    params_backbone.append(p)
                else:
                    params_head.append(p)
            params = [{"params": params_head}, {"params": params_backbone}]
            if slca:
                params = [{"params": params_backbone, "lr": lr / 100}, {"params": params_head}]
            super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params)
        else:
            super().__init__(fabric, network, device, optimizer, lr, wd_reg,snr)
        self.avg_type = avg_type
        self.do_linear_probe = linear_probe
        self.done_linear_probe = False
        self.slca = slca
        self.snr=snr
        self.lr = lr
        self.wd = wd_reg
        self.num_comm_rounds = num_comm_rounds
        self.ewc_lambda = ewc_lambda
        self.ewc_gamma = ewc_gamma
        print(f"ewc_lambda = {self.ewc_lambda} ,ewc_gamma = { self.ewc_gamma} ")
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.optimizer_str = optimizer
        self.prev_params = None
        self.fish = None
        self.minibatch_size = 16
        self.round = 0
        self.do_linear_probe = linear_probe
        self.done_linear_probe = False


    # --------------------------------------------------
    # TASK-IL TRAINING
    # --------------------------------------------------
    def observe(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
        update: bool = True,
    ):
        if self.round == self.num_comm_rounds:
            return 0.0, torch.zeros_like(labels)
  
        # inputs = self.augment(inputs)
        with self.fabric.autocast():
            outputs = self.network(inputs, task_id=task_id)
            loss = self.loss(outputs, labels)

            # 🔒 EWC penalty (backbone only)
            #so the diffetrefnce is in previous we are adding them to  gradiant and now we are adding them to loss

            if self.prev_params is not None:
                penalty = 0.0
                for n, p in self.network.named_parameters():
                    if n in self.fish:
                        penalty += (self.fish[n] * (p - self.prev_params[n]) ** 2).sum()
                loss += self.ewc_lambda * penalty

        if update:
            self.fabric.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            # print(f"True labels:      {labels[:10].cpu().tolist()}")
            # print(f"Predicted labels: {preds[:10].cpu().tolist()}")    

            return loss.item(),preds
    
    def begin_task(self ,task_id=0):
        self.current_task = task_id
        if self.do_linear_probe:
            self.done_linear_probe = False
        self.round =0    
        
    def begin_round_server(self):
        self.round += 1    

    # --------------------------------------------------
    # CLIENT: END TASK → COMPUTE FISHER
    # --------------------------------------------------
    from torch.optim import SGD
    import torch.nn.functional as F

    def end_task_client(self, train_loader, server_info,task_id=0):
        fish = {n: torch.zeros_like(p)
                for n, p in self.network.module.named_parameters()
                if p.requires_grad}

        fake_opt = SGD(self.network.module.parameters(), lr=0)
        for j, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device).long()
            for ex, lab in zip(inputs, labels):
                output = self.network.module(ex.unsqueeze(0), task_id=task_id)
                loss = -F.nll_loss(self.logsoft(output), lab.unsqueeze(0), reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                for n, p in self.network.module.named_parameters():
                    if p.requires_grad and p.grad is not None and n in fish:
                        fish[n] += exp_cond_prob * p.grad ** 2
                fake_opt.zero_grad()
        fake_opt = None

        self.fish = {n: v.cpu() for n, v in fish.items()}

    
        return self.get_client_info(train_loader)
    
    
    
    def end_round_client(self, dataloader: DataLoader,task=0):
        super().end_round_client(dataloader)
        self.optimizer = None
        self.prev_params = None

    # --------------------------------------------------
    # SERVER: END TASK → MERGE FISHER + CHECKPOINT
    # --------------------------------------------------
    def end_task_server(self, client_info: List[dict]):
        """
        Server-side step at the end of a task.
        Updates Fisher information for EWC and stores parameter checkpoint.
        """
        # Always save checkpoint of current model

        self.prev_params = {
            k: v.detach().clone()
            for k, v in self.network.state_dict().items()
        }

        # Total samples across clients
        total_samples = sum(c["num_train_samples"] for c in client_info)

        # Only update Fisher info at the last communication round
        if self.round == self.num_comm_rounds:
            # Compute weighted average of client Fishers
            print("Updating Fisher matrix")
            new_fisher = {}
            for key in client_info[0]["fisher"].keys():
                new_fisher[key] = sum(
                    c["fisher"][key] * (c["num_train_samples"] / total_samples)
                    for c in client_info
                )

            # If no previous Fisher, initialize; else decay + add new
            if self.fish is None:
                self.fish = new_fisher
            else:
                for k in self.fish:
                    self.fish[k] = self.ewc_gamma * self.fish[k] + new_fisher[k].to(self.fish[k].device)

        # Clear total_samples if needed
        self.total_samples = None

    # --------------------------------------------------
    # SERVER: FEDAVG (ALL PARAMETERS)
    # --------------------------------------------------
    def ota_aggregate_single_noise(self, global_model, client_list, device):

        K = len(client_list)
        if K == 0:
            return global_model, 1.0

        p_t = 1.0
        snr_db = self.snr
        print(f"snr from args : {snr_db}")
        P = 1
        snr_linear = 10 ** (snr_db / 10)

        global_state = global_model.state_dict()
        new_state = {}

        for key in global_state.keys():

            global_param = global_state[key].float().to(device)

            # ---- collect client deltas ----
            deltas = []
            for c in client_list:
                client_param = c['model'].state_dict()[key].float().to(device)
                deltas.append(client_param - global_param)

            stacked = torch.stack(deltas, dim=0)

            # ---- OTA superposition ----
            superposed = stacked.sum(dim=0)

            # ---- add layer-wise noise ----
            sigma2 = P / snr_linear
            sigma = math.sqrt(sigma2)
            noise = torch.randn_like(superposed) * sigma

            received = (superposed + noise) / K

            # ---- update global ----
            new_param = global_param + received
            new_state[key] = new_param.to(global_state[key].dtype)

        global_model.load_state_dict(new_state)

        return global_model, p_t
    
    def end_round_server(self, client_info,task):

        if len(client_info) == 0:
            return

        # -----------------------------------
        # Step 1: Convert client_info → OTA format
        # -----------------------------------
        total_samples = sum(c["num_train_samples"] for c in client_info)
        client_list = []

        for c in client_info:
            temp_model = copy.deepcopy(self.network)
            temp_model.load_state_dict(c["state_dict"])

            client_list.append({
                "model": temp_model
            })
            
                # -----------------------------------
        # Step 2: Call OTA aggregation
        # -----------------------------------
        self.network, _ = self.ota_aggregate_single_noise(
            global_model=self.network,
            client_list=client_list,           # or your dimension variable
            device=self.device
        )     

    # --------------------------------------------------
    # CLIENT ROUND START
    # --------------------------------------------------
    def linear_probe(self, dataloader: DataLoader,task_id=0):
        for epoch in range(5):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    # inputs = self.augment(inputs)
                    pre_logits = self.network(inputs, pen=True, train=False)
                outputs = self.network.last(pre_logits,task_id=task_id)
                labels = labels 
                loss = F.cross_entropy(outputs, labels)
                self.optimizer.zero_grad()
                self.fabric.backward(loss)
                self.optimizer.step()
    def begin_round_client(self, dataloader, server_info,task=0):
        self.round+=1

        self.network.load_state_dict(server_info["state_dict"], strict=True)

        if server_info["prev_params"] is not None:
            self.prev_params = {
                k: v.to(self.device)
                for k, v in server_info["prev_params"].items()
            }

        if server_info["fisher"] is not None:
            self.fish = {
                k: v.to(self.device)
                for k, v in server_info["fisher"].items()
            }
        if self.do_linear_probe and not self.done_linear_probe:
            optimizer = self.optimizer_class(self.network.last.parameters(), lr=self.lr, weight_decay=self.wd)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
            self.linear_probe(dataloader)
            self.done_linear_probe = True
            params = [{"params": self.network.parameters()}, {"params": self.network.prompt.parameters()}]
            optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd)
            self.optimizer = self.fabric.setup_optimizers(optimizer)    
        # here we are missing linear probe   
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        self.optimizer_class = OptimizerClass
        self.optimizer = OptimizerClass(self.network.parameters(), lr=self.lr, weight_decay=self.wd)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer) 

    # --------------------------------------------------
    # COMMUNICATION
    # --------------------------------------------------
    def get_client_info(self, dataloader):

        return {
            "state_dict": {
                k: v.detach().cpu()
                for k, v in self.network.state_dict().items()
            },
            "num_train_samples": len(dataloader.dataset),
            "fisher": {
                k: v.detach().cpu()
                for k, v in self.fish.items()
            } if self.fish is not None else None,
        }

    def get_server_info(self):
        return {
            "state_dict": {
                k: v.detach().cpu()
                for k, v in self.network.state_dict().items()
            },
            "prev_params": (
                {k: v.detach().cpu() for k, v in self.prev_params.items()}
                if self.prev_params is not None
                else None
            ),
            "fisher": (
                {k: v.detach().cpu() for k, v in self.fish.items()}
                if self.fish is not None
                else None
            ),
        }

    def to(self, device):
        self.network.to(device)

        if self.prev_params is not None:
            self.prev_params = {
                k: v.to(device)
                for k, v in self.prev_params.items()
            }

        if self.fish is not None:
            self.fish = {
                k: v.to(device)
                for k, v in self.fish.items()
            }

        return self

