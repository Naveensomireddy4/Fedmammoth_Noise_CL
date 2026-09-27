from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
from utils.tools import str_to_bool
from _networks.vit_prompt_hgp import VitHGP
from _networks.vit import VisionTransformer
import math
import copy
@register_model("lwf")
class LwF(BaseModel):

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
        alpha: float = 0.5,
        softmax_temp: float = 2
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
        self.lr = lr
        self.snr=snr
        self.wd = wd_reg
        self.alpha = alpha
        self.softmax_temp = softmax_temp
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.optimizer_str = optimizer
    
    @staticmethod
    def modified_kl_div(old, new):
        return -torch.mean(torch.sum(old * torch.log(new), 1))

    @staticmethod
    def smooth(logits, temp, dim):
        log = logits ** (1 / temp)
        return log / torch.sum(log, dim).unsqueeze(1)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,task_id, update: bool = True) -> float:

        # aug_inputs = self.augment(inputs)
        with self.fabric.autocast():
            outputs = self.network(inputs,task_id)
            loss = self.loss(outputs, labels)
            if self.cur_task > 0 and self.checkpoint is not None:
                with torch.no_grad():
                    old_logits = self.checkpoint(inputs,task_id)
                loss += self.alpha * self.modified_kl_div(self.smooth(self.soft(old_logits).to(self.device), 2, 1),
                                                        self.smooth(self.soft(outputs), 2, 1))
        if update:
            self.fabric.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            # print(f"True labels:      {labels[:10].cpu().tolist()}")
            # print(f"Predicted labels: {preds[:10].cpu().tolist()}")    

            return loss.item(),preds

    def begin_task(self, task_id: int):
        self.cur_task  = task_id
        if self.do_linear_probe:
            self.done_linear_probe = False

    def linear_probe(self, dataloader: DataLoader,task_id):
        for epoch in range(5):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    # inputs = self.augment(inputs)
                    pre_logits = self.network(inputs, penultimate=True, train=False)
                outputs = self.network.last(pre_logits,task_id)

                loss = F.cross_entropy(outputs, labels)
                self.optimizer.zero_grad()
                self.fabric.backward(loss)
                self.optimizer.step()

    def ota_aggregate_single_noise(self, global_model, client_list, device):

        K = len(client_list)
        if K == 0:
            return global_model, 1.0

        p_t = 50
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
            
    def end_task_server(self, client_info: List[dict] = None,task_id=0):
        super().end_task_server(client_info)
        train_status = self.network.training
        self.checkpoint = deepcopy(self.network.eval())
        self.network.train(train_status)

    def begin_round_client(self, dataloader: DataLoader, server_info: dict,task_id):
        self.network.load_state_dict(server_info["state_dict"], strict=True)
        if server_info["checkpoint"] is not None:
            train_status = self.network.training
            self.checkpoint = deepcopy(self.network.eval())
            self.network.train(train_status)
            self.checkpoint.load_state_dict(server_info["checkpoint"])
        if self.do_linear_probe and not self.done_linear_probe:
            optimizer = self.optimizer_class(self.network.last.parameters(), lr=self.lr, weight_decay=self.wd)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
            self.linear_probe(dataloader,task_id)
            self.done_linear_probe = True
            # restore correct optimizer
            params = [{"params": self.network.last.parameters()}, {"params": self.network.prompt.parameters()}]
            optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
        
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        self.optimizer_class = OptimizerClass
        self.optimizer = OptimizerClass(self.network.parameters(), lr=self.lr, weight_decay=self.wd)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)

    def get_client_info(self, dataloader: DataLoader):
        return {
            "state_dict":  self.network.state_dict(),
            "num_train_samples": len(dataloader.dataset.data),
        }

    def end_round_client(self, dataloader: DataLoader,task=0):
        super().end_round_client(dataloader)
        self.optimizer = None
        self.checkpoint = None

    def get_server_info(self):
        dct = {
            "state_dict": self.network.state_dict()
        }
        # Include checkpoint if it exists
        if hasattr(self, "checkpoint") and self.checkpoint is not None:
            dct["checkpoint"] = self.checkpoint.state_dict()
        else:
            dct["checkpoint"] = None
        return dct

    def to(self, device):
        self.network.to(device)
        if self.checkpoint is not None:
            self.checkpoint.to(device)
        return self
