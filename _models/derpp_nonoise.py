from copy import deepcopy
import copy
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
from _models._utils import Buffer
import math
import copy

@register_model("derpp_nonoise")
class Derpp(BaseModel):

    def __init__(
        self,
        fabric,
        network: nn.Module,
        device: str,
        batch_size: int,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        linear_probe: str_to_bool = False,
        snr:None = 0,
        slca: str_to_bool = False,
        buffer_size: int = 1000,
        alpha: float = 0.2,
        beta: float = 0.5
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
            super().__init__(fabric, network, device, optimizer, lr, wd_reg, snr=snr)

        self.avg_type = avg_type
        self.do_linear_probe = linear_probe
        self.done_linear_probe = False
        self.slca = slca
        self.snr = snr
        self.lr = lr
        self.wd = wd_reg
        self.optimizer_str = optimizer
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.transform = None
        self.buffer = Buffer(buffer_size, self.device)
        print(f"snr value is {self.snr}  running wihout noise ***********")

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, task_id, update: bool = True) -> float:
        self.optimizer.zero_grad()
        #aug_inputs = self.augment(inputs)
        with self.fabric.autocast():
            logits = self.network(inputs, task_id)
            loss = self.loss(logits, labels)

            if len(self.buffer) > 0:
                loss_re = 0
                if self.alpha > 0:
                    buf_inputs, _, buf_logits, buf_task_ids = self.buffer.get_data(self.batch_size)
                    buf_outputs = torch.zeros_like(buf_logits)

                    for t in buf_task_ids.unique():
                        idx = (buf_task_ids == t)
                        buf_outputs[idx] = self.network(
                           buf_inputs[idx],
                            int(t.item())
                        )

                    loss_re += self.alpha * F.mse_loss(buf_outputs, buf_logits)

                if self.beta > 0:
                    buf_inputs, buf_labels, _, buf_task_ids = self.buffer.get_data(self.batch_size)
                    buf_outputs = torch.zeros_like(buf_logits)

                    for t in buf_task_ids.unique():
                        idx = (buf_task_ids == t)
                        buf_outputs[idx] = self.network(
                            buf_inputs[idx],
                            int(t.item())
                        )

                    loss_re += self.beta * self.loss(buf_outputs, buf_labels)

                loss += loss_re

        if update:
            self.fabric.backward(loss)
            self.optimizer.step()

        task_ids = torch.full(
            (inputs.size(0),),
            task_id,
            dtype=torch.long,
            device=inputs.device
        )

        self.buffer.add_data(
            examples=inputs.data,
            labels=labels.data,
            logits=logits.data,
            task_ids=task_ids
        )

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            return loss.item(), preds

    def begin_task(self, task_id: int=0):
        self.cur_task = task_id
        if self.do_linear_probe:
            self.done_linear_probe = False

    def end_task_server(self, client_info: List[dict] = None):
        super().end_task_server(client_info)
        train_status = self.network.training
        self.checkpoint = deepcopy(self.network.eval())
        self.network.train(train_status)


    def end_round_server(self, client_info: List[dict],task=0):
        if len(client_info) == 0:
            return

        total_samples = sum(c["num_train_samples"] for c in client_info)

        avg_state = {}

        for key in client_info[0]["state_dict"].keys():
            avg_state[key] = sum(
                (c["state_dict"][key] * (c["num_train_samples"] / total_samples))
                for c in client_info
            )

        self.network.load_state_dict(avg_state, strict=True)

    def begin_round_client(self, dataloader: DataLoader, server_info: dict,task_id=0):
        self.network.load_state_dict(server_info["state_dict"], strict=True)
        if self.do_linear_probe and not self.done_linear_probe:
            optimizer = self.optimizer_class(self.network.parameters(), lr=self.lr, weight_decay=self.wd)
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
            "state_dict": self.network.state_dict(),
            "num_train_samples": len(dataloader.dataset),
        }

    def end_round_client(self, dataloader: DataLoader,task=0):
        super().end_round_client(dataloader)
        self.optimizer = None

    def get_server_info(self):
        return {"state_dict": self.network.state_dict()}
    

    def to(self, device):
        self.network.to(device)
        return self
    
