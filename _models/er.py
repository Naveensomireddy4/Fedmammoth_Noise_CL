import torch
import copy
import math
from torch import nn
from torch.nn import functional as F
from typing import List
from torch.utils.data import DataLoader
from _models._utils import Buffer
from _models import register_model
from _models._utils import BaseModel
from utils.tools import str_to_bool
from _networks.vit_prompt_hgp import VitHGP
from _networks.vit import VisionTransformer


@register_model("er")
class ER(BaseModel):
    """
    ER adapted for Task-Incremental Learning (Task-IL)
    Assumes:
      - network(x, task_id=...) selects correct head
      - all heads exist from initialization
    """

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
    ) -> None:

        self.lr = lr
        self.wd = wd_reg
        self.avg_type = avg_type
        self.snr=snr
        # self.do_linear_probe = linear_probe
        self.do_linear_probe = None
        self.done_linear_probe = False
        self.slca = slca
        self.batch_size = batch_size
        

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
            
        self.buffer = Buffer(1000, self.device)    

    # ------------------------------------------------------------------
    # TASK-IL OBSERVE
    # ------------------------------------------------------------------
    def observe(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
        update: bool = True,
    ) -> float:

        self.optimizer.zero_grad()

        with self.fabric.autocast():
            # Current task loss
            outputs = self.network(inputs, task_id=task_id)
            loss = self.loss(outputs, labels)

            # Experience Replay
            if task_id > 0 and len(self.buffer) > 0:

                buf_inputs, buf_labels, _, buf_task_ids = \
                    self.buffer.get_data(self.batch_size)

                replay_loss = 0.0

                for t in buf_task_ids.unique():
                    idx = (buf_task_ids == t)

                    buf_outputs = self.network(
                        buf_inputs[idx],
                        task_id=int(t.item())
                    )

                    replay_loss += self.loss(
                        buf_outputs,
                        buf_labels[idx]
                    )

                loss +=replay_loss

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
            logits=outputs.data,
            task_ids=task_ids
        )    
            # ---------- DEBUG: print predictions ----------
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            # print(f"True labels:      {labels[:10].cpu().tolist()}")
            # print(f"Predicted labels: {preds[:10].cpu().tolist()}")    

            return loss.item(),preds

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

    # def end_round_server(self, client_info: List[dict]):
    #     if self.avg_type == "weighted":
    #         total_samples = sum([client["num_train_samples"] for client in client_info])
    #         norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
    #     else:
    #         weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
    #         norm_weights = [w / sum(weights) for w in weights]
    #     if len(client_info) > 0:
    #         self.network.set_params(
    #             torch.stack(
    #                 [client["params"] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]
    #             ).sum(0)
    #    
    # )
    def end_round_client(self, task_loader,task=0):
        pass
        
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

    def end_round_server(self, client_info,task=0):

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
    
    
