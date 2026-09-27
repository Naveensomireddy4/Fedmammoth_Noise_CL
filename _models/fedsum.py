import copy
import math
import torch
from torch import nn
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel


@register_model("fedsum")
class FedSum(BaseModel):
    def __init__(
        self,
        fabric,
        network: nn.Module,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        wd_reg: float = 0,
        sum_type: str = "treshold",
        snr: None = 0,
    ) -> None:
        self.lr = lr
        self.wd = wd_reg
        self.sum_type = sum_type
        self.snr = snr

        super().__init__(fabric, network, device, optimizer, lr, wd_reg, snr)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,task_id, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            aug_inputs = self.augment(inputs)
            outputs = self.network(aug_inputs,task_id=task_id)
            loss = self.loss(outputs, labels)

        if update:
            self.fabric.backward(loss)
            self.optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            # print(f"True labels:      {labels[:10].cpu().tolist()}")
            # print(f"Predicted labels: {preds[:10].cpu().tolist()}")    

            return loss.item(),preds
        
    def end_round_client(self, dataloader: DataLoader,task):
        pass

    def ota_aggregate_single_noise(self, global_model, client_list, device):
        # Same OTA-channel noise model used by fedavg/er/ewc/lwf/der/derpp:
        # sums client parameter deltas (simulating analog over-the-air
        # superposition), adds Gaussian noise scaled to args.snr (dB), then
        # normalizes by the number of participating clients.
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

            deltas = []
            for c in client_list:
                client_param = c["model"].state_dict()[key].float().to(device)
                deltas.append(client_param - global_param)

            stacked = torch.stack(deltas, dim=0)

            # ---- OTA superposition ----
            superposed = stacked.sum(dim=0)

            # ---- add layer-wise noise ----
            sigma2 = P / snr_linear
            sigma = math.sqrt(sigma2)
            noise = torch.randn_like(superposed) * sigma

            received = (superposed + noise) / K

            new_param = global_param + received
            new_state[key] = new_param.to(global_state[key].dtype)

        global_model.load_state_dict(new_state)

        return global_model, p_t

    def end_round_server(self, client_info: List[dict],task):
        if len(client_info) == 0:
            return

        if self.sum_type == "treshold":
            total_samples = sum(client["num_train_samples"] for client in client_info)
            norm_weights = [
                client["num_train_samples"] / total_samples
                for client in client_info
            ]

            # threshold-based client selection (unchanged): drop clients
            # whose contribution would be negligible
            selected_state_dicts = [
                client["state_dict"]
                for client, norm_weight in zip(client_info, norm_weights)
                if norm_weight > 1 / (len(client_info) * 10)
            ]

            if len(selected_state_dicts) == 0:
                return

        else:
            # no thresholding: every client participates
            selected_state_dicts = [client["state_dict"] for client in client_info]

        # ---- OTA-noisy aggregation over the selected clients ----
        client_list = []
        for state_dict in selected_state_dicts:
            temp_model = copy.deepcopy(self.network)
            temp_model.load_state_dict(state_dict)
            client_list.append({"model": temp_model})

        self.network, _ = self.ota_aggregate_single_noise(
            global_model=self.network,
            client_list=client_list,
            device=self.device,
        )

    def begin_round_client(self, dataloader: DataLoader, server_info: dict,task):
        self.network.load_state_dict(server_info["state_dict"], strict=True)

    def get_client_info(self, dataloader: DataLoader):
        return {
            "state_dict": self.network.state_dict(),
            "num_train_samples": len(dataloader.dataset),
        }

    def get_server_info(self):
        return {"state_dict": self.network.state_dict()}
    
