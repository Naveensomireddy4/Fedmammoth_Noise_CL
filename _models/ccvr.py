import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from _models import register_model
from typing import List
from torch.utils.data import DataLoader
from _models._utils import BaseModel
from _networks.vit import VisionTransformer as Vit
import os
from utils.tools import str_to_bool
from torch.distributions import MultivariateNormal
import torch
import torch.nn.functional as F

@register_model("ccvr")
class CCVR(BaseModel):
    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 1e-3,
        wd_reg: float = 0,
        avg_type: str = "weighted",
        how_many: int = 1000,
        full_cov: str_to_bool = False,
        linear_probe: str_to_bool = False,
        snr:None = 0,
    ) -> None:
        params = [{"params": network.parameters()}]
        super().__init__(fabric, network, device, optimizer, lr, wd_reg, params=params,snr=snr)
        self.avg_type = avg_type
        self.snr=snr
        self.how_many = how_many
        self.clients_statistics = None
        self.mogs = {}
        self.logit_norm = 0.1
        self.full_cov = full_cov
        self.do_linear_probe = linear_probe
        self.done_linear_probe = False
        self.lr = lr
        self.wd_reg = wd_reg
        self.cpt = []

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,task_id, update: bool = True) -> float:
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            outputs = self.network(inputs,task_id=task_id)
            loss = self.loss(outputs, labels )

        if update:
            self.fabric.backward(loss)
            # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            # print(f"True labels:      {labels[:10].cpu().tolist()}")
            # print(f"Predicted labels: {preds[:10].cpu().tolist()}")    

            return loss.item(),preds

    def linear_probe(self, dataloader: DataLoader,task_id):
        for epoch in range(5):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    inputs = self.augment(inputs)
                    pre_logits = self.network(inputs, pen=True, train=False)
                outputs = self.network.model.head(pre_logits,task_id)
                loss = F.cross_entropy(outputs, labels)
                self.optimizer.zero_grad()
                self.fabric.backward(loss)
                self.optimizer.step()

    def begin_task(self, task_id: int=0):
        #super().begin_task(n_classes_per_task)
        self.cur_task =task_id
        if self.do_linear_probe:
            self.done_linear_probe = False
    def end_round_server(self, client_info: list, task_id=0):
        """
        CCVR server update with OTA channel noise
        """

        import math

        # --------------------------------------------------
        # 1. OTA-noisy model aggregation
        # --------------------------------------------------
        total_samples = sum(
            client["num_train_samples"]
            for client in client_info
        )

        snr_db = self.snr
        print(f"************  SNR (dB): {snr_db}  *********"  )
        snr_linear = 10 ** (snr_db / 10)

        global_state = self.network.state_dict()
        new_state = {}

        for key in global_state.keys():

            global_param = (
                global_state[key]
                .float()
                .to(self.device)
            )

            # weighted FedAvg delta
            agg_delta = torch.zeros_like(global_param)

            for client in client_info:

                weight = (
                    client["num_train_samples"]
                    / total_samples
                )

                client_param = (
                    client["state_dict"][key]
                    .float()
                    .to(self.device)
                )

                agg_delta += weight * (
                    client_param - global_param
                )

            # OTA channel noise
            sigma2 = 1.0 / snr_linear
            sigma = math.sqrt(sigma2)

            noise = torch.randn_like(agg_delta) * sigma

            received = agg_delta + noise

            new_param = global_param + received

            new_state[key] = new_param.to(
                global_state[key].dtype
            )

        self.network.load_state_dict(
            new_state,
            strict=True
        )

        # --------------------------------------------------
        # 2. Aggregate MoGs for current task
        # --------------------------------------------------
        if self.mogs.get(self.cur_task) is None:
            self.mogs[self.cur_task] = {}

        for client in client_info:

            client_gaussians = client["client_statistics"]

            if self.cur_task not in client_gaussians:
                continue

            for cls in client_gaussians[self.cur_task]:

                weight, mean, var = (
                    client_gaussians[self.cur_task][cls]
                )

                if cls not in self.mogs[self.cur_task]:
                    self.mogs[self.cur_task][cls] = [
                        [weight],
                        [mean],
                        [var],
                    ]
                else:
                    self.mogs[self.cur_task][cls][0].append(weight)
                    self.mogs[self.cur_task][cls][1].append(mean)
                    self.mogs[self.cur_task][cls][2].append(var)

        # normalize mixture weights
        for cls, (weights, means, vars_) in self.mogs[self.cur_task].items():

            total_weight = sum(weights)

            self.mogs[self.cur_task][cls][0] = [
                w / total_weight
                for w in weights
            ]

        # --------------------------------------------------
        # 3. Generative replay
        # --------------------------------------------------
        sampled_data = []
        sampled_labels = []

        num_classes = len(self.mogs[self.cur_task])

        samples_per_class = max(
            1,
            self.how_many // num_classes
        )

        for cls, (weights, means, vars_) in self.mogs[self.cur_task].items():

            weights_tensor = torch.tensor(
                weights,
                dtype=torch.float32
            ).to(self.device)

            sampled_clients = torch.multinomial(
                weights_tensor,
                samples_per_class,
                replacement=True
            )

            client_counts = torch.bincount(
                sampled_clients,
                minlength=len(weights)
            )

            for i, count in enumerate(client_counts):

                if count == 0:
                    continue

                mean = means[i]
                var = vars_[i]

                if self.full_cov:
                    cov = (
                        var
                        + 1e-8
                        * torch.eye(
                            mean.shape[-1]
                        ).to(self.device)
                    )
                else:
                    cov = (
                        torch.diag(var)
                        + 1e-8
                        * torch.eye(
                            mean.shape[-1]
                        ).to(self.device)
                    )

                m = MultivariateNormal(mean, cov)

                feats = m.sample((count,))

                sampled_data.append(feats)

                sampled_labels.extend(
                    [cls] * count
                )

        sampled_data = torch.cat(
            sampled_data,
            dim=0
        ).to(self.device)

        sampled_labels = torch.tensor(
            sampled_labels,
            dtype=torch.long
        ).to(self.device)

        perm = torch.randperm(
            sampled_data.size(0)
        )

        sampled_data = sampled_data[perm]
        sampled_labels = sampled_labels[perm]

        # --------------------------------------------------
        # 4. Retrain current task head
        # --------------------------------------------------
        optimizer = torch.optim.SGD(
            self.network.classifiers[
                self.cur_task
            ].parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0,
        )

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=5,
            )
        )

        batch_size = self.how_many

        for epoch in range(5):

            for i in range(
                0,
                sampled_data.size(0),
                batch_size,
            ):

                inp = sampled_data[
                    i : i + batch_size
                ]

                tgt = sampled_labels[
                    i : i + batch_size
                ]

                logits = self.network.classifiers[
                    self.cur_task
                ](inp)

                loss = F.cross_entropy(
                    logits,
                    tgt
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()        


#     def end_round_server(self, client_info: list,task_id=0):
#         """
#         PURE Task-IL server update
#         """

#         # -----------------------------
#         # 1️⃣ Aggregate backbone parameters (task-agnostic)
#         # -----------------------------
#         total_samples = sum([client["num_train_samples"] for client in client_info])
#         norm_weights = [client["num_train_samples"] / total_samples for client in client_info]

#         # -----------------------------
#         # Aggregate FULL model (Task-IL)
#         # -----------------------------


#         avg_state = {}
#         for key in client_info[0]["state_dict"].keys():
#             avg_state[key] = sum(
#                 c["state_dict"][key] * (c["num_train_samples"] / total_samples)
#                 for c in client_info
#             )

#         self.network.load_state_dict(avg_state, strict=True)


#         # -----------------------------
#         # 3️⃣ Aggregate MoGs ONLY for current task
#         # -----------------------------
#         if self.mogs.get(self.cur_task) is None:
#             self.mogs[self.cur_task] = {}

#         for client in client_info:
#             client_gaussians = client["client_statistics"]

#             # ✅ Explicit Task-IL guard
#             if self.cur_task not in client_gaussians:
#                 continue

#             for cls in client_gaussians[self.cur_task]:
#                 weight, mean, var = client_gaussians[self.cur_task][cls]

#                 if cls not in self.mogs[self.cur_task]:
#                     self.mogs[self.cur_task][cls] = [[weight], [mean], [var]]
#                 else:
#                     self.mogs[self.cur_task][cls][0].append(weight)
#                     self.mogs[self.cur_task][cls][1].append(mean)
#                     self.mogs[self.cur_task][cls][2].append(var)

#         # Normalize mixture weights (per class, per task)
#         for cls, (weights, means, vars_) in self.mogs[self.cur_task].items():
#             total_weight = sum(weights)
#             self.mogs[self.cur_task][cls][0] = [w / total_weight for w in weights]

#         # -----------------------------
#         # 4️⃣ Task-IL generative replay (current task ONLY)
#         # -----------------------------
#         sampled_data, sampled_labels = [], []

#         num_classes = len(self.mogs[self.cur_task])  # ✅ Task-local class count
#         samples_per_class = max(1, self.how_many // num_classes)


#         for cls, (weights, means, vars_) in self.mogs[self.cur_task].items():
#             weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
#             sampled_clients = torch.multinomial(
#                 weights_tensor, samples_per_class, replacement=True
#             )
#             client_counts = torch.bincount(sampled_clients, minlength=len(weights))

#             for i, count in enumerate(client_counts):
#                 if count == 0:
#                     continue
#                 mean = means[i]
#                 var = vars_[i]

#                 if hasattr(self, "full_cov") and self.full_cov:
#                     cov = var + 1e-8 * torch.eye(mean.shape[-1]).to(self.device)
#                 else:
#                     cov = torch.diag(var) + 1e-8 * torch.eye(mean.shape[-1]).to(self.device)

#                 m = MultivariateNormal(mean, cov)
#                 feats = m.sample((count,))
#                 sampled_data.append(feats)
#                 sampled_labels.extend([cls] * count)

#         sampled_data = torch.cat(sampled_data, 0).to(self.device)
#         sampled_labels = torch.tensor(sampled_labels, dtype=torch.long).to(self.device)

#         perm = torch.randperm(sampled_data.size(0))
#         sampled_data = sampled_data[perm]
#         sampled_labels = sampled_labels[perm]

#         # -----------------------------
#         # 5️⃣ Train ONLY current task head
#         # -----------------------------
#         optimizer = torch.optim.SGD(
#     self.network.classifiers[self.cur_task].parameters(),
#     lr=0.01, momentum=0.9, weight_decay=0
# )

#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

#         batch_size = self.how_many
#         for epoch in range(5):
#             for i in range(0, sampled_data.size(0), batch_size):
#                 inp = sampled_data[i:i + batch_size]
#                 tgt = sampled_labels[i:i + batch_size]

#                 logits = self.network.classifiers[self.cur_task](inp)
#                 loss = F.cross_entropy(logits, tgt)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             scheduler.step()


    def begin_round_client(self, dataloader: DataLoader, server_info: dict,task_id=0):
        self.network.load_state_dict(server_info["state_dict"], strict=True)
        if self.do_linear_probe and not self.done_linear_probe:
            optimizer = self.optimizer_class(self.network.parameters(), lr=self.lr, weight_decay=self.wd_reg)
            self.optimizer = self.fabric.setup_optimizers(optimizer)
            self.linear_probe(dataloader,task_id)
            self.done_linear_probe = True
        # restore correct optimizer
        params = [{"params": self.network.parameters()}]
        optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd_reg)
        self.optimizer = self.fabric.setup_optimizers(optimizer)

    def get_client_info(self, dataloader: DataLoader):
        return {
            "state_dict": self.network.state_dict(),
            "num_train_samples": len(dataloader.dataset.data),
            "client_statistics": self.clients_statistics,
        }

    def get_server_info(self):
        return {"state_dict": self.network.state_dict()}
    

    def end_round_client(self, dataloader: DataLoader,task_id=0):
        features = torch.tensor([], dtype=torch.float32).to(self.device)
        true_labels = torch.tensor([], dtype=torch.int64).to(self.device)
        num_epochs = 1 if not self.full_cov else 3
        with torch.no_grad():
            client_statistics = {}
            for _ in range(num_epochs):
                for id, data in enumerate(dataloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs = self.augment(inputs)
                    outputs = self.network(inputs, task_id=task_id,penultimate=True)[0]
                    features = torch.cat((features, outputs), 0)
                    true_labels = torch.cat((true_labels, labels), 0)
            client_labels = torch.unique(true_labels).tolist()
            for client_label in client_labels:
                number = (true_labels == client_label).sum().item()
                if number > 1:
                    gaussians = []
                    gaussians.append(number)
                    gaussians.append(torch.mean(features[true_labels == client_label], 0))
                    if self.full_cov:
                        gaussians.append(
                            torch.cov(features[true_labels == client_label].T.type(torch.float64))
                            .type(torch.float32)
                            .to(self.device)
                        )
                    else:
                        gaussians.append(torch.std(features[true_labels == client_label], 0) ** 2)
                    client_statistics[client_label] = gaussians
            # ✅ Task-IL: store statistics PER TASK 
            if self.clients_statistics is None:
                self.clients_statistics = {}
            self.clients_statistics[task_id] = client_statistics

    def save_checkpoint(self, output_folder: str, task: int, comm_round: int) -> None:
        training_status = self.network.training
        self.network.eval()

        checkpoint = {
            "task": task,
            "comm_round": comm_round,
            "network": self.network,
            "optimizer": self.optimizer,
            "mogs": self.mogs,
        }
        name = "hgp_" + "full_cov" if self.full_cov else "diag_cov"
        name += "_linear_probe" if self.do_linear_probe else ""
        name += f"_task_{task}_round_{comm_round}_checkpoint.pt"
        self.fabric.save(os.path.join(output_folder, name), checkpoint)
        self.network.train(training_status)
