# Implementation of LIVAR method from Intrinsic Training Signals for Federated Learning Aggregation paper, https://arxiv.org/pdf/2507.06813 
# All experiments must be run with num_tasks = 1 since the method is not designed for continual learning

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from _models import register_model
from typing import List
from _models._utils import BaseModel
from _networks.vit import VisionTransformer as Vit
from torch.func import functional_call
from copy import deepcopy
from utils.tools import str_to_bool
import numpy as np
from torch.distributions import MultivariateNormal

import math
from typing import Optional, List

@register_model("livar")
class Lora(BaseModel):
    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        clip_grad: str_to_bool = False,
        wd_reg: float = 0.1,
        avg_type: str = "head_class_var_weighted",
        lora_alpha: float = 1.0,
        r: int = 16,
        lora_head: str_to_bool = False,
        cl_merge: str = "individual_mean",
        proxy_weighting: str_to_bool = True,
        si_withmomentum: str_to_bool = True,
        total_weights_sum: float = 1.0,
        doalsoccvr: str_to_bool = False,
        how_many: int = 256,
        full_cov: str_to_bool = False,
        linear_probe: str_to_bool = False,
    ) -> None:
        super().__init__(fabric, network, device, optimizer, lr, wd_reg)
        self.lora_alpha = lora_alpha
        self.r = r
        self.cl_merge = cl_merge
        self.lora_keys = []
        self.lora_params = {}
        self.optimizer_str = optimizer
        self.lr = lr
        self.clip_grad = clip_grad
        self.wd_reg = wd_reg
        self.lora_head = lora_head
        self.head_keys = []
        self.old_delta = {}
        self.cur_B = {}
        self.cur_A = {}
        self.curr_W = {}
        self.init_lora_params(network, r)
        self.optimization_dict = {}
        self.lora_total_grads_A = {n: 0 for n in self.lora_keys}
        self.lora_total_grads_B = {n: 0 for n in self.lora_keys}
        self.clients_statistics = None
        if not self.lora_head:
            self.head = {
                key: nn.Parameter(torch.tensor(self.network.state_dict()[key].clone().detach()), requires_grad=True).to(
                    self.device
                )
                for key in self.head_keys
            }
        self.avg_type = avg_type
        self.proxy_weighting = proxy_weighting
        self.si_withmomentum =si_withmomentum
        self.total_weights_sum = total_weights_sum
        self.doalsoccvr = doalsoccvr
        # parameters from ccvr
        self.cpt = []
        self.how_many = how_many
        self.full_cov = full_cov
        self.mogs = {}
        self.logit_norm = 0.1
        self.do_linear_probe = linear_probe
        self.done_linear_probe = False
    
    def init_lora_params(self, network, r):
        for name, param in network.named_parameters():
            param.requires_grad = False 
            if not self.lora_head and "head" in name:
                self.head_keys.append(name)
            if ("qkv" in name and "weight" in name) or (
                ("mlp" in name and "weight" in name)
                or ("proj" in name and "weight" in name and "attn" in name)
                or (self.lora_head and "head" in name and "weight" in name)
            ):
                self.lora_keys.append(name)
                self.lora_params[name] = {name: [param.shape[1], param.shape[0]]}
                self.old_delta[name] = nn.Parameter(
                    torch.zeros(param.shape[0], param.shape[1]), requires_grad=False
                ).to(self.device)
                self.cur_B[name] = nn.Parameter(torch.zeros(param.shape[0], r), requires_grad=True).to(self.device)
                self.cur_A[name] = nn.Parameter(torch.zeros(r, param.shape[1]), requires_grad=True).to(self.device)
                nn.init.kaiming_uniform_(self.cur_A[name], a=math.sqrt(5))
                self.curr_W[name] = self.cur_B[name] @ self.cur_A[name]


    def init_matrices(self, reverse=False):
        for key in self.lora_keys:
            self.cur_B[key] = nn.Parameter(torch.zeros_like(self.cur_B[key]), requires_grad=True).to(self.device)
            self.cur_A[key] = nn.Parameter(torch.zeros_like(self.cur_A[key]), requires_grad=True).to(self.device)
            if not reverse:
                nn.init.kaiming_uniform_(self.cur_A[key], a=math.sqrt(5))
            else:
                nn.init.kaiming_uniform_(self.cur_B[key], a=math.sqrt(5))
            self.curr_W[key] = self.cur_B[key] @ self.cur_A[key]

    def set_optimization(self):
        self.optimization_dict = deepcopy(dict(self.network.state_dict()))
        for key in self.optimization_dict.keys():
            self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
        for key in self.lora_keys:
            self.old_delta[key].requires_grad = False
            self.cur_B[key].requires_grad = False
            self.cur_A[key].requires_grad = False
            self.curr_W[key].requires_grad = False
            self.cur_A[key] = self.cur_A[key].to(self.device)
            self.cur_B[key] = self.cur_B[key].to(self.device)
            self.curr_W[key]= self.curr_W[key].to(self.device)
        if "run_sum" in self.cl_merge:
            for key in self.lora_keys:
                if self.cur_task > 0:
                    self.optimization_dict[key] += self.old_delta[key]
                self.optimization_dict[key] += self.curr_W[key] 
        elif "run_mean" in self.cl_merge:
            for key in self.lora_keys:
                if self.cur_task > 0:
                    tmp = (self.old_delta[key] * self.cur_task) + self.curr_W[key] 
                    self.optimization_dict[key] += tmp / (self.cur_task + 1)
                else:
                    self.optimization_dict[key] += self.curr_W[key] 
        elif "individual" in self.cl_merge:
            if "sum" in self.cl_merge:
                for key in self.lora_keys:
                    if self.cur_task > 0:
                        tmp = (self.old_delta[key] * self.cur_task) + self.curr_W[key] 
                        self.optimization_dict[key] += tmp
                    else:
                        self.optimization_dict[key] += self.curr_W[key] 
            elif "mean" in self.cl_merge:
                for key in self.lora_keys:
                    if self.cur_task > 0:
                        tmp = (self.old_delta[key] * self.cur_task) + self.curr_W[key].detach() 
                        self.optimization_dict[key] += tmp / (self.cur_task + 1)
                    else:
                        self.optimization_dict[key] += self.curr_W[key].detach() 

        else:
            raise ValueError("Invalid cl_merge type")

    def get_optimization_dict(self, fabric=True):
        if fabric:
            optimization_dict = deepcopy(dict(self.network.state_dict()))
        else:
            optimization_dict = deepcopy(dict(self.network.module.state_dict()))
        if not self.lora_head:
            for key in self.head_keys:
                self.head[key].requires_grad = True
                optimization_dict[key] = self.head[key]
        for key in self.lora_keys:
            self.old_delta[key].requires_grad = False
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
            if self.cur_task > 0 and not "individual" in self.cl_merge:
                optimization_dict[key] += self.old_delta[key]
            optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        return optimization_dict

    def begin_task(self, n_classes_per_task: int):
        self.cur_task += 1
        if self.cur_task > 0:
            self.cur_offset += self.cpt[-1]
        self.cpt.append(n_classes_per_task)

        if self.cur_task > 0:
            if self.cl_merge == "run_sum":
                for key in self.lora_keys:
                    self.old_delta[key] += self.cur_B[key].detach() @ self.cur_A[key].detach()
            elif self.cl_merge == "run_mean" or "individual" in self.cl_merge:
                for key in self.lora_keys:
                    self.old_delta[key] = (
                        self.old_delta[key] * (self.cur_task - 1) + self.cur_B[key].detach() @ self.cur_A[key].detach()
                    ) / self.cur_task
            else:
                raise ValueError("Invalid cl_merge type")
            self.init_matrices()

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        for key in self.lora_keys:
            self.old_delta[key] = self.old_delta[key].detach()
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
        self.old_delta = deepcopy(server_info["old_delta"])
        if not self.lora_head:
            self.network.model.head.load_state_dict(server_info["head"])
            self.head = {
                key: nn.Parameter(self.network.state_dict()[key].clone().detach(), requires_grad=True).to(self.device)
                for key in self.head_keys
            }

        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        if not self.lora_head:
            self.optimizer = OptimizerClass(
                list(self.cur_B.values()) + list(self.cur_A.values()) + list(self.head.values()),
                lr=self.lr,
                weight_decay=self.wd_reg,
            )
        else:
            self.optimizer = OptimizerClass(
                list(self.cur_B.values()) + list(self.cur_A.values()), lr=self.lr, weight_decay=self.wd_reg
            )
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        optimization_dict = self.get_optimization_dict()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            outputs = functional_call(self.network, optimization_dict, inputs)[
                :, self.cur_offset : self.cur_offset + self.cpt[-1]
            ]
            loss = self.loss(outputs, labels - self.cur_offset)

        if update:
            self.fabric.backward(loss)
            pre_values_A = {n: self.cur_A[n].detach().cpu().data.clone() for n in
                            self.lora_keys}
            pre_values_B = {n: self.cur_B[n].detach().cpu().data.clone() for n in
                           self.lora_keys}
            lora_grads_A = {n: torch.nan_to_num(self.cur_A[n].grad, 0).cpu().data for n in
                            self.lora_keys}
            lora_grads_B = {n: torch.nan_to_num(self.cur_B[n].grad, 0).cpu().data for n in
                            self.lora_keys}
            if self.clip_grad:
                try:
                    self.fabric.clip_gradients(self.network, self.optimizer, max_norm=1.0, norm_type=2)
                except:
                    pass

            self.optimizer.step()

            if self.si_withmomentum:
                #add the momentum to the grads
                lora_grads_A = {n: torch.sum(lora_grads_A[n] * (pre_values_A[n] - self.cur_A[n].detach().cpu().data.clone())).abs() for n in
                                self.lora_keys}
                lora_grads_B = {n: torch.sum(lora_grads_B[n] * (pre_values_B[n] - self.cur_B[n].detach().cpu().data.clone())).abs() for n in
                                self.lora_keys}
            else:
                lora_grads_A = {n: torch.sum(lora_grads_A[n]).abs() for n in
                                self.lora_keys} 
                lora_grads_B = {n: torch.sum(lora_grads_B[n]).abs() for n in
                                self.lora_keys}
            for n in self.lora_keys:
                self.lora_total_grads_A[n] += torch.nan_to_num(lora_grads_A[n], 0)
                self.lora_total_grads_B[n] += torch.nan_to_num(lora_grads_B[n], 0)
        return loss.item()

    def forward(self, x, fabric=True):
        if fabric:
            return functional_call(self.network, self.optimization_dict, x)
        return functional_call(self.network.module, self.optimization_dict, x)

    def forward_2(self, x):
        return self.network.module(x)

    def get_client_info(self, dataloader: DataLoader):
        for key in self.lora_keys:
            self.old_delta[key] = self.old_delta[key].detach()
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
        client_info = {
            "cur_A": deepcopy(self.cur_A),
            "cur_B": deepcopy(self.cur_B),
            "num_train_samples": len(dataloader.dataset.data),
            "client_statistics": self.clients_statistics,
            "lora_total_grads_A": deepcopy(self.lora_total_grads_A),
            "lora_total_grads_B": deepcopy(self.lora_total_grads_B),
            'num_train_samples_per_class': {dataloader.dataset.class_to_idx[cl]:
                                             np.count_nonzero(dataloader.dataset.targets ==
                                                              dataloader.dataset.class_to_idx[
                                                                  cl])
                                         for cl in dataloader.dataset.classes},
        }
        client_info["all_classes"] = list(client_info["num_train_samples_per_class"].keys())
        if not self.lora_head:
            client_info["head"] = deepcopy(self.network.model.head.state_dict())
        return client_info

    def get_server_info(self):
        for key in self.lora_keys:
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
        server_info = {
            "cur_A": deepcopy(self.cur_A),
            "cur_B": deepcopy(self.cur_B),
        }
        if getattr(self, "old_delta", None) is not None:
            for key in self.lora_keys:
                self.old_delta[key] = self.old_delta[key].detach()
            server_info["old_delta"] = self.old_delta
        if not self.lora_head:
            server_info["head"] = deepcopy(self.network.model.head.state_dict())
        return server_info

    def end_round_client(self, dataloader: DataLoader):
        if not self.lora_head:
            sd = self.network.state_dict()
            for key in self.head_keys:
                sd[key] = self.head[key]
            self.network.load_state_dict(sd)

        features = torch.tensor([], dtype=torch.float32).to(self.device)
        true_labels = torch.tensor([], dtype=torch.int64).to(self.device)

        num_epochs = 1 if not self.full_cov else 3
        with torch.no_grad():
            client_statistics = {}
            for _ in range(num_epochs):
                for id, data in enumerate(dataloader):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, _ = self.network(inputs, penultimate=True)
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
            self.clients_statistics = client_statistics

    def clients_sample_var_weights(self, client_info, all_classes):
        total_vars_per_class = {
            cla: sum([single_client_info['client_statistics'][cla][2].mean().cpu() if cla in single_client_info['client_statistics']
                      else 0 for single_client_info in client_info])
            for cla in all_classes}

        norm_vars_per_class = [{cla: single_client_info['client_statistics'][cla][2].mean().cpu() /
                                     total_vars_per_class[cla]
                                    if cla in single_client_info['client_statistics'] and total_vars_per_class[cla] > 0
                                    else torch.tensor(0)
                                    for cla in all_classes} for single_client_info in client_info]
        return norm_vars_per_class
    
    def weighted_head_per_class(self, norm_weights_per_class, clients_head):
        classes_covered = set()
        for norm_weight in norm_weights_per_class:
            classes_covered = classes_covered.union(set([k for k,v in norm_weight.items() if v>0]))

        classes_not_covered = torch.tensor([0 if k in classes_covered else 1/len(clients_head) for k in norm_weights_per_class[0].keys()])
        head_sd = self.network.model.head.state_dict()
        for key in head_sd.keys():
            if 'bias' in key:
                head_sd[key] = torch.stack(
                    [(head[key].T * torch.tensor(list(norm_weight.values())) + head_sd[key].cpu() * classes_not_covered).T
                     for head, norm_weight in zip(clients_head, norm_weights_per_class)]
                ).sum(0)
            else:
                head_sd[key] = torch.stack(
                    [(head[key].T * torch.tensor(list(norm_weight.values())) + head_sd[key].cpu().T * classes_not_covered).T
                     for head, norm_weight in zip(clients_head, norm_weights_per_class)]
                ).sum(0)
        return head_sd
    
    def end_round_server(self, client_info: List[dict]):
        all_classes = client_info[0]['all_classes']
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        elif self.avg_type == "head_class_var_weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
            norm_weights_per_class = self.clients_sample_var_weights(client_info, all_classes)
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        cl_B = [client["cur_B"] for client in client_info]  # list of B matrices for all clients
        cl_A = [client["cur_A"] for client in client_info]  # list of A matrices for all clients
        if not self.lora_head:
            if self.avg_type == "head_class_var_weighted":
                head_sd = self.weighted_head_per_class(norm_weights_per_class,[client["head"] for client in client_info])
            else:
                heads = [client["head"] for client in client_info]  # list of head matrices for all clients
                head_sd = self.network.model.head.state_dict()
                for key in head_sd.keys():
                    head_sd[key] = torch.stack(
                        [head[key] * norm_weight for head, norm_weight in zip(heads, norm_weights)]
                    ).sum(0)
            self.network.model.head.load_state_dict(head_sd)
        if len(client_info) > 0:
            A25 = torch.quantile(
                torch.tensor([client["lora_total_grads_A"][key] for client in client_info for key in self.lora_keys]),
                0.25)
            A50 = torch.quantile(
                torch.tensor([client["lora_total_grads_A"][key] for client in client_info for key in self.lora_keys]),
                0.5)
            B60 = torch.quantile(
                torch.tensor([client["lora_total_grads_B"][key] for client in client_info for key in self.lora_keys]),
                0.6)
            B80 = torch.quantile(
                torch.tensor([client["lora_total_grads_B"][key] for client in client_info for key in self.lora_keys]),
                0.8)
            for key in self.lora_keys:
                if self.proxy_weighting == True:
                    weights_clients_delta = []
                    for client in client_info:
                        if client["lora_total_grads_B"][key] < B60:
                            weights_clients_delta.append(torch.tensor(0.03813978))
                        elif client["lora_total_grads_B"][key] > B80:
                            if client["lora_total_grads_A"][key] < A25:
                                weights_clients_delta.append(torch.tensor(0.19313978))
                            elif client["lora_total_grads_A"][key] > A50:
                                weights_clients_delta.append(torch.tensor(0.09313978))
                            else:
                                weights_clients_delta.append(torch.tensor(0.11813978))
                        else:
                            if client["lora_total_grads_A"][key] < A25:
                                weights_clients_delta.append(torch.tensor(0.09813978))
                            elif client["lora_total_grads_A"][key] > A50:
                                weights_clients_delta.append(torch.tensor(0.07813978))
                            else:
                                weights_clients_delta.append(torch.tensor(0.08813978))

                    weights_clients_delta = [x / torch.sum(torch.tensor(weights_clients_delta)) for x in
                                                    weights_clients_delta]
                    weights_clients_delta = [w if not torch.isnan(w) else norm_weights[i] for i,w in enumerate(weights_clients_delta)]
                    weights_clients_delta = [w/self.total_weights_sum for w in weights_clients_delta]
                    self.cur_B[key] = nn.Parameter(
                            torch.stack(
                                [client[key] * norm_weight for client, norm_weight in zip(cl_B, weights_clients_delta)]
                            ).sum(0)
                        )
                    self.cur_A[key] = nn.Parameter(
                            torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_A, weights_clients_delta)]
                                        ).sum(0)
                        )
                    self.curr_W[key] = nn.Parameter(
                            torch.stack([(client_B[key] @ client_A[key]) * norm_weight for client_A, client_B, norm_weight in zip(cl_A, cl_B, norm_weights)]).sum(0)
                        )
                else:
                    self.cur_B[key] = nn.Parameter(
                        torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_B, norm_weights)]).sum(0)
                    )
                    self.cur_A[key] = nn.Parameter(
                        torch.stack([client[key] * norm_weight for client, norm_weight in zip(cl_A, norm_weights)]).sum(0)
                    )
                    self.curr_W[key] = nn.Parameter(
                        torch.stack([(client_B[key] @ client_A[key]) * norm_weight for client_A, client_B, norm_weight in zip(cl_A, cl_B, norm_weights)]).sum(0)
                    )

        if self.doalsoccvr: # Used for ablation studies in the paper
            clients_gaussians = [client["client_statistics"] for client in client_info]
            mogs = {}
            for clas in range(self.cur_offset, self.cur_offset + self.cpt[-1]):
                counter = 0
                for client_gaussians in clients_gaussians:
                    if client_gaussians.get(clas) is not None:
                        gaus_data = []
                        gaus_mean = client_gaussians[clas][1]
                        gaus_var = client_gaussians[clas][2]
                        gaus_data.append(gaus_mean)
                        gaus_data.append(gaus_var)
                        weight = client_gaussians[clas][0]
                        if mogs.get(clas) is None:
                            mogs[clas] = [[weight], [gaus_mean], [gaus_var]]
                        else:
                            mogs[clas][0].append(weight)
                            mogs[clas][1].append(gaus_mean)
                            mogs[clas][2].append(gaus_var)
                        counter += client_gaussians[clas][0]
                if mogs.get(clas) is not None:
                    mogs[clas][0] = [mogs[clas][0][i] / counter for i in range(len(mogs[clas][0]))]
            self.mogs = mogs
            if "t5" not in str(type(self.network.model)).lower():
                copy_head = deepcopy(self.network.model.head)
                copy_head = copy_head.to(self.device)
                for n,v in copy_head.named_parameters():
                    v.requires_grad = True
                optimizer_head = torch.optim.SGD(copy_head.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
            scheduler_head = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_head, T_max=5)
            logits_norm = torch.tensor([], dtype=torch.float32).to(self.device)
            for epoch in range(5):
                sampled_data = []
                sampled_label = []
                num_cur_classes = self.cpt[-1]
                classes_weights = torch.ones(num_cur_classes, dtype=torch.float32).to(self.device)
                classes_samples = torch.multinomial(classes_weights, self.how_many * num_cur_classes, replacement=True)
                _, classes_samples = torch.unique(classes_samples, return_counts=True)
                # sample features from gaussians:
                for clas in range(self.cur_offset, self.cur_offset + self.cpt[-1]):
                    if self.mogs.get([clas][0]) is None:
                        continue
                    weights_list = []
                    for weight in self.mogs[clas][0]:
                        weights_list.append(weight)
                    gaussian_samples = torch.zeros(len(weights_list), dtype=torch.int64).to(self.device)
                    weights_list = torch.tensor(weights_list, dtype=torch.float32).to(self.device)
                    gaussian_samples_fill = torch.multinomial(
                        weights_list, classes_samples[clas - self.cur_offset], replacement=True
                    )
                    gaussian_clients, gaussian_samples_fill = torch.unique(gaussian_samples_fill, return_counts=True)
                    gaussian_samples[gaussian_clients] += gaussian_samples_fill
                    for id, (mean, variance) in enumerate(
                            zip(
                                self.mogs[clas][1],
                                self.mogs[clas][2],
                            )
                    ):
                        cls_mean = mean 
                        cls_var = variance
                        if self.full_cov:
                            cov = cls_var + 1e-8 * torch.eye(cls_mean.shape[-1]).to(self.device)
                        else:
                            cov = (torch.eye(cls_mean.shape[-1]).to(self.device) * cls_var) + (
                                        1e-8 * torch.eye(cls_mean.shape[-1]).to(self.device))
                        m = MultivariateNormal(cls_mean, cov)
                        n_samples = int(torch.round(gaussian_samples[id]))
                        sampled_data_single = m.sample((n_samples,))
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([clas] * n_samples)
                sampled_data = torch.cat(sampled_data, 0).float().to(self.device)
                sampled_label = torch.tensor(sampled_label, dtype=torch.int64).to(self.device)
                inputs = sampled_data
                targets = sampled_label

                sf_indexes = torch.randperm(inputs.size(0))
                inputs = inputs[sf_indexes]
                targets = targets[sf_indexes]
                for _iter in range(self.cpt[-1]):
                    inp = inputs[_iter * self.how_many: (_iter + 1) * self.how_many].to(self.device)
                    tgt = targets[_iter * self.how_many: (_iter + 1) * self.how_many].to(self.device)
                    if "t5" not in str(type(self.network.model)).lower():
                        outputs = copy_head(inp) 
                    else:
                        outputs = self.network.head(inp)
                    logits = outputs[:, self.cur_offset:self.cur_offset + self.cpt[-1]]
                    temp_norm = torch.norm(logits, p=2, dim=-1, keepdim=True)
                    norms = temp_norm.mean(dim=-1, keepdim=True)

                    decoupled_logits = torch.div(logits + 1e-12, norms + 1e-12) / self.logit_norm
                    loss_head = F.cross_entropy(decoupled_logits, tgt - self.cur_offset)
                    loss_head.backward()
                    optimizer_head.step()
                    optimizer_head.zero_grad()
                scheduler_head.step()

            self.network.model.head = deepcopy(copy_head)
        self.set_optimization()

    def to(self, device="cpu"):
        self.network.to(device)
        for key in self.lora_keys:
            self.cur_B[key] = self.cur_B[key].to(device)
            self.cur_A[key] = self.cur_A[key].to(device)
            self.old_delta[key] = self.old_delta[key].to(device)
        if not self.lora_head:
            for key in self.head_keys:
                self.head[key] = self.head[key].to(device)
        return self
