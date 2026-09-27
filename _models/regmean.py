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

from _models.lora import Lora, merge_AB, zero_pad
from tqdm import tqdm


@register_model("regmean")
class RegMean(BaseModel):

    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 1e-5,
        wd_reg: float = 0.1,
        avg_type: str = "weighted",
        regmean_all: str_to_bool = True,
        snr:None = 0,
        alpha_regmean_head: float = 0.5,
        alpha_regmean_backbone: float = -1,
        gram_dtype: str = "32",
        reg_dtype_64: str_to_bool = True,
        lr_back: float = -1,
        only_square: int = 0,
        train_bias: str = "all",
        clip_grad: str_to_bool = False,
        linear_probe_epochs: int = 0,
        regmean_rounds: int = 1,
    ) -> None:

        self.reg_dtype_64 = reg_dtype_64
        self.optimizer_str = optimizer
        self.lr = lr
        self.wd_reg = wd_reg
        self.regmean_rounds = regmean_rounds
        self.clip_grad = clip_grad

        if alpha_regmean_backbone < 0:
            alpha_regmean_backbone = alpha_regmean_head

        self.lr_back = lr_back if lr_back >= 0 else lr

        super().__init__(fabric, network, device, optimizer, lr, wd_reg,snr)

        self.avg_type = avg_type
        self.alpha_regmean = [alpha_regmean_backbone, alpha_regmean_head]

        # dtype
        self.gram_dtype = (
            torch.float32 if gram_dtype == "32"
            else torch.float16 if gram_dtype == "16"
            else torch.bfloat16 if gram_dtype == "b16"
            else torch.float64
        )

        # -------------------------------------------------
        # SELECT BACKBONE LAYERS ONLY (CRITICAL FIX)
        # -------------------------------------------------
        self.gram_modules = []
        self.middle_names = {}

        for name, module in self.network.named_modules():
            if (
                len(list(module.parameters())) > 0
                and len(list(module.children())) == 0
                and "classifiers" not in name   # ❌ exclude task heads
                and "head" not in name          # extra safety
                and (
                    "mlp" in name
                    or ("proj" in name and "attn" in name)
                    or "qkv" in name
                )
            ):
                self.gram_modules.append(name)
                clean = name.replace("_forward_module.", "").replace("module.", "")
                self.middle_names[clean + ".weight"] = name

        print("RegMean layers:", self.gram_modules)  # helpful debug

        self.features = {key: None for key in self.gram_modules}
        self.linear_probe_epochs = linear_probe_epochs
        self.classifier = None
        self.snr=snr
    # -----------------------
    # Backbone vs Head split
    # -----------------------
    def split_backbone_head(self):
        backbone_params = []
        head_params = []
        for n, p in self.network.named_parameters():
            if "classifiers" in n:
                head_params.append(p)
            else:
                backbone_params.append(p)
        return backbone_params, head_params

    # -----------------------
    # Forward
    # -----------------------
    def forward(self, x: torch.Tensor, task_id: int, penultimate: bool = False, str=""):
        if self.classifier is not None:
            x = x
            feat, _ = self.network(x, task_id, penultimate=True)
            x = self.classifier(feat)
        else:
            if penultimate:
                x = x
                feat, logits = self.network(x, task_id, penultimate=True)
                return feat, logits
            else:
                x = x
                x = self.network(x, task_id)
        return x

    # -----------------------
    # Observation step
    # -----------------------
    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, task_id, update: bool = True):
        self.optimizer.zero_grad()
        with self.fabric.autocast():
            
            outputs = self.network(inputs, task_id)
            loss = self.loss(outputs, labels)

        if update:
            self.fabric.backward(loss)
            if self.clip_grad:
                self.fabric.clip_gradients(self.network, self.optimizer, max_norm=1.0, norm_type=2)
            self.optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
        return loss.item(), preds

    # -----------------------
    # Begin task
    # -----------------------
    def begin_task(self, task_id: int):
        self.classifier = None
        self.cur_task = task_id
        return super().begin_task(task_id)

    # -----------------------
    # Begin client round
    # -----------------------
    def begin_round_client(self, dataloader: DataLoader, server_info: dict, task_id):
        sd = server_info["state_dict"]
        self.network.load_state_dict(sd)
        self.network.train()
        backbone_params, head_params = self.split_backbone_head()
        params = [{"params": backbone_params, "lr": self.lr_back}, {"params": head_params}]
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        self.optimizer = OptimizerClass(params, lr=self.lr, weight_decay=self.wd_reg)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)
        for name in self.gram_modules:
            self.features[name] = None

    # -----------------------
    # End client round (Gram computation)
    # -----------------------
    def end_round_client(self, dataloader: DataLoader, task):
        self.network.eval()
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            self.optimizer = None

        hooks = {}
        for name, module in self.network.named_modules():
            if name in self.gram_modules:
                hooks[name] = module.register_forward_hook(self.hook_handler(name))
        total_samples = 0
        with torch.no_grad():
            for _ in range(self.regmean_rounds):
                for x, y in tqdm(dataloader, desc="Computing Gram matrices"):
                    total_samples+=x.shape[0]
                    x, y = x.to(self.device), y.to(self.device)
                    
                    self.forward(x, task_id=task)

        # Apply alpha + identity
        for name in self.gram_modules:
            if self.features[name] is not None:
                self.features[name] /= 1 #total_samples
                alpha = self.alpha_regmean[1] if "classifiers" in name else self.alpha_regmean[0]
                self.features[name] = self.features[name].to("cpu")
                shape = self.features[name].shape[-1]
                I = torch.eye(shape, dtype=self.gram_dtype)
                self.features[name] = alpha * self.features[name] + (1 - alpha) * I

        for h in hooks.values():
            h.remove()

    # -----------------------
    # Forward hook
    # -----------------------
    def hook_handler(self, name):
        def hook_forward(module, inputs, _):
            x = inputs[0].detach().to(self.gram_dtype)

            if len(x.shape) == 3:
                x = x.view(-1, x.size(-1))

              # number of feature vectors
            tmp = (x.T @ x)   # ✅ NORMALIZED GRAM

            if self.features[name] is None:
                self.features[name] = tmp
            else:
                self.features[name] = self.features[name] + tmp
        return hook_forward

    # -----------------------
    # Client info
    # -----------------------
    def get_client_info(self, dataloader: DataLoader):
        client_info = super().get_client_info(dataloader)
        if client_info is None:
            client_info = {}
        client_info["state_dict"] = deepcopy(self.network.state_dict())
        client_info["grams"] = deepcopy(self.features)
        client_info["num_train_samples"] = len(dataloader.dataset.data)
        return client_info

    # -----------------------
    # Move to device
    # -----------------------
    def to(self, device="cpu"):
        self.network.to(device)
        for name in self.gram_modules:
            if self.features[name] is not None:
                self.features[name] = self.features[name].to(device)
        return self

    # -----------------------
    # Server info
    # -----------------------
    def get_server_info(self):
        return {"state_dict": deepcopy(self.network.state_dict())}

    # -----------------------
    # End server round
    # -----------------------
    def end_round_server(self, client_info: List[dict], task):

        if len(client_info) == 0:
            return

        snr_db = self.snr   # default SNR

        # -----------------------------
        # Client weights
        # -----------------------------
        if self.avg_type == "weighted":
            total_samples = sum(c["num_train_samples"] for c in client_info)
            norm_weights = [c["num_train_samples"] / total_samples for c in client_info]
        else:
            weights = [1 if c["num_train_samples"] > 0 else 0 for c in client_info]
            norm_weights = [w / sum(weights) for w in weights]

        dtype = torch.float64 if self.reg_dtype_64 else self.gram_dtype
        sd = self.network.state_dict()

        # -----------------------------
        # Layer-wise aggregation
        # -----------------------------
        for key in sd.keys():

            # =============================
            # REGMEAN for BACKBONE weights
            # =============================
            if "weight" in key and key in self.middle_names:

                name = self.middle_names[key]

                G_sum = None
                WG_sum = None

                for client, w in zip(client_info, norm_weights):
                    gram = client["grams"].get(name)
                    if gram is None:
                        continue

                    G = gram.to(dtype)
                    W = client["state_dict"][key].to(dtype)

                    if G_sum is None:
                        G_sum = w * G
                        WG_sum = w * (W @ G)
                    else:
                        G_sum += w * G
                        WG_sum += w * (W @ G)

                if G_sum is None:
                    agg_param = torch.stack(
                        [client["state_dict"][key] * w for client, w in zip(client_info, norm_weights)]
                    ).sum(0)
                else:
                    eps = 1e-4 * torch.trace(G_sum) / G_sum.shape[0]
                    eps = eps * torch.eye(G_sum.shape[0], device=G_sum.device, dtype=dtype)

                    agg_param = (WG_sum @ torch.linalg.pinv(G_sum + eps)).to(torch.float32)

            # =============================
            # FedAvg for everything else
            # =============================
            else:
                agg_param = torch.stack(
                    [client["state_dict"][key] * w for client, w in zip(client_info, norm_weights)]
                ).sum(0)

            # =============================
            # Add SNR Noise
            # =============================
            signal_power = torch.mean(agg_param ** 2)
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise_std = torch.sqrt(noise_power)

            noise = torch.randn_like(agg_param) * noise_std
            sd[key] = agg_param + noise

        self.network.load_state_dict(sd)

    # -----------------------
    # End task client
    # -----------------------
    def end_task_client(self, dataloader: DataLoader = None, server_info: dict = None,task_id=0):
        return dataloader

    # -----------------------
    # End task server
    # -----------------------
    def end_task_server(self, client_info: List[dict] = None): 
        if self.linear_probe_epochs == 0 or client_info is None:
            return
        self.network.train()
        features = []
        labels_ = []
        embed_dim = self.network.model.head.weight.shape[1]
        num_classes = self.network.model.head.weight.shape[0]
        self.classifier = nn.Linear(embed_dim, num_classes).to(self.device)
        nn.init.xavier_normal_(self.classifier.weight)
        torch.cuda.empty_cache()

        for dl in client_info:
            for i, (inputs, labels) in enumerate(dl):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with self.fabric.autocast(), torch.no_grad():
                   
                    prelogits, _ = self.network(inputs, task_id=self.cur_task, penultimate=True)
                features.append(prelogits.to("cpu"))
                labels_.append(labels.to("cpu"))

        features = torch.cat(features)
        labels_ = torch.cat(labels_)

        batch_size = 256
        lr = 1e-3
        params = [{"params": self.classifier.parameters()}]
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        optimizer = OptimizerClass(params, lr=lr, weight_decay=self.wd_reg)

        for epoch in tqdm(range(self.linear_probe_epochs)):
            for i in range(0, len(features), batch_size):
                inputs, labels = features[i:i+batch_size].to(self.device), labels_[i:i+batch_size].to(self.device)
                optimizer.zero_grad()
                outputs = self.classifier(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                optimizer.step()

        self.network.eval()
