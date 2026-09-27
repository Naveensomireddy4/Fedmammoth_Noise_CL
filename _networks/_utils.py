# import torch
# import torch.nn as nn


# class BaseNetwork(nn.Module):
#     """
#     BaseNetwork adapted for Task-Incremental Learning (Task-IL)

#     Assumes:
#       - Multiple classifier heads (e.g., self.classifiers)
#       - Shared backbone
#       - No class slicing
#     """

#     def __init__(self) -> None:
#         super().__init__()
#         self.embed_dim = 0

#     # ------------------------------------------------------------------
#     # PARAMETER VECTORIZATION (FedAvg)
#     # ------------------------------------------------------------------
#     def get_params(self) -> torch.Tensor:
#         """
#         Returns all model parameters as a single vector.
#         """
#         return torch.cat([p.data.view(-1) for p in self.parameters()])

#     def set_params(self, new_params: torch.Tensor) -> None:
#         """
#         Loads parameters from a flat vector.
#         """
#         assert new_params.numel() == self.get_params().numel()

#         progress = 0
#         for p in self.parameters():
#             numel = p.numel()
#             p.data.copy_(
#                 new_params[progress : progress + numel].view_as(p)
#             )
#             progress += numel

#     # ------------------------------------------------------------------
#     # GRADIENT HANDLING (OPTIONAL HEAD EXCLUSION)
#     # ------------------------------------------------------------------
#     def _is_classifier_param(self, name: str) -> bool:
#         """
#         Identifies classifier (task head) parameters.
#         """
#         return (
#             "classifier" in name
#             or "classifiers" in name
#             or name.endswith("head.weight")
#             or name.endswith("head.bias")
#         )

#     def get_grads(self, discard_classifier: bool = False) -> torch.Tensor:
#         """
#         Returns flattened gradients.
#         Optionally excludes all classifier heads.
#         """
#         grads = []

#         for name, p in self.named_parameters():
#             if p.grad is None:
#                 continue

#             if discard_classifier and self._is_classifier_param(name):
#                 continue

#             grads.append(p.grad.view(-1))

#         return torch.cat(grads) if len(grads) > 0 else torch.tensor([])

#     def set_grads(
#         self,
#         new_grads: torch.Tensor,
#         discard_classifier: bool = False,
#     ) -> None:
#         """
#         Loads gradients from a flat vector.
#         Optionally skips classifier heads.
#         """
#         progress = 0

#         for name, p in self.named_parameters():
#             if discard_classifier and self._is_classifier_param(name):
#                 continue

#             numel = p.numel()
#             p.grad = new_grads[
#                 progress : progress + numel
#             ].view_as(p).clone()

#             progress += numel


import torch
import torch.nn as nn

class BaseNetwork(nn.Module):
    """
    BaseNetwork adapted for Task-Incremental Learning (Task-IL)
    - Supports multiple classifier heads (self.classifiers)
    - Shared backbone
    - Flattened parameter and gradient vectors for EWC/FedAvg
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 0

    # ------------------------------------------------------------------
    # PARAMETER VECTORIZATION (FedAvg)
    # ------------------------------------------------------------------
    def get_params(self, discard_classifier: bool = False) -> torch.Tensor:
        """
        Returns all model parameters as a single flattened vector.
        Can optionally discard classifier (task head) parameters.
        """
        params = []
        for name, p in self.named_parameters():
            if discard_classifier and self._is_classifier_param(name):
                continue
            params.append(p.data.view(-1))
        return torch.cat(params) if params else torch.tensor([])

    def set_params(self, new_params: torch.Tensor, discard_classifier: bool = False) -> None:
        """
        Loads parameters from a flat vector.
        Must match the same ordering as get_params().
        """
        total_expected = self.get_params(discard_classifier).numel()
        assert new_params.numel() == total_expected, \
            f"Parameter vector size mismatch: expected {total_expected}, got {new_params.numel()}"

        progress = 0
        for name, p in self.named_parameters():
            if discard_classifier and self._is_classifier_param(name):
                continue
            numel = p.numel()
            p.data.copy_(new_params[progress:progress + numel].view_as(p))
            progress += numel

    # ------------------------------------------------------------------
    # GRADIENT HANDLING (OPTIONAL HEAD EXCLUSION)
    # ------------------------------------------------------------------
    def _is_classifier_param(self, name: str) -> bool:
        """
        Identifies classifier (task head) parameters.
        """
        return (
            "classifier" in name
            or "classifiers" in name
            or name.endswith("head.weight")
            or name.endswith("head.bias")
        )

    def get_grads(self, discard_classifier: bool = False) -> torch.Tensor:
        grads = []

        for name, p in self.named_parameters():
            if discard_classifier and self._is_classifier_param(name):
                continue

            if p.grad is None:
                grads.append(torch.zeros_like(p).view(-1))
            else:
                grads.append(p.grad.view(-1))

        return torch.cat(grads)


    def set_grads(self, new_grads: torch.Tensor, discard_classifier: bool = False) -> None:
        """
        Loads gradients from a flat vector.
        Must match the same ordering as get_grads().
        """
        total_expected = self.get_grads(discard_classifier).numel()
        assert new_grads.numel() == total_expected, \
            f"Gradient vector size mismatch: expected {total_expected}, got {new_grads.numel()}"

        progress = 0
        for name, p in self.named_parameters():
            if discard_classifier and self._is_classifier_param(name):
                continue
            numel = p.numel()
            grad_slice = new_grads[progress:progress + numel].view_as(p)
            if p.grad is None:
                p.grad = grad_slice.clone()
            else:
                p.grad.copy_(grad_slice)
            progress += numel
