"""
Tensor Pagination for PaddleOCR-VL-1.5

This module manages large image tensors by offloading them to CPU/Disk
when VRAM is constrained, enabling 'spotting' on weak hardware.
"""

import torch
import logging

logger = logging.getLogger(__name__)

class TensorPaginator:
    def __init__(self, device: str = "cuda", offload_device: str = "cpu"):
        self.device = device
        self.offload_device = offload_device
        self.paginated_tensors = {} # id -> tensor (on cpu)

    def paginate(self, tensor: torch.Tensor, tensor_id: str):
        """Moves tensor to offload device to free up VRAM"""
        if tensor.device.type != self.offload_device:
            self.paginated_tensors[tensor_id] = tensor.to(self.offload_device)
            logger.debug(f"Paginated tensor {tensor_id} to {self.offload_device}")
        else:
            self.paginated_tensors[tensor_id] = tensor

    def retrieve(self, tensor_id: str) -> torch.Tensor:
        """Moves tensor back to active device for computation"""
        if tensor_id in self.paginated_tensors:
            tensor = self.paginated_tensors[tensor_id].to(self.device)
            # Optional: keep on CPU? Or remove?
            # For strict pagination we might keep it on CPU and only bring copy.
            return tensor
        raise KeyError(f"Tensor {tensor_id} not found in pagination")

    def cleanup(self, tensor_id: str):
        if tensor_id in self.paginated_tensors:
            del self.paginated_tensors[tensor_id]
