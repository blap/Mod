from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    Module = torch.nn.Module
else:
    Module = Any

class ShardingManagerInterface:
    """
    Interface for model sharding managers.
    """

    def shard_model(self, model: Module, num_shards: int) -> List[Module]:
        """
        Shard the model into multiple pieces.

        Args:
            model: The model to shard
            num_shards: Number of shards to create

        Returns:
            List of sharded model parts
        """
        raise NotImplementedError

    def load_sharded_model(self, shard_paths: List[str]) -> Module:
        """
        Load a sharded model from disk.

        Args:
            shard_paths: List of paths to model shards

        Returns:
            Reassembled model
        """
        raise NotImplementedError

    def get_sharding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current sharding configuration.

        Returns:
            Dictionary containing sharding statistics
        """
        raise NotImplementedError
