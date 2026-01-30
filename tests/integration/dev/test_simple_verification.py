"""
Simple test to verify that the sharding system is working properly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.inference_pio.common.model_sharder import create_extreme_sharding_system
from src.inference_pio.common.base_plugin_interface import ModelPluginInterface, ModelPluginMetadata, PluginType
import torch
import torch.nn as nn
from datetime import datetime


class SimpleTestModel(nn.Module):
    """Simple test model for sharding."""
    
    def __init__(self, num_layers=5, hidden_size=128):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size)
            ) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class TestPlugin(ModelPluginInterface):
    """Test plugin to verify sharding functionality."""
    
    def __init__(self):
        metadata = ModelPluginMetadata(
            name="TestPlugin",
            version="1.0.0",
            author="Test",
            description="Test plugin for sharding",
            plugin_type=PluginType.MODEL_COMPONENT,
            dependencies=[],
            compatibility={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        super().__init__(metadata)
        self._model = SimpleTestModel()
    
    def initialize(self, **kwargs) -> bool:
        return True
    
    def load_model(self, config=None) -> nn.Module:
        return self._model
    
    def infer(self, data) -> any:
        return self._model(data)
    
    def cleanup(self) -> bool:
        return True


def test_sharding_creation():
    """Test that the sharding system can be created."""
    print("Testing sharding system creation...")
    
    try:
        # Create sharding system
        sharder, loader = create_extreme_sharding_system(
            storage_path="./test_shards",
            num_shards=10
        )
        print("SUCCESS: Sharding system created successfully")
        
        # Create and shard a model
        model = SimpleTestModel(num_layers=3, hidden_size=64)
        shards = sharder.shard_model(model, num_shards=10)
        print(f"SUCCESS: Model sharded into {len(shards)} shards")
        
        # Test loading a shard
        if shards:
            test_shard = sharder.load_shard(shards[0].id)
            print(f"SUCCESS: Successfully loaded shard: {shards[0].id}")
            
            # Test unloading
            sharder.unload_shard(shards[0].id)
            print(f"SUCCESS: Successfully unloaded shard: {shards[0].id}")
        
        # Test plugin integration
        plugin = TestPlugin()
        plugin.enable_sharding(num_shards=5, storage_path="./test_plugin_shards")
        plugin.shard_model(plugin._model, num_shards=5)
        stats = plugin.get_sharding_stats()
        print(f"SUCCESS: Plugin sharding stats: {stats['total_shards']} total shards")
        
        print("\nALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"ERROR in sharding test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running sharding system verification test...\n")
    success = test_sharding_creation()
    
    if success:
        print("\nSharding system is working correctly!")
    else:
        print("\nSharding system has issues.")
        sys.exit(1)