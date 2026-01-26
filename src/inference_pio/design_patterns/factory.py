"""
Factory Pattern Implementation for Inference-PIO

This module implements the Factory pattern for creating plugins, models, and optimization strategies.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Union

from ..common.standard_plugin_interface import ModelPluginInterface
# Import plugins inside factory methods to avoid circular imports
from .strategy import OptimizationStrategy, MemoryOptimizationStrategy, ComputeOptimizationStrategy
from .adapter import ModelAdapter, GLM47ModelAdapter, Qwen34BInstruct2507ModelAdapter, Qwen3Coder30BModelAdapter, Qwen3VL2BModelAdapter


logger = logging.getLogger(__name__)


class PluginFactory(ABC):
    """
    Abstract factory for creating model plugins.
    """
    
    @abstractmethod
    def create_plugin(self) -> ModelPluginInterface:
        """
        Create a model plugin instance.
        
        Returns:
            ModelPluginInterface: Created plugin instance
        """
        pass


class GLM47PluginFactory(PluginFactory):
    """
    Factory for creating GLM-4.7 plugins.
    """

    def create_plugin(self) -> ModelPluginInterface:
        """
        Create a GLM-4.7 plugin instance.

        Returns:
            GLM_4_7_Plugin: Created plugin instance
        """
        from ..models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
        return create_glm_4_7_flash_plugin()


class Qwen34BInstruct2507PluginFactory(PluginFactory):
    """
    Factory for creating Qwen3-4B-Instruct-2507 plugins.
    """

    def create_plugin(self) -> ModelPluginInterface:
        """
        Create a Qwen3-4B-Instruct-2507 plugin instance.

        Returns:
            Qwen3_4B_Instruct_2507_Plugin: Created plugin instance
        """
        from ..models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
        return create_qwen3_4b_instruct_2507_plugin()


class Qwen3Coder30BPluginFactory(PluginFactory):
    """
    Factory for creating Qwen3-Coder-30B plugins.
    """

    def create_plugin(self) -> ModelPluginInterface:
        """
        Create a Qwen3-Coder-30B plugin instance.

        Returns:
            Qwen3_Coder_30B_Plugin: Created plugin instance
        """
        from ..models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
        return create_qwen3_coder_30b_plugin()


class Qwen3VL2BPluginFactory(PluginFactory):
    """
    Factory for creating Qwen3-VL-2B plugins.
    """

    def create_plugin(self) -> ModelPluginInterface:
        """
        Create a Qwen3-VL-2B plugin instance.

        Returns:
            Qwen3_VL_2B_Instruct_Plugin: Created plugin instance
        """
        from ..models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
        return create_qwen3_vl_2b_instruct_plugin()


class OptimizationStrategyFactory(ABC):
    """
    Abstract factory for creating optimization strategies.
    """
    
    @abstractmethod
    def create_strategy(self) -> OptimizationStrategy:
        """
        Create an optimization strategy instance.
        
        Returns:
            OptimizationStrategy: Created strategy instance
        """
        pass


class MemoryOptimizationStrategyFactory(OptimizationStrategyFactory):
    """
    Factory for creating memory optimization strategies.
    """
    
    def create_strategy(self) -> OptimizationStrategy:
        """
        Create a memory optimization strategy instance.
        
        Returns:
            MemoryOptimizationStrategy: Created strategy instance
        """
        return MemoryOptimizationStrategy()


class ComputeOptimizationStrategyFactory(OptimizationStrategyFactory):
    """
    Factory for creating compute optimization strategies.
    """
    
    def create_strategy(self) -> OptimizationStrategy:
        """
        Create a compute optimization strategy instance.
        
        Returns:
            ComputeOptimizationStrategy: Created strategy instance
        """
        return ComputeOptimizationStrategy()


class ModelAdapterFactory(ABC):
    """
    Abstract factory for creating model adapters.
    """
    
    @abstractmethod
    def create_adapter(self, model: Any) -> 'ModelAdapter':
        """
        Create a model adapter instance.
        
        Args:
            model: The model to adapt
            
        Returns:
            ModelAdapter: Created adapter instance
        """
        pass


class GLM47ModelAdapterFactory(ModelAdapterFactory):
    """
    Factory for creating GLM-4.7 model adapters.
    """
    
    def create_adapter(self, model: Any) -> ModelAdapter:
        """
        Create a GLM-4.7 model adapter instance.
        
        Args:
            model: The GLM-4.7 model to adapt
            
        Returns:
            GLM47ModelAdapter: Created adapter instance
        """
        return GLM47ModelAdapter(model)


class Qwen34BInstruct2507ModelAdapterFactory(ModelAdapterFactory):
    """
    Factory for creating Qwen3-4B-Instruct-2507 model adapters.
    """
    
    def create_adapter(self, model: Any) -> ModelAdapter:
        """
        Create a Qwen3-4B-Instruct-2507 model adapter instance.
        
        Args:
            model: The Qwen3-4B-Instruct-2507 model to adapt
            
        Returns:
            Qwen34BInstruct2507ModelAdapter: Created adapter instance
        """
        return Qwen34BInstruct2507ModelAdapter(model)


class Qwen3Coder30BModelAdapterFactory(ModelAdapterFactory):
    """
    Factory for creating Qwen3-Coder-30B model adapters.
    """
    
    def create_adapter(self, model: Any) -> ModelAdapter:
        """
        Create a Qwen3-Coder-30B model adapter instance.
        
        Args:
            model: The Qwen3-Coder-30B model to adapt
            
        Returns:
            Qwen3Coder30BModelAdapter: Created adapter instance
        """
        return Qwen3Coder30BModelAdapter(model)


class Qwen3VL2BModelAdapterFactory(ModelAdapterFactory):
    """
    Factory for creating Qwen3-VL-2B model adapters.
    """
    
    def create_adapter(self, model: Any) -> ModelAdapter:
        """
        Create a Qwen3-VL-2B model adapter instance.
        
        Args:
            model: The Qwen3-VL-2B model to adapt
            
        Returns:
            Qwen3VL2BModelAdapter: Created adapter instance
        """
        return Qwen3VL2BModelAdapter(model)


class PluginFactoryProvider:
    """
    Provider class that manages and provides access to different plugin factories.
    """
    
    _factories: Dict[str, Type[PluginFactory]] = {
        'glm_4_7_flash': GLM47PluginFactory,
        'qwen3_4b_instruct_2507': Qwen34BInstruct2507PluginFactory,
        'qwen3_coder_30b': Qwen3Coder30BPluginFactory,
        'qwen3_vl_2b': Qwen3VL2BPluginFactory,
    }
    
    @classmethod
    def get_factory(cls, model_type: str) -> PluginFactory:
        """
        Get a plugin factory for the specified model type.
        
        Args:
            model_type: Type of model ('glm_4_7', 'qwen3_4b_instruct_2507', etc.)
            
        Returns:
            PluginFactory: Requested factory instance
        """
        factory_class = cls._factories.get(model_type.lower())
        if factory_class is None:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return factory_class()
    
    @classmethod
    def register_factory(cls, model_type: str, factory_class: Type[PluginFactory]):
        """
        Register a new plugin factory.
        
        Args:
            model_type: Type of model
            factory_class: Factory class to register
        """
        cls._factories[model_type.lower()] = factory_class
    
    @classmethod
    def create_plugin(cls, model_type: str) -> ModelPluginInterface:
        """
        Create a plugin using the appropriate factory.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            ModelPluginInterface: Created plugin instance
        """
        factory = cls.get_factory(model_type)
        return factory.create_plugin()


class OptimizationStrategyFactoryProvider:
    """
    Provider class that manages and provides access to different optimization strategy factories.
    """
    
    _factories: Dict[str, Type[OptimizationStrategyFactory]] = {
        'memory': MemoryOptimizationStrategyFactory,
        'compute': ComputeOptimizationStrategyFactory,
    }
    
    @classmethod
    def get_factory(cls, strategy_type: str) -> OptimizationStrategyFactory:
        """
        Get an optimization strategy factory for the specified strategy type.
        
        Args:
            strategy_type: Type of strategy ('memory', 'compute', etc.)
            
        Returns:
            OptimizationStrategyFactory: Requested factory instance
        """
        factory_class = cls._factories.get(strategy_type.lower())
        if factory_class is None:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return factory_class()
    
    @classmethod
    def register_factory(cls, strategy_type: str, factory_class: Type[OptimizationStrategyFactory]):
        """
        Register a new optimization strategy factory.
        
        Args:
            strategy_type: Type of strategy
            factory_class: Factory class to register
        """
        cls._factories[strategy_type.lower()] = factory_class
    
    @classmethod
    def create_strategy(cls, strategy_type: str) -> OptimizationStrategy:
        """
        Create an optimization strategy using the appropriate factory.
        
        Args:
            strategy_type: Type of strategy to create
            
        Returns:
            OptimizationStrategy: Created strategy instance
        """
        factory = cls.get_factory(strategy_type)
        return factory.create_strategy()


class ModelAdapterFactoryProvider:
    """
    Provider class that manages and provides access to different model adapter factories.
    """
    
    _factories: Dict[str, Type[ModelAdapterFactory]] = {
        'glm_4_7_flash': GLM47ModelAdapterFactory,
        'qwen3_4b_instruct_2507': Qwen34BInstruct2507ModelAdapterFactory,
        'qwen3_coder_30b': Qwen3Coder30BModelAdapterFactory,
        'qwen3_vl_2b': Qwen3VL2BModelAdapterFactory,
    }
    
    @classmethod
    def get_factory(cls, model_type: str) -> ModelAdapterFactory:
        """
        Get a model adapter factory for the specified model type.
        
        Args:
            model_type: Type of model ('glm_4_7', 'qwen3_4b_instruct_2507', etc.)
            
        Returns:
            ModelAdapterFactory: Requested factory instance
        """
        factory_class = cls._factories.get(model_type.lower())
        if factory_class is None:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return factory_class()
    
    @classmethod
    def register_factory(cls, model_type: str, factory_class: Type[ModelAdapterFactory]):
        """
        Register a new model adapter factory.
        
        Args:
            model_type: Type of model
            factory_class: Factory class to register
        """
        cls._factories[model_type.lower()] = factory_class
    
    @classmethod
    def create_adapter(cls, model_type: str, model: Any) -> ModelAdapter:
        """
        Create a model adapter using the appropriate factory.
        
        Args:
            model_type: Type of model to create adapter for
            model: The model to adapt
            
        Returns:
            ModelAdapter: Created adapter instance
        """
        factory = cls.get_factory(model_type)
        return factory.create_adapter(model)


__all__ = [
    'PluginFactory',
    'GLM47PluginFactory',
    'Qwen34BInstruct2507PluginFactory',
    'Qwen3Coder30BPluginFactory',
    'Qwen3VL2BPluginFactory',
    'OptimizationStrategyFactory',
    'MemoryOptimizationStrategyFactory',
    'ComputeOptimizationStrategyFactory',
    'ModelAdapterFactory',
    'GLM47ModelAdapterFactory',
    'Qwen34BInstruct2507ModelAdapterFactory',
    'Qwen3Coder30BModelAdapterFactory',
    'Qwen3VL2BModelAdapterFactory',
    'PluginFactoryProvider',
    'OptimizationStrategyFactoryProvider',
    'ModelAdapterFactoryProvider'
]