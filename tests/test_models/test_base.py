"""
Tests for the base model system of the Flexible Model System.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from src.models.base import (
    IModel, 
    ModelAdapter, 
    FlexibleModelManager, 
    BaseModelAdapter,
    HuggingFaceAdapter,
    Qwen3VLAdapter,
    ModelLoadError,
    ModelNotFoundError,
    ModelValidationError
)


class DummyModel(nn.Module):
    """A dummy model for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)
        
    def generate(self, input_ids, max_length=10):
        # Simple mock generation
        return torch.cat([input_ids, torch.ones((input_ids.shape[0], max_length - input_ids.shape[1]))], dim=1)


class TestIModelInterface:
    """Tests for the IModel interface."""
    
    def test_i_model_abstract_methods(self):
        """Test that IModel has the required abstract methods."""
        # This should raise TypeError when trying to instantiate
        with pytest.raises(TypeError):
            IModel()
            
    def test_i_model_method_signatures(self):
        """Test that IModel defines the correct method signatures."""
        # Just checking that the abstract methods exist
        assert hasattr(IModel, '__call__')
        assert hasattr(IModel, 'forward')
        assert hasattr(IModel, 'train')
        assert hasattr(IModel, 'eval')
        assert hasattr(IModel, 'get_config')
        assert hasattr(IModel, 'get_device')
        assert hasattr(IModel, 'to_device')


class TestModelAdapter:
    """Tests for the ModelAdapter base class."""
    
    def test_model_adapter_initialization(self):
        """Test ModelAdapter initialization."""
        model = DummyModel()
        adapter = ModelAdapter(model)
        
        assert adapter.model is model
        assert adapter.config == {}
        assert adapter.tokenizer is None
        
    def test_model_adapter_with_config_and_tokenizer(self):
        """Test ModelAdapter with config and tokenizer."""
        model = DummyModel()
        config = {"param": "value"}
        tokenizer = Mock()
        
        adapter = ModelAdapter(model, config, tokenizer)
        
        assert adapter.model is model
        assert adapter.config == config
        assert adapter.tokenizer is tokenizer
        
    def test_model_adapter_call_method(self):
        """Test ModelAdapter __call__ method."""
        model = DummyModel()
        adapter = ModelAdapter(model)
        
        x = torch.randn(1, 10)
        result = adapter(x)
        expected = model(x)
        
        assert torch.allclose(result, expected)
        
    def test_model_adapter_forward_method(self):
        """Test ModelAdapter forward method."""
        model = DummyModel()
        adapter = ModelAdapter(model)
        
        x = torch.randn(1, 10)
        result = adapter.forward(x)
        expected = model(x)
        
        assert torch.allclose(result, expected)
        
    def test_model_adapter_train_method(self):
        """Test ModelAdapter train method."""
        model = DummyModel()
        adapter = ModelAdapter(model)
        
        # Initially in eval mode
        model.eval()
        assert not model.training
        
        adapter.train()
        assert model.training
        
    def test_model_adapter_eval_method(self):
        """Test ModelAdapter eval method."""
        model = DummyModel()
        adapter = ModelAdapter(model)
        
        # Initially in train mode
        model.train()
        assert model.training
        
        adapter.eval()
        assert not model.training
        
    def test_model_adapter_get_config(self):
        """Test ModelAdapter get_config method."""
        model = DummyModel()
        config = {"param": "value"}
        adapter = ModelAdapter(model, config)
        
        assert adapter.get_config() == config
        
    def test_model_adapter_get_device(self):
        """Test ModelAdapter get_device method."""
        model = DummyModel()
        adapter = ModelAdapter(model)
        
        # Initially on CPU
        assert str(adapter.get_device()) == "cpu"
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
            adapter = ModelAdapter(model)
            assert adapter.get_device().type == "cuda"
            
    def test_model_adapter_to_device(self):
        """Test ModelAdapter to_device method."""
        model = DummyModel()
        adapter = ModelAdapter(model)
        
        # Test moving to CPU
        adapter.to_device("cpu")
        assert str(adapter.get_device()) == "cpu"
        
        # Test moving to CUDA if available
        if torch.cuda.is_available():
            adapter.to_device("cuda")
            assert adapter.get_device().type == "cuda"


class TestBaseModelAdapter:
    """Tests for the BaseModelAdapter class."""
    
    def test_base_model_adapter_inheritance(self):
        """Test that BaseModelAdapter inherits from ModelAdapter."""
        model = DummyModel()
        adapter = BaseModelAdapter(model)
        
        assert isinstance(adapter, ModelAdapter)
        
    def test_base_model_adapter_load_model_not_implemented(self):
        """Test that BaseModelAdapter.load_model raises NotImplementedError."""
        adapter = BaseModelAdapter(DummyModel())
        
        with pytest.raises(NotImplementedError):
            adapter.load_model("dummy_path")


class TestHuggingFaceAdapter:
    """Tests for the HuggingFaceAdapter class."""
    
    @patch('src.models.base.AutoModel')
    @patch('src.models.base.AutoConfig')
    @patch('src.models.base.AutoTokenizer')
    def test_hugging_face_adapter_load_model(self, mock_tokenizer, mock_config, mock_model):
        """Test HuggingFaceAdapter load_model method."""
        # Mock the return values
        mock_config.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = DummyModel()
        mock_tokenizer.from_pretrained.return_value = Mock()
        
        adapter = HuggingFaceAdapter(DummyModel())
        new_adapter = adapter.load_model("dummy_path")
        
        # Verify the methods were called
        mock_config.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once()
        
        assert isinstance(new_adapter, HuggingFaceAdapter)


class TestQwen3VLAdapter:
    """Tests for the Qwen3VLAdapter class."""

    def test_qwen3vl_adapter_load_model(self):
        """Test Qwen3VLAdapter load_model method."""
        from unittest.mock import patch

        # Mock the return value
        dummy_model = DummyModel()

        # Patch at the location where the import occurs (the base_model module)
        with patch('src.models.base_model.create_model_from_pretrained') as mock_create_model:
            mock_create_model.return_value = dummy_model

            # Create a temporary adapter instance to call load_model on
            temp_adapter = Qwen3VLAdapter(None)  # Create with None initially
            new_adapter = temp_adapter.load_model("dummy_path")

            # Verify the method was called
            mock_create_model.assert_called_once()

            # Verify that the returned adapter is a Qwen3VLAdapter with the correct model
            assert isinstance(new_adapter, Qwen3VLAdapter)
            assert new_adapter.model == dummy_model


class TestFlexibleModelManager:
    """Tests for the FlexibleModelManager class."""
    
    def test_flexible_model_manager_initialization(self):
        """Test FlexibleModelManager initialization."""
        manager = FlexibleModelManager()

        assert manager.models == {}
        # Check that default adapters are registered
        assert len(manager.adapters) >= 3  # Should have at least huggingface, qwen3-vl, and default
        assert 'huggingface' in manager.adapters
        assert 'qwen3-vl' in manager.adapters
        assert 'default' in manager.adapters
        assert manager.active_model is None
        
    def test_register_adapter(self):
        """Test registering an adapter."""
        manager = FlexibleModelManager()
        adapter_class = Mock()
        
        manager.register_adapter("test_adapter", adapter_class)
        
        assert "test_adapter" in manager.adapters
        assert manager.adapters["test_adapter"] == adapter_class
        
    def test_get_registered_adapter(self):
        """Test getting a registered adapter."""
        manager = FlexibleModelManager()
        adapter_class = Mock()
        
        manager.register_adapter("test_adapter", adapter_class)
        retrieved = manager.get_registered_adapter("test_adapter")
        
        assert retrieved == adapter_class
        
    def test_get_nonexistent_adapter(self):
        """Test getting a non-existent adapter."""
        manager = FlexibleModelManager()
        
        retrieved = manager.get_registered_adapter("nonexistent")
        
        assert retrieved is None
        
    def test_load_model(self):
        """Test loading a model."""
        manager = FlexibleModelManager()
        model = DummyModel()
        config = {"param": "value"}

        # Create a mock adapter class
        class MockAdapter(BaseModelAdapter):
            def __init__(self, model, config=None, tokenizer=None):
                super().__init__(model, config, tokenizer)
                self.load_model_result = Mock()
                self.load_model_result.to_device = Mock(return_value=self.load_model_result)

            def load_model(self, model_path, config=None):
                return self.load_model_result

        # Register the mock adapter class
        manager.register_adapter("dummy", MockAdapter)

        # Load the model
        manager.load_model("test_model", "dummy_path", "dummy", config)

        # Check that the model is stored
        assert "test_model" in manager.models
        
    def test_load_model_with_invalid_adapter(self):
        """Test loading a model with an invalid adapter."""
        manager = FlexibleModelManager()
        
        with pytest.raises(ModelNotFoundError):
            manager.load_model("test_model", "dummy_path", "invalid_adapter")
            
    def test_unload_model(self):
        """Test unloading a model."""
        manager = FlexibleModelManager()
        model = DummyModel()
        adapter = ModelAdapter(model)
        
        # Add a model manually
        manager.models["test_model"] = adapter
        manager.active_model = "test_model"
        
        # Unload the model
        manager.unload_model("test_model")
        
        assert "test_model" not in manager.models
        assert manager.active_model is None
        
    def test_unload_nonexistent_model(self):
        """Test unloading a non-existent model."""
        manager = FlexibleModelManager()
        
        # Should not raise an error
        manager.unload_model("nonexistent_model")
        
    def test_switch_model(self):
        """Test switching between models."""
        manager = FlexibleModelManager()
        model1 = DummyModel()
        model2 = DummyModel()
        adapter1 = ModelAdapter(model1)
        adapter2 = ModelAdapter(model2)
        
        # Add models manually
        manager.models["model1"] = adapter1
        manager.models["model2"] = adapter2
        manager.active_model = "model1"
        
        # Switch to model2
        manager.switch_model("model2")
        
        assert manager.active_model == "model2"
        
    def test_switch_to_nonexistent_model(self):
        """Test switching to a non-existent model."""
        manager = FlexibleModelManager()
        
        with pytest.raises(ModelNotFoundError):
            manager.switch_model("nonexistent_model")
            
    def test_get_active_model(self):
        """Test getting the active model."""
        manager = FlexibleModelManager()
        model = DummyModel()
        adapter = ModelAdapter(model)
        
        # Add a model and make it active
        manager.models["active_model"] = adapter
        manager.active_model = "active_model"
        
        active = manager.get_active_model()
        
        assert active is adapter
        
    def test_get_active_model_none(self):
        """Test getting the active model when none is set."""
        manager = FlexibleModelManager()
        
        active = manager.get_active_model()
        
        assert active is None
        
    def test_list_models(self):
        """Test listing models."""
        manager = FlexibleModelManager()
        model = DummyModel()
        adapter = ModelAdapter(model)
        
        # Add some models
        manager.models["model1"] = adapter
        manager.models["model2"] = adapter
        
        model_list = manager.list_models()
        
        assert "model1" in model_list
        assert "model2" in model_list
        assert len(model_list) == 2
        
    def test_validate_config_valid(self):
        """Test validating a valid config."""
        manager = FlexibleModelManager()
        config = {
            "model_path": "test_path",
            "adapter_type": "test_adapter",
            "device": "cpu"
        }
        
        # Should not raise an exception
        manager.validate_config(config)
        
    def test_validate_config_missing_required(self):
        """Test validating a config with missing required fields."""
        manager = FlexibleModelManager()
        config = {
            "model_path": "test_path"
            # Missing adapter_type
        }
        
        with pytest.raises(ModelValidationError):
            manager.validate_config(config)
            
    def test_invoke_active_model(self):
        """Test invoking the active model."""
        manager = FlexibleModelManager()
        model = DummyModel()
        adapter = ModelAdapter(model)
        
        # Add a model and make it active
        manager.models["active_model"] = adapter
        manager.active_model = "active_model"
        
        x = torch.randn(1, 10)
        result = manager.invoke_active_model(x)
        
        expected = model(x)
        assert torch.allclose(result, expected)
        
    def test_invoke_active_model_no_active(self):
        """Test invoking when no model is active."""
        manager = FlexibleModelManager()
        
        with pytest.raises(ModelNotFoundError):
            manager.invoke_active_model(torch.randn(1, 10))


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])