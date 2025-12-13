import torch
from nas_system import PerformancePredictor, LayerConfig

# Create a simple predictor
predictor = PerformancePredictor(input_dim=7, hidden_dim=256, num_layers=3)

# Print the first layer's weight shape
print("First layer weight shape:", predictor.predictor[0].weight.shape)

# Create a sample config
config = LayerConfig(layer_type="attention", hidden_size=512, num_attention_heads=8, intermediate_size=2048, layer_idx=0)

# Test prediction
result = predictor.predict_performance([config])
print("Prediction result shape:", result.shape)
print("Prediction result:", result)