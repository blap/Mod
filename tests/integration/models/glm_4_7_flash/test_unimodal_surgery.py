import torch
import torch.nn as nn
from src.inference_pio.common.unimodal_model_surgery import UnimodalModelSurgerySystem
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


print('Testing unimodal model surgery system...')

# Create a simple test model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(32)
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.linear(x)
        return x

model = TestModel()
print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')

# Test the unimodal model surgery system
surgery_system = UnimodalModelSurgerySystem()
print('Surgery system created')

# Identify removable components
components = surgery_system.identify_removable_components(model)
print(f'Identified {len(components)} removable components')
for comp in components:
    print(f'  - {comp.name}: {comp.type} (priority: {comp.priority})')

# Perform surgery
modified_model = surgery_system.perform_unimodal_surgery(model)
print('Surgery completed successfully')

# Test that the modified model still works
test_input = torch.randint(0, 100, (2, 5))
output = modified_model(test_input)
print('Modified model output shape:', output.shape)
print('Success: Unimodal model surgery system working correctly!')