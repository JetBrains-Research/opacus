# Ghost Clipping with Controller-Based Privacy Engine

This document describes the integration of Ghost Clipping with the controller-based privacy engine implementation.

## Overview

This integration combines two powerful features:

1. **Controller-Based Hooks**: Attaches privacy hooks directly to models without wrapping them
2. **Ghost Clipping**: Computes per-sample gradient norms without materializing full gradients

The result is memory-efficient differential privacy training that preserves model types and works seamlessly with complex architectures like transformers.

## Implementation

### New Files

- `opacus/grad_sample/grad_sample_controller_fast_gradient_clipping.py`: Main controller class with ghost clipping support
- `opacus/tests/grad_sample_controller_fast_gradient_clipping_test.py`: Unit tests
- `tutorials/controller_ghost_clipping.ipynb`: Tutorial demonstrating usage

### Modified Files

- `opacus/grad_sample/utils.py`: Added support for `ghost` mode in controller factory functions
- `opacus/grad_sample/__init__.py`: Exported new controller class
- `opacus/privacy_engine_gsc.py`: Extended to support ghost clipping mode
- `opacus/utils/fast_gradient_clipping_utils.py`: Made loss wrappers compatible with controllers

## Usage

### Basic Example

```python
from opacus.privacy_engine_gsc import PrivacyEngineGradSampleController

model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

privacy_engine = PrivacyEngineGradSampleController()

# Use ghost mode for memory efficiency
controller, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    grad_sample_mode="ghost",  # Enable ghost clipping
    return_controller=True,
)

# Model type is preserved
assert isinstance(model, YourModel)

# Training requires loss wrapper
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping

criterion = nn.CrossEntropyLoss(reduction="mean")
loss_fn = DPLossFastGradientClipping(
    module=controller,
    optimizer=optimizer,
    criterion=criterion,
    loss_reduction="mean",
)

# Training loop
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()  # Performs two-pass backward for ghost clipping
    optimizer.step()

# Cleanup when done
controller.cleanup()
```

### With Target Epsilon

```python
controller, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    target_epsilon=3.0,
    target_delta=1e-5,
    epochs=10,
    max_grad_norm=1.0,
    grad_sample_mode="ghost",
    return_controller=True,
)
```

## Arithmetic Operations on DPTensorFastGradientClipping

The `DPTensorFastGradientClipping` class implements full arithmetic operations, making it behave like a regular `torch.Tensor` for downstream code that needs to compose or transform losses:

```python
# Division
scaled_loss = loss / 2.0

# Multiplication
weighted_loss = 0.5 * loss  # or loss * 0.5

# Addition
combined_loss = loss1 + loss2  # or loss + scalar

# Subtraction
adjusted_loss = loss - regularization_term

# Negation
negative_loss = -loss
```

### Supported Operations

- `__truediv__` (`/`): Division by scalar
- `__mul__` and `__rmul__` (`*`): Multiplication (both left and right)
- `__add__` and `__radd__` (`+`): Addition with scalar or another DPTensor
- `__sub__` and `__rsub__` (`-`): Subtraction with scalar or another DPTensor
- `__neg__` (`-loss`): Negation
- `__repr__` and `__str__`: String representations
- `.item()`: Get scalar value

### Example: Complex Loss Composition

```python
# Main task loss
main_loss = loss_fn(output, target)

# Auxiliary losses
reg_loss = compute_regularization(model)
aux_loss = auxiliary_task_loss(model, data)

# Compose losses with arithmetic operations
total_loss = main_loss + 0.01 * reg_loss + 0.05 * aux_loss

# Ghost clipping handles the composed loss correctly
total_loss.backward()
optimizer.step()
```

**Important**: All operations preserve the `DPTensorFastGradientClipping` wrapper, ensuring ghost clipping's two-pass backward mechanism works correctly even with complex loss combinations.

## Architecture

### GradSampleControllerFastGradientClipping

The main controller class inherits the structure of `GradSampleController` but adds:

1. **NORM_SAMPLERS Registry**: Maps layer types to functions that compute gradient norms directly
2. **Two-Mode Operation**:
   - **Ghost Clipping**: For supported layers (Linear, LayerNorm), computes norms directly
   - **Fast Gradient Clipping**: For other layers, computes full gradients then norms
3. **Clipping Coefficient Computation**: `get_clipping_coef()` method for loss wrapper

### Hook Management

Hooks are attached during initialization and can be:
- Enabled/disabled with `enable_hooks()`/`disable_hooks()`
- Removed with `remove_hooks()`
- Fully cleaned up with `cleanup()`

### Two-Pass Backward

Ghost clipping requires two backward passes:

1. **First Pass**: Forward → Compute loss → Backward → Compute per-sample norms
2. **Compute Coefficients**: `coeff = min(1, max_grad_norm / norm)`
3. **Second Pass**: Forward → Compute scaled loss → Backward → Accumulate clipped gradients

This is handled automatically by the `DPTensorFastGradientClipping` wrapper.

## Benefits

### Memory Efficiency

Ghost clipping avoids storing full per-sample gradients:
- **Standard Mode**: Stores `(batch_size, *param_shape)` per parameter
- **Ghost Mode**: Stores only `(batch_size,)` norms per parameter

For large models, this can reduce memory usage by 50% or more.

### Type Preservation

Unlike wrapped approaches, the model type is preserved:

```python
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
controller, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    grad_sample_mode="ghost",
    return_controller=True,
    ...
)

# Type is preserved!
assert isinstance(model, BertForSequenceClassification)
assert type(model).__name__ == "BertForSequenceClassification"
```

### Clean State Dicts

State dicts have no `_module.` prefixes:

```python
# Save and load work seamlessly
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

## Supported Layers

Ghost clipping (direct norm computation) is supported for:
- `nn.Linear`
- `nn.LayerNorm`
- `nn.Embedding`
- Custom layers with registered `@register_norm_sampler`

Other layers automatically fall back to Fast Gradient Clipping.

## Testing

Run tests with:

```bash
python -m pytest opacus/tests/grad_sample_controller_fast_gradient_clipping_test.py -v
```

Or with unittest:

```bash
python -m unittest opacus.tests.grad_sample_controller_fast_gradient_clipping_test
```

## Performance Considerations

### When to Use Ghost Clipping

**Best for:**
- Large models (transformers, ResNets)
- Limited GPU memory
- Models with many Linear/LayerNorm layers
- When you want larger batch sizes

**Not ideal for:**
- Small models where memory isn't a concern
- Models with many custom layers
- When per-sample gradients are needed for analysis

### Memory vs Speed Tradeoff

- **Memory**: Ghost clipping uses ~50% less memory
- **Speed**: Ghost clipping is ~20-30% slower (two backward passes)

For large models, the memory savings often outweigh the speed cost.

## Comparison Matrix

| Feature | Standard PrivacyEngine | Controller + Hooks | Controller + Ghost |
|---------|------------------------|--------------------|--------------------|
| Model Wrapping | Yes | No | No |
| Type Preserved | No | Yes | Yes |
| State Dict Clean | No | Yes | Yes |
| Memory Usage | High | High | Low |
| Speed | Fast | Fast | Medium |
| Transformer Compatible | Limited | Yes | Yes |
| Best Use Case | Simple models | Complex models | Large models |

## Future Work

Potential extensions:

1. **FSDP Support**: `GradSampleControllerFastGradientClippingFSDP` for distributed training
2. **Tensor Parallel**: Support for tensor parallelism
3. **More Layers**: Extend ghost clipping support to Conv layers
4. **Adaptive Clipping**: Integrate with adaptive clipping for automatic threshold tuning

## References

- Ghost Clipping Paper: https://arxiv.org/abs/2205.09632
- Opacus Documentation: https://opacus.ai
- Tutorial: `tutorials/controller_ghost_clipping.ipynb`

## Contributing

To add ghost clipping support for a new layer type:

1. Implement norm sampler function:
```python
def compute_my_layer_norm_sample(module, activations, backprops):
    # Compute per-sample gradient norms directly
    # Return dict {param: norm_tensor}
    pass
```

2. Register it:
```python
from opacus.grad_sample.utils import register_norm_sampler

@register_norm_sampler(MyLayerType)
def compute_my_layer_norm_sample(module, activations, backprops):
    ...
```

The registration will automatically make it available to both wrapped and controller-based approaches.
