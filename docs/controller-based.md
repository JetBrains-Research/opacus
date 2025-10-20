# GradSampleController-Based Privacy Engine

## Overview

`PrivacyEngineGradSampleController` is an alternative implementation of Opacus's privacy engine that attaches hooks directly to model modules **without wrapping them** in a `GradSampleModule`. This approach provides better compatibility with transformer models and other architectures that have complex module introspection or custom attribute access patterns.

## Motivation

The standard `PrivacyEngine` wraps models in a `GradSampleModule`, which can cause issues with:

1. **Transformer models**: HuggingFace transformers and other libraries that use complex `__getattr__` logic
2. **State dict compatibility**: Wrapped models have `_module.` prefixes that complicate checkpoint loading
3. **Type checking**: `isinstance()` checks fail because the model is wrapped
4. **Direct attribute access**: Accessing submodules requires going through the wrapper

### Issues with GradSampleModule Wrapping

```python
# Standard PrivacyEngine wraps the model
model = BertModel(...)
privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private(...)

# Issues:
isinstance(model, BertModel)  # False! It's a GradSampleModule
model.encoder  # Works via __getattr__ forwarding, but can be fragile
model.state_dict()  # May have "_module." prefixes
```

### GradSampleController-Based Approach

```python
# PrivacyEngineGradSampleController does NOT wrap the model
model = BertModel(...)
privacy_engine = PrivacyEngineGradSampleController()
model, optimizer, dataloader = privacy_engine.make_private(...)

# Benefits:
isinstance(model, BertModel)  # True! Model is unchanged
model.encoder  # Direct access, no forwarding
model.state_dict()  # Clean keys, no prefixes
```

## Architecture

### Components

1. **GradSampleController** (`opacus/grad_sample/grad_sample_controller.py`)
   - Manages the lifecycle of forward and backward hooks
   - Attaches hooks directly to model submodules
   - Computes per-sample gradients via hooks
   - Stores state separately from the model

2. **PrivacyEngineGradSampleController** (`opacus/privacy_engine_gsc.py`)
   - Drop-in replacement for `PrivacyEngine`
   - Creates and manages `GradSampleController`
   - Same API as standard `PrivacyEngine`
   - Returns unwrapped models

### How It Works

Instead of wrapping the model in a GradSampleModule:

```python
# Standard approach (wraps model)
wrapped_model = GradSampleModule(model)
```

The controller-based approach attaches hooks directly:

```python
# Controller-based approach (no wrapping)
grad_sample_controller = GradSampleController(model)
# model remains unchanged, hooks are attached
```

The `GradSampleController`:
- Iterates over model submodules
- Registers `forward_hook` and `full_backward_hook` on each submodule
- Captures activations in forward pass
- Computes per-sample gradients in backward pass
- Stores `grad_sample` attribute directly on parameters using `setattr`

## Usage

### Basic Example

```python
import torch
from opacus.privacy_engine_gsc import PrivacyEngineGradSampleController

# Create your model
model = MyTransformerModel()
optimizer = torch.optim.Adam(model.parameters())
dataloader = ...

# Initialize hook-based privacy engine
privacy_engine = PrivacyEngineGradSampleController()

# Make private (model is NOT wrapped!)
model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# Train normally
for data, target in dataloader:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Clean up when done
privacy_engine.cleanup()
```

### With Epsilon Budget

```python
privacy_engine = PrivacyEngineGradSampleController()

model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    target_epsilon=3.0,
    target_delta=1e-5,
    epochs=10,
    max_grad_norm=1.0,
)
```

### Checkpoint Saving and Loading

```python
# Save checkpoint
privacy_engine.save_checkpoint(
    path="checkpoint.pt",
    module=model,
    optimizer=optimizer,
)

# Load checkpoint
privacy_engine.load_checkpoint(
    path="checkpoint.pt",
    module=model,
    optimizer=optimizer,
)
```

Note: State dicts are clean (no `_module.` prefixes) because the model is not wrapped!

## API Reference

### PrivacyEngineGradSampleController

#### `__init__(accountant="prv", secure_mode=False)`

Initialize the privacy engine.

**Parameters:**
- `accountant` (str): Accounting mechanism ("rdp", "gdp", or "prv")
- `secure_mode` (bool): Use secure random number generator (requires torchcsprng)

#### `make_private(...)`

Make a model, optimizer, and dataloader private.

**Parameters:**
- `module` (nn.Module): The model to make private
- `optimizer` (Optimizer): The optimizer
- `data_loader` (DataLoader): The data loader
- `noise_multiplier` (float): Noise multiplier for DP
- `max_grad_norm` (float): Maximum gradient norm for clipping
- `batch_first` (bool): Whether batch dimension is first
- `loss_reduction` (str): "mean" or "sum"
- `poisson_sampling` (bool): Use Poisson sampling
- `clipping` (str): Clipping strategy ("flat", "per_layer", "adaptive")
- `grad_sample_mode` (str): Mode for computing gradients
- `force_functorch` (bool): Force use of functorch for all modules

**Returns:**
- Tuple of (model, optimizer, dataloader)
- **Note**: Model is NOT wrapped - it's the original model with hooks attached

#### `cleanup()`

Remove all hooks and attributes from the model. Call this when done with DP training.

### GradSampleController

#### `__init__(model, batch_first=True, loss_reduction="mean", ...)`

Create a hook controller for a model.

**Parameters:**
- `model` (nn.Module): The model to attach hooks to
- `batch_first` (bool): Whether batch dimension is first
- `loss_reduction` (str): "mean" or "sum"
- `force_functorch` (bool): Force functorch for all modules
- `grad_samplers` (dict): Dictionary of grad sampler functions

#### `remove_hooks()`

Remove all hooks from the model.

#### `cleanup()`

Remove hooks and clean up parameter attributes.

## Comparison with Standard PrivacyEngine

| Feature | PrivacyEngine | PrivacyEngineGradSampleController |
|---------|--------------|------------------------|
| Model wrapping | Yes (GradSampleModule) | No |
| Type preservation | No | Yes |
| State dict prefixes | `_module.` prefix | Clean, no prefix |
| Direct attribute access | Via forwarding | Direct |
| Transformer compatibility | Can have issues | Better |
| API compatibility | Standard | Same as standard |
| Performance | Baseline | Similar |
| Privacy guarantees | Equivalent | Equivalent |

## Advantages

1. **No model wrapping**: Model remains its original type
2. **Type checking works**: `isinstance(model, MyModel)` returns `True`
3. **Direct attribute access**: No `__getattr__` forwarding needed
4. **Clean state dicts**: No `_module.` prefixes to handle
5. **Better transformer compatibility**: Works with HuggingFace transformers
6. **Simpler debugging**: Model structure unchanged

## Disadvantages

1. **Hook management**: Must explicitly call `cleanup()` to remove hooks
2. **Newer approach**: Less battle-tested than standard PrivacyEngine
3. **Parameter pollution**: Adds attributes (`grad_sample`, etc.) directly to parameters

## Migration Guide

### From Standard PrivacyEngine

Minimal changes needed:

```python
# Before
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()

# After
from opacus.privacy_engine_gsc import PrivacyEngineGradSampleController

privacy_engine = PrivacyEngineGradSampleController()

# Rest of the code remains the same!
model, optimizer, dataloader = privacy_engine.make_private(...)
```

### Additional Cleanup Step

With controller-based approach, remember to cleanup:

```python
# At the end of training
privacy_engine.cleanup()  # Removes hooks and attributes
```

## Limitations

1. **Not compatible with ghost clipping**: Currently only supports standard hooks and functorch
2. **Requires explicit cleanup**: Must call `cleanup()` to remove hooks
3. **Python 3.8+**: Requires modern Python for proper hook support

## Examples

See `examples/gsc_example.py` for a complete example with a transformer model.

## Testing

Run tests with:

```bash
python -m pytest opacus/tests/privacy_engine_gsc_test.py -v
```

Or run the example:

```bash
python examples/gsc_example.py
```

## Implementation Details

### Parameter Attributes

The hook controller adds these attributes to parameters:
- `grad_sample`: Per-sample gradients (batch_size, *param.shape)
- `_forward_counter`: Tracks forward passes (for RNNs)
- `_current_grad_sample`: Accumulates gradients during backward pass
- `_norm_sample`: Per-sample gradient norms (for fast gradient clipping)

### Hook Lifecycle

1. **Initialization**: `GradSampleController` registers hooks on all trainable submodules
2. **Forward pass**: `capture_activations_hook` stores input activations
3. **Backward pass**: `capture_backprops_hook` computes per-sample gradients
4. **Optimizer step**: DPOptimizer clips and adds noise using `grad_sample`
5. **Cleanup**: `cleanup()` removes hooks and parameter attributes

### Thread Safety

The controller-based approach is compatible with PyTorch's autograd threading model. However, be aware that hooks are called in the order they were registered, which may matter for certain advanced use cases.

## Future Work

- [ ] Support for ghost clipping mode
- [ ] Integration with FSDP and DDP
- [ ] Performance optimizations for large models
- [ ] Automatic hook cleanup on model deletion

## Contributing

Contributions are welcome! Please ensure:
1. Tests pass for both standard and GradSampleController-based engines
2. Documentation is updated
3. Examples demonstrate new features

## License

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the Apache License, Version 2.0.
