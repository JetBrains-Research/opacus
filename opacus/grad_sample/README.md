# Grad Samples

Computing per sample gradients is an integral part of Opacus framework. We strive to provide out-of-the-box support for
wide range of models, while keeping computations efficient.

We currently provide two independent approaches for computing per sample gradients: hooks-based ``GradSampleModule``
(stable implementation, exists since the very first version of Opacus) and ``GradSampleModuleExpandedWeights``
(based on a beta functionality available in PyTorch 1.12).

Each of the two implementations comes with its own set of limitations, and we leave the choice up to the client
which one to use.

``GradSampleModuleExpandedWeights`` is currently in early beta and can produce unexpected errors, but potentially
improves upon ``GradSampleModule`` on performance and functionality.

**TL;DR:** If you want stable implementation, use ``GradSampleModule`` (`grad_sample_mode="hooks"`).
If you want to experiment with the new functionality, you have two options. Try
``GradSampleModuleExpandedWeights``(`grad_sample_mode="ew"`) for better performance and `grad_sample_mode=functorch`
if your model is not supported by ``GradSampleModule``.

Please switch back to ``GradSampleModule``(`grad_sample_mode="hooks"`) if you encounter strange errors or unexpected behaviour.
We'd also appreciate it if you report these to us

## Hooks-based approach
- Model wrapping class: ``opacus.grad_sample.grad_sample_module.GradSampleModule``
- Keyword argument for ``PrivacyEngine.make_private()``: `grad_sample_mode="hooks"`

Computes per-sample gradients for a model using backward hooks. It requires custom grad sampler methods for every
trainable layer in the model. We provide such methods for most popular PyTorch layers. Additionally, client can
provide their own grad sampler for any new unsupported layer (see [tutorial](https://github.com/pytorch/opacus/blob/main/tutorials/guide_to_grad_sampler.ipynb))

## Functorch approach
- Model wrapping class: ``opacus.grad_sample.grad_sample_module.GradSampleModule (force_functorch=True)``
- Keyword argument for ``PrivacyEngine.make_private()``: `grad_sample_mode="functorch"`

[functorch](https://pytorch.org/functorch/stable/) is JAX-like composable function transforms for PyTorch.
With functorch we can compute per-sample-gradients efficiently by using function transforms. With the efficient
parallelization provided by `vmap`, we can obtain per-sample gradients for any function function (i.e. any model) by
doing essentially `vmap(grad(f(x)))`.

Our experiments show, that `vmap` computations in most cases are as fast as manually written grad samplers used in
hooks-based approach.

With the current implementation `GradSampleModule` will use manual grad samplers for known modules (i.e. maintain the
old behaviour for all previously supported models) and will only use functorch for unknown modules.

With `force_functorch=True` passed to the constructor `GradSampleModule` will rely exclusively on functorch.

## ExpandedWeights approach
- Model wrapping class: ``opacus.grad_sample.gsm_exp_weights.GradSampleModuleExpandedWeights``
- Keyword argument for ``PrivacyEngine.make_private()``: `grad_sample_mode="ew"`

Computes per-sample gradients for a model using core functionality available in PyTorch 1.12+. Unlike hooks-based
grad sampler, which works on a module level, ExpandedWeights work on the function level, i.e. if your layer is not
explicitly supported, but only uses known operations, ExpandedWeights will support it out of the box.

At the time of writing, the coverage for custom grad samplers between ``GradSampleModule`` and ``GradSampleModuleExpandedWeights``
is roughly the same.

## Non-wrapping mode (Hooks without wrapper)

**New in version X.X**: You can now use hooks-based grad sample computation **without wrapping** your model.

### Why use non-wrapping mode?

By default, Opacus wraps your model in a `GradSampleModule` wrapper, which can cause compatibility issues with some models:
- **Type checking fails**: `isinstance(model, MyModel)` returns `False` after wrapping
- **State dict keys change**: Wrapped models add `_module.` prefix to all parameter names
- **Attribute access issues**: Some models with custom `__getattr__` (e.g., HuggingFace Transformers) may not work correctly
- **Introspection breaks**: Any code that inspects the model structure sees the wrapper, not your original model

Non-wrapping mode solves these issues by attaching hooks directly to your model's parameters without changing the model object itself.

### Usage

Set `wrap_model=False` in `PrivacyEngine.make_private()`:

```python
from opacus import PrivacyEngine

# Your model - untouched by Opacus
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

privacy_engine = PrivacyEngine()
hooks, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    wrap_model=False,  # Enable non-wrapping mode
)
# hooks is a GradSampleHooks object for cleanup
# model is your original model - use it directly!
```

### Important: Use Your Model Directly

In non-wrapping mode, your **original model is unchanged**. You already have it - just use it!

```python
# Use your model normally - you already have it
output = model(input)                     # Forward pass
state_dict = model.state_dict()          # Get state dict
model.train()                            # Switch to train mode
torch.save(model.state_dict(), 'model.pt')  # Save checkpoint
```

The `hooks` object (returned by `make_private`) is only for cleanup. It does **not** support `nn.Module` methods like `.state_dict()` or `forward()` - use your original model for those.

### Cleanup

When you're done training, clean up using the hooks object:

```python
# Clean up hooks when done
hooks.cleanup()
```

This removes all hooks and monkeypatched attributes from your model parameters.

### Limitations

- **ExpandedWeights not supported**: The `grad_sample_mode="ew"` mode requires overriding `.forward()` and only works with wrapping
- **Manual cleanup required**: Unlike wrapped mode, you need to explicitly clean up hooks when switching datasets or ending training
- **Hooks stored on model**: The hooks object is stored at `model._opacus_hooks` which could potentially conflict with other libraries

### When to use non-wrapping mode

Use `wrap_model=False` when:
- Working with HuggingFace Transformers or other models with complex `__getattr__` logic
- You need `isinstance()` checks to work correctly (e.g., for model-specific optimizations)
- You want clean state dicts without `_module.` prefixes
- Your pipeline relies on model type introspection

Use default wrapped mode (`wrap_model=True`) when:
- You have simple models without complex introspection needs
- You want automatic cleanup (wrapper is discarded when model goes out of scope)
- You don't need the benefits listed above

See the [non-wrapping mode tutorial](../tutorials/non_wrapping_mode.ipynb) for a complete example.

## Comparative analysis

Please note that these are known limitations and we plan to improve Expanded Weights and bridge the gap in feature completeness


| Feature                      | Hooks (Wrapped)                 | Expanded Weights | Functorch    |
|:----------------------------:|:-------------------------------:|:----------------:|:------------:|
| Required PyTorch version     | 1.8+                            | 1.13+            | 1.12 (to be updated) |
| Development status           | Underlying mechanism deprecated | Beta             | Beta         |
| **Model wrapping**           | **Yes (GradSampleModule)**      | **Yes**          | **Yes**      |
| **Non-wrapping mode**        | **✅ Supported (`wrap_model=False`)** | **❌ Not supported** | **✅ Supported (`wrap_model=False`)** |
| Runtime Performance†          | baseline                       | ✅ ~25% faster  | 🟨 0-50% slower |
| Any DP-allowed†† layers       | Not supported                   | Not supported   | ✅ Supported |
| Most popular nn.* layers     | ✅ Supported                    | ✅ Supported    | ✅ Supported  |
| torchscripted models         | Not supported                   | ✅ Supported    | Not supported |
| Client-provided grad sampler | ✅ Supported                    | Not supported   | ✅ Not needed |
| `batch_first=False`          | ✅ Supported                    | Not supported   | ✅ Supported  |
| Recurrent networks           | ✅ Supported                    | Not supported   | ✅ Supported  |
| Padding `same` in Conv       | ✅ Supported                    | Not supported   | ✅ Supported  |
| Empty poisson batches        | ✅ Supported                    | Not supported   | Not supported  |

† Note, that performance differences are unstable and can vary a lot depending on the exact model and batch size.
Numbers above are averaged over benchmarks with small models consisting of convolutional and linear layers.
Note, that performance differences are only observed on GPU training, CPU performance seem to be almost identical
for all approaches.

†† Layers that produce joint computations on batch samples (e.g. BatchNorm) are not allowed under any approach
