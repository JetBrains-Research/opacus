# Implementation Notes: Tensor Parallelism Controller

## Answers to Key Questions

### 1. Does the standard PrivacyEngine auto-detect ghost_tp or check model for TP?

**Answer: NO auto-detection for TP/FSDP grad_sample_mode.**

The standard `PrivacyEngine` in `privacy_engine.py` does the following:

#### What it DOES auto-detect:
```python
# Line 392 in privacy_engine.py
distributed = isinstance(module, (DPDDP, DDP, FSDPModule))
```
It detects if the model is wrapped in **DDP** or **FSDP** for optimizer selection, but NOT for grad_sample_mode.

#### What it DOES NOT auto-detect:
- **Tensor Parallelism** (DTensor-based): Not detected
- **Ghost clipping mode**: User must explicitly pass `grad_sample_mode="ghost"` or `"ghost_fsdp"`

#### How ghost mode is triggered:
```python
# Line 432 in privacy_engine.py
if "ghost" in grad_sample_mode:
    criterion = self._prepare_criterion(...)
    return module, optimizer, criterion, data_loader
```

The user must explicitly specify:
- `grad_sample_mode="ghost"` for ghost clipping
- `grad_sample_mode="ghost_fsdp"` for FSDP + ghost clipping
- ❌ There is NO `"ghost_tp"` mode in the standard PrivacyEngine

### Our Implementation Decision

For **GradSampleControllerTP**, we follow the same pattern:

1. **No auto-detection**: User must explicitly pass `grad_sample_mode="tp"`
2. **No ghost clipping**: We only support standard gradient computation (no ghost mode)
3. **Consistent API**: Matches existing Opacus patterns

#### Usage Example:
```python
privacy_engine = PrivacyEngineGradSampleController()

controller, optimizer, dataloader = privacy_engine.make_private(
    module=model,  # Model with TP already applied via parallelize_module
    optimizer=optimizer,
    data_loader=dataloader,
    grad_sample_mode="tp",  # ← User must explicitly specify TP
    ...
)
```

### Why No Auto-Detection?

1. **TP is applied externally**: Tensor parallelism is set up via `parallelize_module()` before calling `make_private()`. The privacy engine doesn't control TP setup.

2. **DTensor detection is non-trivial**: Detecting DTensors requires inspecting all parameters, checking their types and placements - expensive and error-prone.

3. **Explicit is better**: User knows if they set up TP, so explicit mode selection is clearer than magic auto-detection.

4. **Consistent with existing patterns**: Standard PrivacyEngine requires explicit `grad_sample_mode` for ghost/FSDP modes.

---

### 2. Should we follow multigpu_*.py test patterns?

**Answer: YES! And we have.**

#### Existing Pattern in opacus/tests/multigpu_*.py

Looking at `multigpu_gradcheck_test.py`:

```python
def setup(rank, world_size):
    """Setup distributed environment."""
    if sys.platform == "win32":
        raise ValueError("Windows not supported")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.distributed.init_process_group(
        init_method="env://",
        backend="nccl",
    )

def cleanup():
    """Cleanup distributed environment."""
    dist.destroy_process_group()

def run_some_test(rank, world_size):
    """Test function executed on each rank."""
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # ... test logic ...

    cleanup()
    return result

class SomeTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires 2+ GPUs")
    def test_something(self):
        world_size = 2
        mp.spawn(run_some_test, args=(world_size,), nprocs=world_size, join=True)
```

#### Our Updated Implementation

We created **`multigpu_controller_tp_test.py`** following this exact pattern:

**Key changes from original `grad_sample_controller_tp_test.py`:**

1. ✅ **Renamed file**: `grad_sample_controller_tp_test.py` → `multigpu_controller_tp_test.py`
2. ✅ **Added `setup()` function**: Centralizes distributed initialization with `init_method="env://"`
3. ✅ **Added `cleanup()` function**: Properly tears down distributed process group
4. ✅ **Uses env:// init method**: Sets `RANK` and `WORLD_SIZE` env vars for cleaner init
5. ✅ **Added Windows check**: Raises error on Windows platform
6. ✅ **Consistent test structure**: All test functions call `setup()` then `cleanup()`
7. ✅ **Same file organization**: Follows multigpu_*.py conventions

#### Benefits of This Pattern

1. **Consistent with codebase**: Matches existing multi-GPU test patterns
2. **Cleaner distributed init**: Uses `init_method="env://"` like other tests
3. **Better resource management**: Explicit `cleanup()` ensures proper teardown
4. **Discoverable**: File naming convention `multigpu_*` makes it clear these need GPUs
5. **Platform safety**: Explicit Windows check prevents confusing failures

#### File Locations

- **New (recommended)**: `opacus/tests/multigpu_controller_tp_test.py`
- **Old (can delete)**: `opacus/tests/grad_sample_controller_tp_test.py`

---

## Summary

### Question 1: Auto-Detection
- ❌ Standard PrivacyEngine does NOT auto-detect TP or select ghost modes
- ✅ User must explicitly pass `grad_sample_mode="tp"` (or "ghost", "ghost_fsdp", etc.)
- ✅ Our implementation follows this pattern - no magic auto-detection

### Question 2: Test Pattern
- ✅ YES, we should follow multigpu_*.py pattern
- ✅ We've created `multigpu_controller_tp_test.py` with proper pattern
- ✅ Uses `setup()`/`cleanup()` functions and `env://` init method
- ✅ Matches existing multigpu test conventions

## Files Summary

### Core Implementation
- `opacus/grad_sample/grad_sample_controller_tp.py` - Main TP controller class
- `opacus/grad_sample/utils.py` - Factory function updated for `grad_sample_mode="tp"`
- `opacus/grad_sample/__init__.py` - Export GradSampleControllerTP
- `opacus/__init__.py` - Export at package level

### Tests
- ✅ **`opacus/tests/multigpu_controller_tp_test.py`** - Multi-GPU tests (NEW, recommended)
- ~~`opacus/tests/grad_sample_controller_tp_test.py`~~ - Old version (can delete)

### Examples
- `examples/tensor_parallelism_toy_controller.py` - Simple 2-layer network demo
- `examples/tensor_parallelism_llama_controller.py` - Full Llama training with DP+TP

### Documentation
- `TENSOR_PARALLELISM_CONTROLLER.md` - Comprehensive usage guide
- `IMPLEMENTATION_NOTES.md` - This file

## Testing Commands

```bash
# Run multi-GPU tests (requires 2+ GPUs)
python -m pytest opacus/tests/multigpu_controller_tp_test.py -v

# Run specific test
python -m pytest opacus/tests/multigpu_controller_tp_test.py::GradSampleControllerTPTest::test_merge_flag_setting -v

# Run toy example
python examples/tensor_parallelism_toy_controller.py

# Run Llama example (requires HF token)
python examples/tensor_parallelism_llama_controller.py --token YOUR_TOKEN
```

## Next Steps

1. **Delete old test file** (optional):
   ```bash
   rm opacus/tests/grad_sample_controller_tp_test.py
   ```

2. **Run tests on multi-GPU machine** to verify everything works

3. **Test with real 7B model** using the Llama example

4. **Future enhancements**:
   - Add ghost clipping support (if needed)
   - Add FSDP support (`GradSampleControllerFSDP`)
   - Add combined TP+FSDP support
