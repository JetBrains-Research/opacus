#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Regression tests for AdaClipDPOptimizer + BatchMemoryManager:

Training with equal effective batch size but opposite physical-batch / accumulation
configurations (e.g. 8×16 vs 16×8) must produce identical results.

Root cause that these tests guard against
-----------------------------------------
``AdaClipDPOptimizer.zero_grad()`` previously reset ``sample_size`` and
``unclipped_num`` unconditionally, even for skipped (intermediate) steps.
With ``BatchMemoryManager`` this wiped out the counts accumulated from all
previous physical mini-batches, so only the *last* physical mini-batch
contributed to ``update_max_grad_norm()``.  That made 8×16 use denominator 8
and 16×8 use denominator 16 — producing different clipping-norm trajectories
and therefore different losses.

The fix mirrors the parent-class handling of ``p.summed_grad``: the counters
are preserved when ``_is_last_step_skipped is True`` and only reset after a
full (non-skipped) optimizer step.
"""

import copy
import unittest

import torch
import torch.nn as nn
from opacus.grad_sample import GradSampleModule
from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOGICAL_BATCH = 128
_INPUT_DIM = 8
_OUTPUT_DIM = 2


def _make_model_and_data(seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Linear(_INPUT_DIM, _OUTPUT_DIM, bias=False)
    data_x = torch.randn(_LOGICAL_BATCH, _INPUT_DIM)
    data_y = (torch.rand(_LOGICAL_BATCH) > 0.5).long()
    return model, data_x, data_y


def _make_adaclip_optimizer(gsm: GradSampleModule) -> AdaClipDPOptimizer:
    base_opt = torch.optim.SGD(gsm.parameters(), lr=0.1)
    return AdaClipDPOptimizer(
        optimizer=base_opt,
        noise_multiplier=0,            # deterministic gradient update
        max_grad_norm=1.0,
        expected_batch_size=_LOGICAL_BATCH,
        loss_reduction="mean",
        target_unclipped_quantile=0.5,
        clipbound_learning_rate=0.2,
        max_clipbound=10.0,
        min_clipbound=0.01,
        unclipped_num_std=1e-9,        # negligible noise on unclipped count
    )


def _run_logical_batch(
    gsm: GradSampleModule,
    dp_opt: AdaClipDPOptimizer,
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    physical_batch_size: int,
) -> int:
    """
    Process one logical batch (``_LOGICAL_BATCH`` samples) split into physical
    mini-batches of ``physical_batch_size``.

    Simulates the ``BatchMemoryManager`` pattern:
      for each physical mini-batch:
        signal_skip_step(not is_last)
        zero_grad()
        forward + backward
        step()

    Returns ``dp_opt.sample_size`` *after* the final step (before the next
    ``zero_grad()``), which equals the logical batch size if counting is correct.
    """
    criterion = nn.CrossEntropyLoss()
    num_steps = _LOGICAL_BATCH // physical_batch_size

    for i in range(num_steps):
        is_last = i == num_steps - 1
        x = data_x[i * physical_batch_size : (i + 1) * physical_batch_size]
        y = data_y[i * physical_batch_size : (i + 1) * physical_batch_size]

        dp_opt.signal_skip_step(not is_last)
        dp_opt.zero_grad()

        out = gsm(x)
        loss = criterion(out, y)
        loss.backward()
        dp_opt.step()

    # sample_size is still set here; it gets reset on the *next* zero_grad()
    return dp_opt.sample_size


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class AdaClipGradAccumulationTest(unittest.TestCase):
    """
    Verifies that AdaClipDPOptimizer correctly accumulates ``sample_size`` and
    ``unclipped_num`` across physical mini-batches when used with the
    BatchMemoryManager split-batch pattern.
    """

    # ------------------------------------------------------------------
    # sample_size tracking
    # ------------------------------------------------------------------

    def test_sample_size_equals_logical_batch_small_physical(self):
        """Physical batch 8 × 16 steps: sample_size must equal 128 at step end."""
        model, data_x, data_y = _make_model_and_data()
        gsm = GradSampleModule(model)
        dp_opt = _make_adaclip_optimizer(gsm)

        sample_size = _run_logical_batch(gsm, dp_opt, data_x, data_y, physical_batch_size=8)

        self.assertEqual(
            sample_size,
            _LOGICAL_BATCH,
            f"sample_size={sample_size} != {_LOGICAL_BATCH} for physical_batch=8",
        )

    def test_sample_size_equals_logical_batch_large_physical(self):
        """Physical batch 16 × 8 steps: sample_size must equal 128 at step end."""
        model, data_x, data_y = _make_model_and_data()
        gsm = GradSampleModule(model)
        dp_opt = _make_adaclip_optimizer(gsm)

        sample_size = _run_logical_batch(gsm, dp_opt, data_x, data_y, physical_batch_size=16)

        self.assertEqual(
            sample_size,
            _LOGICAL_BATCH,
            f"sample_size={sample_size} != {_LOGICAL_BATCH} for physical_batch=16",
        )

    def test_sample_size_resets_between_logical_batches(self):
        """After a full step, sample_size is reset to 0 at the start of the next batch."""
        model, data_x, data_y = _make_model_and_data()
        gsm = GradSampleModule(model)
        dp_opt = _make_adaclip_optimizer(gsm)

        _run_logical_batch(gsm, dp_opt, data_x, data_y, physical_batch_size=16)

        # Simulate the first zero_grad of the NEXT logical batch.
        # _is_last_step_skipped is False after a complete step → counters must reset.
        dp_opt.zero_grad()
        self.assertEqual(dp_opt.sample_size, 0)
        self.assertEqual(dp_opt.unclipped_num, 0)

    # ------------------------------------------------------------------
    # Equivalence: 8×16 == 16×8
    # ------------------------------------------------------------------

    def _run_n_logical_batches(
        self,
        physical_batch_size: int,
        n_batches: int,
        model_state: dict,
        data_x: torch.Tensor,
        data_y: torch.Tensor,
    ):
        """Run ``n_batches`` logical batches and return (max_grad_norm, param_list)."""
        model = nn.Linear(_INPUT_DIM, _OUTPUT_DIM, bias=False)
        model.load_state_dict(model_state)
        gsm = GradSampleModule(model)
        dp_opt = _make_adaclip_optimizer(gsm)

        for _ in range(n_batches):
            _run_logical_batch(gsm, dp_opt, data_x, data_y, physical_batch_size)

        params = [p.data.clone() for p in model.parameters()]
        return dp_opt.max_grad_norm, params

    def test_max_grad_norm_equiv_8x16_vs_16x8(self):
        """
        After N logical batches with the same data, the adaptive clipping bound
        must be identical for physical_batch=8 (×16) and physical_batch=16 (×8).
        """
        model, data_x, data_y = _make_model_and_data(seed=7)
        initial_state = copy.deepcopy(model.state_dict())

        norm_a, _ = self._run_n_logical_batches(8,  3, initial_state, data_x, data_y)
        norm_b, _ = self._run_n_logical_batches(16, 3, initial_state, data_x, data_y)

        self.assertAlmostEqual(
            norm_a,
            norm_b,
            places=5,
            msg=f"max_grad_norm mismatch: 8×16={norm_a:.6f}  16×8={norm_b:.6f}",
        )

    def test_model_weights_equiv_8x16_vs_16x8(self):
        """
        After N logical batches (noise_multiplier=0, deterministic), the model
        weights must be identical for physical_batch=8 (×16) and 16 (×8).
        """
        model, data_x, data_y = _make_model_and_data(seed=13)
        initial_state = copy.deepcopy(model.state_dict())

        _, params_a = self._run_n_logical_batches(8,  3, initial_state, data_x, data_y)
        _, params_b = self._run_n_logical_batches(16, 3, initial_state, data_x, data_y)

        for w_a, w_b in zip(params_a, params_b):
            self.assertTrue(
                torch.allclose(w_a, w_b, atol=1e-5),
                f"Weight mismatch between 8×16 and 16×8:\n{w_a}\nvs\n{w_b}",
            )

    # ------------------------------------------------------------------
    # Intermediate state preservation
    # ------------------------------------------------------------------

    def test_sample_size_accumulates_across_skipped_steps(self):
        """
        After each skipped step, sample_size grows by physical_batch_size.
        After the final step it equals the logical batch size.
        """
        model, data_x, data_y = _make_model_and_data()
        gsm = GradSampleModule(model)
        dp_opt = _make_adaclip_optimizer(gsm)
        criterion = nn.CrossEntropyLoss()

        physical_batch_size = 8
        num_steps = _LOGICAL_BATCH // physical_batch_size

        expected_sample_sizes = []
        for i in range(num_steps):
            is_last = i == num_steps - 1
            x = data_x[i * physical_batch_size : (i + 1) * physical_batch_size]
            y = data_y[i * physical_batch_size : (i + 1) * physical_batch_size]

            dp_opt.signal_skip_step(not is_last)
            dp_opt.zero_grad()

            out = gsm(x)
            loss = criterion(out, y)
            loss.backward()
            dp_opt.step()

            expected_sample_sizes.append((i + 1) * physical_batch_size)
            self.assertEqual(
                dp_opt.sample_size,
                expected_sample_sizes[-1],
                f"At step {i}: expected sample_size={expected_sample_sizes[-1]}, "
                f"got {dp_opt.sample_size}",
            )


if __name__ == "__main__":
    unittest.main()
