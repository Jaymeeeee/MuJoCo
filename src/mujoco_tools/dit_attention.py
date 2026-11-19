"""Passive DiT action self-attention and RGB-D spatial-pooling diagnostics.

The action attention axes are trajectory timesteps, never image locations.
The RGB-D encoder's SpatialSoftmax maps are a separate feature-pooling view;
they are neither decoder cross-attention nor causal pixel importance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diffusion_attention import DiffusionAttentionTrace


class RGBDSpatialAttentionTrace(DiffusionAttentionTrace):
    """Reuse spatial tracing with explicitly validated early-fused RGB-D inputs.

This leaves the historical RGB-only tracer unchanged. Camera names remain the
RGB feature names, while a saved depth_camera_map authorizes a fourth channel.
An undeclared four-channel encoder is rejected instead of silently accepted.
"""

    def __init__(self, model, *, image_keys=None):
        super().__init__(model, image_keys=image_keys)
        mapping = getattr(model.config, 'depth_camera_map', {})
        if mapping:
            from lerobot.policies.rgbd import validate_depth_camera_map

            validate_depth_camera_map(model.config)
        expected_channels = 4 if mapping else 3
        for key, encoder in zip(self.image_keys, self._encoders, strict=True):
            first_conv = next((module for module in encoder.modules()
                               if isinstance(module, self._torch.nn.Conv2d)), None)
            if first_conv is None or first_conv.in_channels != expected_channels:
                raise ValueError('Camera encoder input channels disagree with declared depth_camera_map')
            self._image_shapes[key] = (expected_channels, *self._image_shapes[key][1:])
        self.depth_camera_map = dict(mapping)


@dataclass(frozen=True)
class DiTAttentionCapture:
    """Selected denoising steps from one completed denoiser schedule.

``weights`` is (selected steps, layers, heads, query actions, key actions).
``mean_weights`` averages heads only. ``attention_gates`` is
(selected steps, layers, hidden width), containing the actual AdaLN residual
gate, which is signed and is not a probability or camera-importance weight.
Arrays are owned, read-only CPU copies. Denoiser sampling ends just after the
last denoiser call; archive only after the caller's action-selection succeeds.
"""

    timesteps: np.ndarray
    sampling_timesteps: np.ndarray
    weights: np.ndarray
    mean_weights: np.ndarray
    attention_gates: np.ndarray
    reference_weights_max_abs_error: np.ndarray
    live_output_max_abs_error: np.ndarray
    layer_indices: np.ndarray
    num_heads: int
    action_horizon: int


class DiTAttentionTrace:
    """Read real MHA inputs while leaving its need_weights=False path intact.

Use ``with DiTAttentionTrace(adapter.policy.diffusion) as trace: ...``.
``latest`` and ``capture_count`` advance after a complete sampler schedule;
cached actions create no capture. Selected steps are saved in scheduler order.
No model, argument, output, evaluation mode, parameter or RNG is modified.

Attention is reconstructed from the actual packed Q/K/V projection and checked
against PyTorch's functional MHA weights. Its reconstructed value/output is
also checked against the actual live MHA output. The numerical checks do not
replace a caller's fixed-observation action/RNG parity verification.
"""

    def __init__(self, model, *, timesteps=(99, 75, 50, 25, 0)):
        import torch

        self._torch = torch
        self.model = model
        config = getattr(model, 'config', None)
        if getattr(config, 'type', None) != 'diffusion' or getattr(config, 'denoiser_type', None) != 'dit':
            raise ValueError('DiT tracing requires explicit config.type="diffusion", denoiser_type="dit"')
        if getattr(config, 'compile_model', False):
            raise ValueError('DiT tracing requires the uncompiled local denoiser')
        if not timesteps or any(type(t) is not int or t < 0 for t in timesteps):
            raise ValueError('timesteps must be nonempty nonnegative integers')
        if len(set(timesteps)) != len(timesteps):
            raise ValueError('timesteps must be unique')
        self.selected_timesteps = tuple(timesteps)
        self._selected = set(timesteps)
        self.denoiser = getattr(model, 'denoiser', None)
        blocks = getattr(self.denoiser, 'blocks', None)
        if not isinstance(self.denoiser, torch.nn.Module) or not isinstance(blocks, torch.nn.ModuleList) or not len(blocks):
            raise ValueError('Expected local DiT denoiser.blocks')
        self._blocks = tuple(blocks)
        self._attentions = tuple(getattr(block, 'attention', None) for block in blocks)
        for attention in self._attentions:
            if (not isinstance(attention, torch.nn.MultiheadAttention) or not attention.batch_first
                    or not attention._qkv_same_embed_dim or attention.in_proj_weight is None
                    or attention.bias_k is not None or attention.bias_v is not None or attention.add_zero_attn):
                raise ValueError('Only packed-projection, batch-first DiT self-attention is supported')
        self.num_heads = self._attentions[0].num_heads
        self.hidden_dim = self._attentions[0].embed_dim
        if any(attention.num_heads != self.num_heads or attention.embed_dim != self.hidden_dim
               for attention in self._attentions):
            raise ValueError('All DiT attention blocks must have matching heads and width')
        if any(not isinstance(getattr(block, 'conditioning', None), torch.nn.Module) for block in blocks):
            raise ValueError('Expected DiT blocks with AdaLN conditioning')
        self.capture_count = 0
        self.denoiser_call_count = 0
        self.latest: DiTAttentionCapture | None = None
        self._handles = []
        self._clear_pending()

    @staticmethod
    def _readonly(array, *, dtype=np.float32):
        value = np.array(array, dtype=dtype, copy=True, order='C')
        value.flags.writeable = False
        return value

    def _clear_pending(self):
        self._schedule = None
        self._step_index = 0
        self._active_timestep = None
        self._sequence_length = None
        self._pending_steps = []
        self._layer_rows = {}
        self._gates = {}
        self._pending_attention = {}

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._clear_pending()

    def clear(self):
        if self._active_timestep is not None:
            raise RuntimeError('Cannot clear attention inside a denoiser forward')
        self._clear_pending()
        self.latest = None

    def _fail(self, message):
        self.close()
        raise ValueError(message)

    def __enter__(self):
        if self._handles:
            raise RuntimeError('DiTAttentionTrace is already active')
        if self.model.training or any(module.training for module in self.denoiser.modules()):
            raise ValueError('DiT tracing requires model.eval()')
        try:
            self._handles.append(self.denoiser.register_forward_pre_hook(self._begin_step, with_kwargs=True))
            self._handles.append(self.denoiser.register_forward_hook(self._finish_step, always_call=True))
            for index, block in enumerate(self._blocks):
                self._handles.append(block.conditioning.register_forward_hook(
                    lambda module, args, output, i=index: self._capture_gate(i, output)))
                self._handles.append(block.attention.register_forward_pre_hook(
                    lambda module, args, kwargs, i=index: self._begin_attention(i, module, args, kwargs),
                    with_kwargs=True))
                self._handles.append(block.attention.register_forward_hook(
                    lambda module, args, output, i=index: self._finish_attention(i, output), always_call=True))
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _begin_step(self, module, args, kwargs):
        torch = self._torch
        if self.model.training or any(block.training for block in self._blocks):
            self._fail('DiT tracing requires model.eval()')
        if self._active_timestep is not None:
            self._fail('Nested DiT denoiser forwards are unsupported')
        sample = args[0] if args else kwargs.get('x')
        timestep = args[1] if len(args) > 1 else kwargs.get('timestep')
        if not isinstance(sample, torch.Tensor) or sample.ndim != 3 or sample.shape[0] != 1:
            self._fail('DiT attention capture requires batch size one and (B,T,A) input')
        if sample.dtype not in (torch.float32, torch.float64):
            self._fail('DiT attention capture currently requires float32/float64 inference without autocast')
        t = torch.as_tensor(timestep).detach().reshape(-1)
        if t.numel() != 1 or not bool(torch.isfinite(t).all()) or float(t[0]) != int(t[0]):
            self._fail('DiT attention requires one integer diffusion timestep')
        actual_timestep = int(t[0])
        if self._schedule is None:
            scheduler = getattr(self.model, 'noise_scheduler', None)
            steps = getattr(scheduler, 'timesteps', None)
            if steps is None:
                self._fail('Missing active diffusion scheduler timesteps')
            self._schedule = tuple(int(x) for x in steps)
            if not self._schedule or len(set(self._schedule)) != len(self._schedule):
                self._fail('Expected a nonempty schedule without duplicate timesteps')
            if not self._selected.issubset(self._schedule):
                self._fail('Requested attention timesteps are absent from the active sampler schedule')
            self._sequence_length = int(sample.shape[1])
        if (actual_timestep != self._schedule[self._step_index]
                or sample.shape[1] != self._sequence_length):
            self._fail('Denoiser calls do not follow the declared sampler schedule/sequence length')
        self._active_timestep = actual_timestep
        self._layer_rows.clear()
        self._gates.clear()
        self._pending_attention.clear()

    def _capture_gate(self, index, output):
        if self._active_timestep not in self._selected:
            return
        if index in self._gates or tuple(output.shape) != (1, 6*self.hidden_dim):
            self._fail('Unexpected DiT conditioning invocation or shape')
        gate = output.detach()[0, 2*self.hidden_dim:3*self.hidden_dim].float().cpu().numpy()
        if not np.isfinite(gate).all():
            self._fail('Nonfinite DiT attention gate')
        self._gates[index] = self._readonly(gate)

    def _begin_attention(self, index, module, args, kwargs):
        if self._active_timestep not in self._selected:
            return
        torch = self._torch
        functional = torch.nn.functional
        if module.training or index not in self._gates or index in self._pending_attention or index in self._layer_rows:
            self._fail('Unexpected DiT attention order, duplicate call or training mode')
        if (len(args) != 3 or args[0] is not args[1] or args[0] is not args[2]
                or kwargs.get('need_weights', True) is not False
                or kwargs.get('key_padding_mask') is not None or kwargs.get('attn_mask') is not None
                or kwargs.get('is_causal', False)):
            self._fail('Expected unmasked self-attention with unchanged need_weights=False')
        tokens = args[0]
        expected = (1, self._sequence_length, self.hidden_dim)
        if tuple(tokens.shape) != expected or tokens.dtype not in (torch.float32, torch.float64):
            self._fail('Unexpected DiT attention token shape/dtype (autocast unsupported)')
        if torch.is_autocast_enabled(tokens.device.type):
            self._fail('DiT attention capture does not support autocast')
        with torch.no_grad():
            # Same packed projection, on the actual already-modulated inputs.
            qkv = functional.linear(tokens.detach(), module.in_proj_weight, module.in_proj_bias)
            head_dim = self.hidden_dim // self.num_heads
            q, k, v = [item.reshape(1, self._sequence_length, self.num_heads, head_dim).transpose(1, 2)
                       for item in qkv.chunk(3, dim=-1)]
            weights = torch.softmax((q * (head_dim ** -.5)) @ k.transpose(-2, -1), dim=-1)
            attended = (weights @ v).transpose(1, 2).contiguous().reshape(expected)
            reconstructed = functional.linear(attended, module.out_proj.weight, module.out_proj.bias)
            transposed = tokens.detach().transpose(0, 1)
            _, reference = functional.multi_head_attention_forward(
                transposed, transposed, transposed, self.hidden_dim, self.num_heads,
                module.in_proj_weight, module.in_proj_bias, None, None, False, 0.,
                module.out_proj.weight, module.out_proj.bias, training=False, need_weights=True,
                average_attn_weights=False, is_causal=False)
            tolerance = 2e-6 if tokens.dtype == torch.float32 else 1e-10
            if not torch.isfinite(weights).all() or not torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)),
                                                                     rtol=tolerance, atol=tolerance):
                self._fail('DiT attention probabilities are nonfinite or do not sum to one')
            if not torch.allclose(weights, reference, rtol=2e-5, atol=tolerance):
                self._fail('Reconstructed DiT attention differs from functional PyTorch MHA weights')
            self._pending_attention[index] = dict(
                weights=self._readonly(weights[0].float().cpu().numpy()),
                expected_output=reconstructed.detach(), tolerance=tolerance,
                reference_error=float((weights-reference).abs().max().cpu()))

    def _finish_attention(self, index, output):
        if self._active_timestep not in self._selected:
            return
        if output is None:
            self.close()
            return
        row = self._pending_attention.pop(index, None)
        if row is None or not isinstance(output, tuple) or len(output) != 2 or output[1] is not None:
            self._fail('Unexpected live DiT attention return value')
        actual = output[0].detach()
        expected = row.pop('expected_output')
        if not self._torch.allclose(actual, expected, rtol=2e-5, atol=row.pop('tolerance')):
            self._fail('Reconstructed DiT attention values differ from the actual live MHA output')
        row['live_error'] = float((actual-expected).abs().max().cpu())
        row['gate'] = self._gates[index]
        self._layer_rows[index] = row

    def _finish_step(self, module, args, output):
        if self._active_timestep is None:
            return
        if output is None:
            self.close()
            return
        if self._active_timestep in self._selected:
            if len(self._layer_rows) != len(self._blocks) or self._pending_attention:
                self._fail('DiT denoiser finished without every selected attention block')
            self._pending_steps.append((self._active_timestep, [self._layer_rows[i] for i in range(len(self._blocks))]))
        self.denoiser_call_count += 1
        self._step_index += 1
        self._active_timestep = None
        if self._step_index == len(self._schedule):
            steps = self._pending_steps
            weights = np.stack([np.stack([layer['weights'] for layer in layers]) for _, layers in steps])
            self.latest = DiTAttentionCapture(
                timesteps=self._readonly([t for t, _ in steps], dtype=np.int64),
                sampling_timesteps=self._readonly(self._schedule, dtype=np.int64),
                weights=self._readonly(weights), mean_weights=self._readonly(weights.mean(axis=2)),
                attention_gates=self._readonly([[layer['gate'] for layer in layers] for _, layers in steps]),
                reference_weights_max_abs_error=self._readonly([[layer['reference_error'] for layer in layers] for _, layers in steps]),
                live_output_max_abs_error=self._readonly([[layer['live_error'] for layer in layers] for _, layers in steps]),
                layer_indices=self._readonly(range(len(self._blocks)), dtype=np.int64),
                num_heads=self.num_heads, action_horizon=self._sequence_length)
            self.capture_count += 1
            self._clear_pending()
