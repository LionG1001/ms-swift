# Copyright (c) ModelScope Contributors. All rights reserved.
"""Frozen teacher pre-head payloads with explicit identity and position checks."""
import hashlib
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import Dict, Iterator, Optional


def head_fingerprint(weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> str:
    """Fingerprint the exact head values, shapes and dtypes, including a bias."""
    digest = hashlib.sha256()
    for name, value in (('weight', weight), ('bias', bias)):
        digest.update(name.encode())
        if value is None:
            digest.update(b'none')
            continue
        value = value.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TeacherIdentity:
    teacher_id: str
    checkpoint_id: str
    head_version: str
    tokenizer_id: str
    template_id: str
    projection_dtype: str = 'native'

    def __post_init__(self):
        if not all((self.teacher_id, self.checkpoint_id, self.head_version, self.tokenizer_id, self.template_id)):
            raise ValueError('Teacher, checkpoint, head, tokenizer and template identities must be explicit.')
        if self.projection_dtype not in ('native', 'float32'):
            raise ValueError('Teacher projection dtype must be native or float32.')


@dataclass(frozen=True)
class TeacherRequest:
    identity: TeacherIdentity
    trajectory_id: str
    policy_version: int
    target_ids: torch.Tensor
    positions: torch.Tensor
    temperature: float = 1.0

    def __post_init__(self):
        if not self.trajectory_id or self.policy_version < 0 or self.temperature <= 0:
            raise ValueError('Invalid trajectory, policy version or temperature.')
        if self.positions.shape != self.target_ids.shape + (2, ):
            raise ValueError('Positions must carry batch and sequence coordinates for every target.')
        if self.target_ids.dtype != torch.int64 or self.positions.dtype != torch.int64:
            raise ValueError('Token IDs and positions must be int64.')

    def select(self, mask: torch.Tensor) -> 'TeacherRequest':
        return replace(self, target_ids=self.target_ids[mask], positions=self.positions[mask])

    def to_device(self, device) -> 'TeacherRequest':
        return replace(self, target_ids=self.target_ids.to(device), positions=self.positions.to(device))


@dataclass(frozen=True)
class TeacherPayload:
    hidden_states: torch.Tensor
    context: TeacherRequest
    capture_point: str = 'lm_head_input'

    def __post_init__(self):
        if self.capture_point != 'lm_head_input':
            raise ValueError('Hidden states must be captured at the actual LM-head input.')
        if self.hidden_states.requires_grad:
            raise ValueError('Teacher payload must be detached from autograd.')
        if self.hidden_states.shape[:-1] != self.context.target_ids.shape:
            raise ValueError('Teacher hidden states and token positions disagree.')

    def select(self, mask: torch.Tensor) -> 'TeacherPayload':
        return replace(self, hidden_states=self.hidden_states[mask], context=self.context.select(mask))

    def to_device(self, device) -> 'TeacherPayload':
        return replace(self, hidden_states=self.hidden_states.to(device), context=self.context.to_device(device))


@dataclass
class _HeadEntry:
    identity: TeacherIdentity
    weight: torch.Tensor
    bias: Optional[torch.Tensor]


class TeacherHeadStore:
    """Store immutable head snapshots; reject mismatched payloads before projection."""

    def __init__(self):
        self._heads: Dict[str, _HeadEntry] = {}

    def register(self, identity: TeacherIdentity, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        if weight.ndim != 2 or weight.requires_grad or (bias is not None and bias.requires_grad):
            raise ValueError('Teacher head must be a frozen two-dimensional weight with optional frozen bias.')
        if bias is not None and bias.shape != (weight.shape[0], ):
            raise ValueError('Teacher head bias has an incompatible vocabulary dimension.')
        if head_fingerprint(weight, bias) != identity.head_version:
            raise ValueError('Teacher head values do not match the declared head version.')
        if identity.teacher_id in self._heads:
            raise ValueError('Teacher head already registered; use a new store for a new frozen version.')
        self._heads[identity.teacher_id] = _HeadEntry(identity,
                                                      weight.detach().cpu().clone(),
                                                      bias.detach().cpu().clone() if bias is not None else None)

    def project(self, payload: TeacherPayload, expected: TeacherRequest) -> torch.Tensor:
        context = payload.context
        if context.identity != expected.identity:
            raise ValueError('Teacher/checkpoint/head/tokenizer/template identity mismatch.')
        if (context.trajectory_id, context.policy_version,
                context.temperature) != (expected.trajectory_id, expected.policy_version, expected.temperature):
            raise ValueError('Teacher trajectory, policy or temperature mismatch.')
        if not torch.equal(context.target_ids, expected.target_ids):
            raise ValueError('Teacher response token IDs do not match the request.')
        if not torch.equal(context.positions, expected.positions):
            raise ValueError('Teacher token position offset detected.')
        entry = self._heads.get(expected.identity.teacher_id)
        if entry is None or entry.identity != expected.identity:
            raise ValueError('Requested frozen teacher head is not loaded.')
        hidden = payload.hidden_states
        if hidden.shape[-1] != entry.weight.shape[1] or hidden.dtype != entry.weight.dtype:
            raise ValueError('Teacher hidden width or dtype is incompatible with the registered head.')
        # The head snapshot is CPU-backed; GPU residency lasts for this projection.
        # A multi-teacher cache/scheduler can later implement a separate lifecycle.
        with torch.no_grad():
            dtype = torch.float32 if entry.identity.projection_dtype == 'float32' else hidden.dtype
            weight = entry.weight.to(device=hidden.device, dtype=dtype)
            bias = entry.bias.to(device=hidden.device, dtype=dtype) if entry.bias is not None else None
            precision = torch.autocast(hidden.device.type, enabled=False) if dtype == torch.float32 else nullcontext()
            with precision:
                return F.linear(hidden.to(dtype), weight, bias)


class TeacherProvider(ABC):
    """Interface for obtaining teacher payloads without coupling the loss to a backend."""

    @abstractmethod
    def capture(self) -> Iterator[None]:
        raise NotImplementedError

    @abstractmethod
    def make_payload(self, request: TeacherRequest) -> TeacherPayload:
        raise NotImplementedError


class PreHeadTeacherProvider(TeacherProvider):
    """Capture the real output-layer input, after the model's final normalization.

    The initial local Megatron integration supports TP=PP=CP=1. Other model
    parallel layouts need an explicit gather/position implementation.
    """

    def __init__(self, head: torch.nn.Module, *, input_layout: str):
        if input_layout not in ('SBH', 'BSH'):
            raise ValueError('Expected SBH (Megatron) or BSH (Transformers) input layout.')
        self.head = head
        self.input_layout = input_layout
        self._hidden: Optional[torch.Tensor] = None
        self._active = False
        self._calls = 0
        self._weight: Optional[torch.Tensor] = None
        self._bias: Optional[torch.Tensor] = None

    @property
    def head_weight(self) -> torch.Tensor:
        if self._weight is None:
            raise RuntimeError('No actual LM-head weight was captured.')
        return self._weight

    @property
    def head_bias(self) -> Optional[torch.Tensor]:
        return self._bias

    @contextmanager
    def capture(self) -> Iterator[None]:
        if self._active:
            raise RuntimeError('Nested teacher capture is unsupported.')
        self._active = True
        self._hidden = None
        self._calls = 0
        self._weight = None
        self._bias = None

        def before_head(module, inputs, kwargs):
            if not inputs or not isinstance(inputs[0], torch.Tensor) or inputs[0].ndim != 3:
                raise ValueError('LM-head input must be a three-dimensional tensor.')
            hidden = inputs[0].detach()
            if self.input_layout == 'SBH':
                hidden = hidden.transpose(0, 1)
            self._hidden = hidden.clone().contiguous()
            # Megatron tied embeddings pass the weight to output_layer rather
            # than allocating output_layer.weight. Capture the weight actually used.
            weight = kwargs.get('weight')
            if weight is None and len(inputs) > 1:
                weight = inputs[1]
            self._weight = weight if weight is not None else module.weight
            self._bias = module.bias
            if self._weight is None:
                raise ValueError('The actual LM-head weight is unavailable.')
            self._calls += 1

        handle = self.head.register_forward_pre_hook(before_head, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()
            self._active = False

    def make_payload(self, request: TeacherRequest) -> TeacherPayload:
        if self._calls != 1 or self._hidden is None:
            raise RuntimeError('Exactly one actual LM-head call must produce each teacher payload.')
        # Independent metadata copies let the receiver detect accidental shifts
        # or stale metadata without changing its authoritative request.
        context = replace(request, target_ids=request.target_ids.clone(), positions=request.positions.clone())
        return TeacherPayload(self._hidden, context)
