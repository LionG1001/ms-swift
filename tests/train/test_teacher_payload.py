# Copyright (c) ModelScope Contributors. All rights reserved.
"""Identity failures and loss/gradient equivalence for real pre-head captures."""
import pytest
import torch
import torch.nn.functional as F
from dataclasses import replace

from swift.rlhf_trainers.gkd_loss import TeacherOutput, gkd_loss
from swift.rlhf_trainers.teacher_payload import (PreHeadTeacherProvider, TeacherHeadStore, TeacherIdentity,
                                                 TeacherPayload, TeacherRequest, head_fingerprint)


def fixture_payload():
    torch.manual_seed(42)
    teacher = torch.nn.Sequential(torch.nn.LayerNorm(7), torch.nn.Linear(7, 13, bias=True))
    teacher.requires_grad_(False).eval()
    head = teacher[1]
    identity = TeacherIdentity('teacher-a', 'checkpoint-a', head_fingerprint(head.weight, head.bias), 'vocab-a',
                               'template-a')
    store = TeacherHeadStore()
    store.register(identity, head.weight, head.bias)
    labels = torch.arange(10).reshape(2, 5).long()
    labels[:, :2] = -100
    batch, sequence = torch.meshgrid(torch.arange(2), torch.arange(5), indexing='ij')
    positions = torch.stack((batch, sequence), dim=-1)
    request = TeacherRequest(identity, 'trajectory-a', 0, labels.clone(), positions)
    provider = PreHeadTeacherProvider(head, input_layout='BSH')
    with torch.no_grad(), provider.capture():
        reference = teacher(torch.randn(2, 5, 7))
    return teacher, store, provider.make_payload(request), request, labels, reference


@pytest.mark.parametrize('chunk_size', [1, 4, 512])
@pytest.mark.parametrize('all_masked', [False, True])
def test_full_vocab_loss_and_student_hidden_head_gradients(chunk_size, all_masked):
    teacher, store, payload, request, labels, reference = fixture_payload()
    if all_masked:
        labels = torch.full_like(labels, -100)
        request = replace(request, target_ids=labels.clone())
        payload = replace(payload, context=replace(payload.context, target_ids=labels.clone()))
    student_hidden = torch.randn(2, 5, 7, requires_grad=True)
    student_head = torch.randn(13, 7, requires_grad=True)
    student_bias = torch.randn(13, requires_grad=True)
    logits = F.linear(student_hidden, student_head, student_bias)
    output = TeacherOutput(labels=labels, payload=payload, request=request, head_store=store)
    actual, count = gkd_loss(logits, output, labels, beta=1, temperature=1, chunk_size=chunk_size)
    mask = labels.ne(-100)
    s_log = logits[mask].log_softmax(-1)
    t_log = reference[mask].log_softmax(-1)
    expected = (s_log.exp() * (s_log - t_log)).sum()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    parameters = (student_hidden, student_head, student_bias)
    actual_grads = torch.autograd.grad(actual, parameters, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-6)
    assert count.item() == mask.sum().item()
    assert not actual_grads[0][~mask].count_nonzero()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    torch.testing.assert_close(store.project(payload, request), reference)


@pytest.mark.parametrize('field', ['teacher_id', 'checkpoint_id', 'head_version', 'tokenizer_id', 'template_id'])
def test_wrong_identity_is_rejected(field):
    _, store, payload, request, _, _ = fixture_payload()
    wrong = replace(payload.context.identity, **{field: 'incorrect'})
    payload = replace(payload, context=replace(payload.context, identity=wrong))
    with pytest.raises(ValueError, match='identity mismatch'):
        store.project(payload, request)


@pytest.mark.parametrize('field,value', [('trajectory_id', 'other'), ('policy_version', 1), ('temperature', 2.0)])
def test_stale_payload_is_rejected(field, value):
    _, store, payload, request, _, _ = fixture_payload()
    payload = replace(payload, context=replace(payload.context, **{field: value}))
    with pytest.raises(ValueError, match='trajectory, policy or temperature'):
        store.project(payload, request)


def test_wrong_head_values_and_token_positions_are_rejected():
    _, store, payload, request, _, _ = fixture_payload()
    with pytest.raises(ValueError, match='head version'):
        TeacherHeadStore().register(request.identity, torch.zeros(13, 7))
    shifted = replace(payload, context=replace(payload.context, positions=payload.context.positions + 1))
    with pytest.raises(ValueError, match='position offset'):
        store.project(shifted, request)
    shifted = replace(payload, context=replace(payload.context, target_ids=payload.context.target_ids + 1))
    with pytest.raises(ValueError, match='token IDs'):
        store.project(shifted, request)


def test_capture_requires_actual_head_and_normalized_input():
    _, store, payload, request, _, reference = fixture_payload()
    torch.testing.assert_close(store.project(payload, request), reference)
    provider = PreHeadTeacherProvider(torch.nn.Linear(7, 13), input_layout='SBH')
    with provider.capture():
        pass
    with pytest.raises(RuntimeError, match='Exactly one'):
        provider.make_payload(request)
    with pytest.raises(ValueError, match='detached'):
        TeacherPayload(payload.hidden_states.clone().requires_grad_(), request)


def test_ragged_masks_match_full_vocab_reference():
    _, store, payload, request, labels, reference = fixture_payload()
    labels = labels.clone()
    labels[0, 3:] = -100
    request = replace(request, target_ids=labels.clone())
    payload = replace(payload, context=replace(payload.context, target_ids=labels.clone()))
    logits = torch.randn_like(reference, requires_grad=True)
    output = TeacherOutput(labels=labels, payload=payload, request=request, head_store=store)
    actual, count = gkd_loss(logits, output, labels, beta=1, temperature=1, chunk_size=2)
    active = labels.ne(-100)
    s_log = logits[active].log_softmax(-1)
    t_log = reference[active].log_softmax(-1)
    expected = (s_log.exp() * (s_log - t_log)).sum()
    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual, logits, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert count.item() == 4


def test_identical_distributions_have_zero_loss_and_gradient():
    _, store, payload, request, labels, reference = fixture_payload()
    logits = reference.clone().requires_grad_()
    output = TeacherOutput(labels=labels, payload=payload, request=request, head_store=store)
    loss, count = gkd_loss(logits, output, labels, beta=1, temperature=1)
    gradient, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0)
    assert count.item() == 6


@pytest.mark.parametrize('failure', ['token', 'sequence_count', 'temperature', 'vocabulary'])
def test_loss_rejects_incompatible_student(failure):
    _, store, payload, request, labels, reference = fixture_payload()
    student_labels = labels.clone()
    logits = reference.clone().requires_grad_()
    temperature = 1
    if failure == 'token':
        student_labels[0, 2] += 1
    elif failure == 'sequence_count':
        student_labels[0, 2] = -100
        student_labels[1, 1] = 0
    elif failure == 'temperature':
        temperature = 2
    else:
        logits = logits[..., :-1]
    output = TeacherOutput(labels=labels, payload=payload, request=request, head_store=store)
    with pytest.raises(ValueError):
        gkd_loss(logits, output, student_labels, beta=1, temperature=temperature)


@pytest.mark.parametrize('keyword_weight', [False, True])
def test_megatron_layout_and_actual_tied_head_weight(keyword_weight):

    class TiedHead(torch.nn.Module):
        weight = None
        bias = None

        def forward(self, hidden, weight=None):
            return F.linear(hidden, weight)

    _, _, _, request, _, _ = fixture_payload()
    head = TiedHead()
    weight = torch.randn(13, 7)
    normalized_hidden = torch.nn.functional.layer_norm(torch.randn(5, 2, 7), (7, ))
    provider = PreHeadTeacherProvider(head, input_layout='SBH')
    with torch.no_grad(), provider.capture():
        if keyword_weight:
            reference = head(normalized_hidden, weight=weight)
        else:
            reference = head(normalized_hidden, weight)
    identity = replace(request.identity, head_version=head_fingerprint(provider.head_weight))
    request = replace(request, identity=identity)
    store = TeacherHeadStore()
    store.register(identity, provider.head_weight, provider.head_bias)
    actual = store.project(provider.make_payload(request), request)
    torch.testing.assert_close(actual, reference.transpose(0, 1))


@pytest.mark.parametrize('chunk_size', [1, 4, 512])
@pytest.mark.parametrize('all_masked', [False, True])
def test_bf16_payload_uses_fp32_probabilities_and_matches_gradients(chunk_size, all_masked):
    torch.manual_seed(47)
    teacher = torch.nn.Sequential(torch.nn.LayerNorm(7), torch.nn.Linear(7, 13)).bfloat16()
    teacher.requires_grad_(False).eval()
    head = teacher[1]
    identity = TeacherIdentity('teacher-bf16', 'checkpoint-bf16', head_fingerprint(head.weight, head.bias),
                               'vocab-bf16', 'template-bf16')
    store = TeacherHeadStore()
    store.register(identity, head.weight, head.bias)
    labels = torch.arange(10).reshape(2, 5).long()
    labels[0, :2] = -100
    labels[1, :4] = -100
    if all_masked:
        labels.fill_(-100)
    batch, sequence = torch.meshgrid(torch.arange(2), torch.arange(5), indexing='ij')
    request = TeacherRequest(identity, 'trajectory-bf16', 0, labels, torch.stack((batch, sequence), -1))
    provider = PreHeadTeacherProvider(head, input_layout='BSH')
    with torch.no_grad(), provider.capture():
        teacher_logits = teacher(torch.randn(2, 5, 7).bfloat16())
    payload = provider.make_payload(request)
    hidden = torch.randn(2, 5, 7).bfloat16().requires_grad_()
    weight = torch.randn(13, 7).bfloat16().requires_grad_()
    bias = torch.randn(13).bfloat16().requires_grad_()
    logits = F.linear(hidden, weight, bias)
    output = TeacherOutput(labels=labels, payload=payload, request=request, head_store=store)
    actual, count = gkd_loss(logits, output, labels, beta=1, temperature=1, chunk_size=chunk_size)
    mask = labels.ne(-100)
    s_log = logits[mask].float().log_softmax(-1)
    t_log = teacher_logits[mask].float().log_softmax(-1)
    expected = (s_log.exp() * (s_log - t_log)).sum()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual_grads = torch.autograd.grad(actual, (hidden, weight, bias), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (hidden, weight, bias))
    for actual_gradient, expected_gradient in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_gradient.float(), expected_gradient.float(), atol=1e-5, rtol=1 / 64)
    assert count.item() == mask.sum().item()
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_explicit_fp32_projection_preserves_native_snapshot_and_identity():
    _, _, payload, request, labels, _ = fixture_payload()
    weight = torch.randn(13, 7).bfloat16()
    hidden = payload.hidden_states.bfloat16()
    identity = replace(request.identity, head_version=head_fingerprint(weight), projection_dtype='float32')
    request = replace(request, identity=identity)
    payload = TeacherPayload(hidden, request)
    store = TeacherHeadStore()
    store.register(identity, weight)
    before = head_fingerprint(weight)
    actual = store.project(payload, request)
    expected = F.linear(hidden.float(), weight.float())
    assert actual.dtype == torch.float32 and weight.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert head_fingerprint(weight) == before
    with pytest.raises(ValueError, match='identity mismatch'):
        store.project(payload, replace(request, identity=replace(identity, projection_dtype='native')))
    student = actual.detach().clone().requires_grad_()
    loss, _ = gkd_loss(student, TeacherOutput(labels=labels, payload=payload, request=request, head_store=store),
                       labels, 1, 1)
    loss.backward()
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)
    torch.testing.assert_close(student.grad, torch.zeros_like(student), atol=1e-7, rtol=0)


def test_unknown_projection_precision_is_rejected():
    with pytest.raises(ValueError, match='projection dtype'):
        TeacherIdentity('t', 'c', 'h', 'v', 'template', projection_dtype='unsupported')
