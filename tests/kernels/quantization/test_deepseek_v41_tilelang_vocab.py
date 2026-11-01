# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang vocabulary heads: the exact packed BF16 head and BF16
heads, against FP32 references, with logits independent of the batch."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm.models.deepseek_v4_1.tilelang import vocab as packed_format  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.vocab import (  # noqa: E402
    ROW_TILES,
    nvfp4_vocab_projection,
    pack_bf16_weight,
    packed_vocab_projection,
    prepare_bf16_head,
    prepare_nvfp4_head,
    prepare_packed_head,
    project_head,
    project_head_greedy_pairs,
    project_head_markov_greedy_pairs,
    row_tile,
    tile_config,
    unpack_bf16_weight,
)

DEVICE = "cuda"
SHAPES = [(1000, 512), (32320, 5120)]


def _head(n, k, seed):
    """Head-like BF16 weights with values outside any exponent window, and zeros."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    weight = (torch.randn(n, k, generator=generator, device=DEVICE) * 0.02).bfloat16()
    flat = weight.view(-1)
    picks = torch.randint(0, flat.numel(), (64,), generator=generator, device=DEVICE)
    flat[picks[:32]] = 300.0
    flat[picks[32:48]] = -1.5e-7
    flat[picks[48:]] = 0.0
    return weight


def _project(packed, n, k, x, block_M, **config):
    kernel = packed_vocab_projection(
        n,
        k,
        packed.exponent_base,
        block_M=block_M,
        **(config or tile_config(block_M, k)),
    )
    out = torch.empty(x.shape[0], n, dtype=torch.bfloat16, device=DEVICE)
    kernel(
        x,
        packed.values,
        packed.exception_rows,
        packed.exception_entries,
        packed.row_offsets,
        out,
    )
    return out


def _close_to_fp32(out, x, weight):
    reference = x.float() @ weight.float().T
    error = (out.float() - reference).abs()
    bound = reference.abs() * 2.0**-7 + 1e-3 * reference.abs().max()
    return bool((error <= bound).all())


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("rows", [1, 5, 16, 33, 64, 100])
def test_packed_head_matches_fp32_reference(n, k, rows):
    weight = _head(n, k, seed=n + k)
    packed = packed_format.pack(weight.clone())
    assert packed.exceptions > 0
    x = (torch.randn(rows, k, device=DEVICE) * 0.5).bfloat16()
    assert _close_to_fp32(_project(packed, n, k, x, row_tile(rows)), x, weight)


@pytest.mark.parametrize("n,k", SHAPES)
def test_packed_head_rows_independent_of_batch_and_tile(n, k):
    weight = _head(n, k, seed=7)
    packed = packed_format.pack(weight.clone())
    x = (torch.randn(128, k, device=DEVICE) * 0.5).bfloat16()
    full = _project(packed, n, k, x, 64)
    for block_M in ROW_TILES:
        assert torch.equal(_project(packed, n, k, x, block_M), full), block_M
    for rows in (1, 2, 6, 17, 40, 65, 96, 128):
        part = _project(packed, n, k, x[:rows].contiguous(), row_tile(rows))
        assert torch.equal(part, full[:rows]), rows
    assert torch.equal(_project(packed, n, k, x[13:14].contiguous(), 16), full[13:14])
    # Other tiles, step widths and pipeline depths give the same bits.
    for block_M, config in (
        (16, dict(block_N=32, num_stages=2, step_groups=1)),
        (16, dict(block_N=16, threads=64, num_stages=4, step_groups=2)),
        (64, dict(block_N=32, threads=128, num_stages=3)),
    ):
        rows = 9 if block_M == 16 else 64
        part = _project(packed, n, k, x[:rows].contiguous(), block_M, **config)
        assert torch.equal(part, full[:rows]), config


def test_packed_head_exceptions_are_exact():
    n, k = 512, 256
    weight = torch.full((n, k), 0.01, dtype=torch.bfloat16, device=DEVICE)
    weight[5, 17] = 1.0e30  # far outside any window holding 0.01
    packed = packed_format.pack(weight.clone())
    assert packed.exceptions == 1
    x = torch.zeros(3, k, dtype=torch.bfloat16, device=DEVICE)
    x[:, 17] = 1.0
    out = _project(packed, n, k, x, 16)
    assert torch.equal(out[:, 5], weight[5, 17].expand(3))
    assert torch.equal(out[:, 4], weight[4, 17].expand(3))


def test_packed_head_without_exceptions():
    n, k = 640, 384
    weight = (torch.rand(n, k, device=DEVICE) * 0.5 + 0.5).bfloat16()
    packed = packed_format.pack(weight.clone())
    assert packed.exceptions == 0
    x = (torch.randn(7, k, device=DEVICE) * 0.5).bfloat16()
    assert _close_to_fp32(_project(packed, n, k, x, 16), x, weight)


def _packed_layer(weight):
    packed = packed_format.pack(weight.clone())
    return SimpleNamespace(
        weight=packed.values,
        weight_exception_rows=packed.exception_rows,
        weight_exception_entries=packed.exception_entries,
        weight_row_offsets=packed.row_offsets,
        packed_bf16_in_features=weight.shape[1],
        packed_bf16_exponent_base=packed.exponent_base,
    )


@pytest.mark.parametrize("rows", [1, 6, 48, 130])
def test_prepared_packed_head(rows):
    n, k = 1000, 512
    weight = _head(n, k, seed=3)
    layer = _packed_layer(weight)
    prepare_packed_head(layer)
    assert set(layer.tilelang_vocab) == set(ROW_TILES)
    x = (torch.randn(rows, k, device=DEVICE) * 0.5).bfloat16()
    logits = project_head(layer, x)
    assert logits.shape == (rows, n) and logits.dtype == torch.bfloat16
    assert _close_to_fp32(logits, x, weight)


@pytest.mark.parametrize("n", [32320, 43092])
def test_bf16_head_matches_reference_and_is_batch_invariant(n):
    k = 256
    generator = torch.Generator(device=DEVICE).manual_seed(n)
    weight = (torch.randn(n, k, generator=generator, device=DEVICE) * 0.05).bfloat16()
    layer = SimpleNamespace(weight=weight)
    prepare_bf16_head(layer)
    x = torch.randn(130, k, device=DEVICE).bfloat16()
    full = project_head(layer, x)
    assert _close_to_fp32(full, x, weight)
    for rows in (1, 5, 16, 17, 33, 64):
        assert torch.equal(project_head(layer, x[:rows]), full[:rows]), rows
    assert torch.equal(project_head(layer, x[70:71]), full[70:71])


def test_logits_processor_projects_prepared_heads():
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.head_dtype = None
    processor.use_tilelang_vocab_projection = True
    n, k = 1000, 512
    layer = _packed_layer(_head(n, k, seed=11))
    prepare_packed_head(layer)
    hidden = (torch.randn(2, 3, k, device=DEVICE) * 0.5).bfloat16()
    logits = processor._apply_head(layer, hidden, None)
    expected = project_head(layer, hidden.reshape(-1, k)).reshape(2, 3, n)
    assert torch.equal(logits, expected)


@pytest.mark.parametrize("rows", [1, 6, 48, 130])
def test_packed_head_writes_fp32_logits(rows):
    """FP32 logits are the BF16 head's accumulator unrounded: rounding them gives the
    BF16 head's bits, and they sit closer to an FP64 reference."""
    n, k = 1000, 512
    weight = _head(n, k, seed=5)
    bf16, fp32 = _packed_layer(weight), _packed_layer(weight)
    prepare_packed_head(bf16)
    prepare_packed_head(fp32, torch.float32)
    x = (torch.randn(rows, k, device=DEVICE) * 0.5).bfloat16()
    low, full = project_head(bf16, x), project_head(fp32, x)
    assert full.dtype == torch.float32 and full.shape == (rows, n)
    assert torch.equal(full.bfloat16(), low)
    reference = x.double() @ weight.double().T
    assert (full.double() - reference).abs().max() <= (
        low.double() - reference
    ).abs().max()


def test_bf16_head_writes_fp32_logits_batch_invariant():
    n, k = 32320, 256
    generator = torch.Generator(device=DEVICE).manual_seed(9)
    weight = (torch.randn(n, k, generator=generator, device=DEVICE) * 0.05).bfloat16()
    bf16, fp32 = SimpleNamespace(weight=weight), SimpleNamespace(weight=weight)
    prepare_bf16_head(bf16)
    prepare_bf16_head(fp32, torch.float32)
    x = torch.randn(130, k, device=DEVICE).bfloat16()
    full = project_head(fp32, x)
    assert full.dtype == torch.float32
    assert torch.equal(full.bfloat16(), project_head(bf16, x))
    for rows in (1, 5, 16, 17, 33, 64):
        assert torch.equal(project_head(fp32, x[:rows]), full[:rows]), rows


def test_logits_processor_projects_fp32_heads():
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.head_dtype = torch.float32
    processor.use_tilelang_vocab_projection = True
    n, k = 1000, 512
    layer = _packed_layer(_head(n, k, seed=13))
    prepare_packed_head(layer, torch.float32)
    hidden = (torch.randn(2, 3, k, device=DEVICE) * 0.5).bfloat16()
    logits = processor._apply_head(layer, hidden, None)
    assert logits.dtype == torch.float32
    expected = project_head(layer, hidden.reshape(-1, k)).reshape(2, 3, n)
    assert torch.equal(logits, expected)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_packed_head_greedy_pairs_match_greedy_shard_pair(n, k, out_dtype):
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
        vocab_parallel_pairs,
        vocab_parallel_pairs_from_head,
    )

    weight = _head(n, k, seed=n + 19)
    # Include identical rows inside and across vocabulary tiles to exercise tie-breaking.
    weight[100] = weight[5]
    weight[500] = weight[5]
    layer = _packed_layer(weight)
    prepare_packed_head(layer, out_dtype)
    assert set(layer.tilelang_vocab_greedy) == set(ROW_TILES)
    x = (torch.randn(130, k, device=DEVICE) * 0.5).bfloat16()
    for vocab, vocab_start in ((n, 0), (n - 37, 32320)):
        for rows in (1, 6, 24, 48, 65, 96, 128, 130):
            part = x[:rows].contiguous()
            logits = project_head(layer, part)
            expected = vocab_parallel_pairs(logits, vocab, vocab_start, tp_size=1)
            actual = project_head_greedy_pairs(
                layer, part, vocab=vocab, vocab_start=vocab_start
            )
            from_head = vocab_parallel_pairs_from_head(
                layer, part, vocab, vocab_start, tp_size=1
            )
            assert torch.equal(actual[:, :2], expected[:, :2]), (rows, vocab)
            assert torch.equal(from_head[:, :2], expected[:, :2]), (rows, vocab)


def test_project_head_greedy_pairs_borrows_workspace_and_reuses_layer_scratch(
    monkeypatch,
):
    import vllm.v1.worker.gpu.spec_decode.greedy_verification as gv
    import vllm.v1.worker.workspace as ws_mod
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
        PAIR_WIDTH,
        vocab_parallel_pairs,
        vocab_parallel_pairs_from_head,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    n, k = 1000, 512
    weight = _head(n, k, seed=77)
    layer = _packed_layer(weight)
    prepare_packed_head(layer)
    x = (torch.randn(24, k, device=DEVICE) * 0.5).bfloat16()
    logits = project_head(layer, x)
    expected = vocab_parallel_pairs(logits, n, 0, tp_size=1)

    # 1. Without WorkspaceManager initialized: persistent layer scratch is reused
    # across tile sizes (16 -> 24 -> 6) and when scratch_pairs=True.
    monkeypatch.setattr(ws_mod, "_manager", None)
    p6_a = project_head_greedy_pairs(layer, x[:6], vocab=n, vocab_start=0)
    scratch_6 = layer._tilelang_vocab_greedy_scratch
    assert scratch_6 is not None
    p6_b = project_head_greedy_pairs(layer, x[:6], vocab=n, vocab_start=0)
    assert layer._tilelang_vocab_greedy_scratch[0].data_ptr() == scratch_6[0].data_ptr()
    assert p6_a.data_ptr() != p6_b.data_ptr()
    assert torch.equal(p6_a[:, :2], expected[:6, :2])
    assert torch.equal(p6_b[:, :2], expected[:6, :2])

    p24 = project_head_greedy_pairs(
        layer, x, vocab=n, vocab_start=0, scratch_pairs=True
    )
    assert p24.data_ptr() == layer._tilelang_vocab_greedy_scratch[2].data_ptr()
    assert torch.equal(p24[:, :2], expected[:, :2])

    # 2. With WorkspaceManager initialized: partial_vals, partial_idxs, and
    # TP>1 local pairs are borrowed directly from the active workspace.
    init_workspace_manager(torch.device(DEVICE), num_ubatches=1)
    manager = ws_mod.current_workspace_manager()
    monkeypatch.setattr(
        gv,
        "tensor_model_parallel_all_gather",
        lambda t, dim=-1: t.repeat(1, 3),
    )
    tp3_pairs = vocab_parallel_pairs_from_head(layer, x[:6], n, 0, tp_size=3)
    assert tp3_pairs.shape == (6, PAIR_WIDTH * 3)
    assert torch.equal(tp3_pairs[:, :2], expected[:6, :2])
    ws_base = manager._current_workspaces[0].data_ptr()
    ws_end = ws_base + manager.available_bytes()
    p6_ws = project_head_greedy_pairs(
        layer, x[:6], vocab=n, vocab_start=0, scratch_pairs=True
    )
    assert ws_base <= p6_ws.data_ptr() < ws_end
    assert torch.equal(p6_ws[:, :2], expected[:6, :2])


@pytest.mark.parametrize("n,k", [(512, 256), (2500, 512), (32320, 5120)])
@pytest.mark.parametrize("in_place", [False, True])
def test_tilelang_pack_bf16_weight_round_trips_and_matches_b12x(n, k, in_place):
    weight = _head(n, k, seed=n + k + 41)
    weight[0, :7] = torch.tensor(
        [0.0, -0.0, 1e-30, 3.0, -2.5, 1e-7, 6e-39], dtype=torch.bfloat16, device=DEVICE
    )
    original = weight.clone()
    packed = pack_bf16_weight(weight.clone() if in_place else weight, in_place=in_place)
    assert packed.out_features == n and packed.in_features == k
    assert packed.exceptions > 0
    unpacked = unpack_bf16_weight(packed)
    assert torch.equal(unpacked.view(torch.int16), original.view(torch.int16))

    b12x_packed_mod = pytest.importorskip("b12x.gemm.packed_bf16_vocab_projection")
    ref = b12x_packed_mod.pack(original.clone(), in_place=False)
    assert packed.exponent_base == ref.exponent_base
    assert packed.exceptions == ref.exceptions
    assert torch.equal(packed.values, ref.values)
    assert torch.equal(packed.row_offsets, ref.row_offsets)
    assert torch.equal(packed.exception_rows, ref.exception_rows)
    assert torch.equal(packed.exception_entries, ref.exception_entries)


def test_packed_bf16_lm_head_method_tilelang_without_b12x(monkeypatch):
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.online import packed_bf16 as pbf16_mod
    from vllm.model_executor.layers.quantization.online.packed_bf16 import (
        PackedBf16LmHeadMethod,
        packed_bf16_supported,
    )

    config = VllmConfig(kernel_config=KernelConfig(linear_backend="tilelang"))
    monkeypatch.setattr(
        pbf16_mod, "get_b12x_packed_bf16_vocab_projection", lambda: None
    )
    with set_current_vllm_config(config):
        assert packed_bf16_supported(512, torch.bfloat16)
        method = PackedBf16LmHeadMethod()
        assert method.tilelang
        layer = torch.nn.Module()
        with torch.device(DEVICE):
            method.create_weights(layer, 512, [1000], 512, 1000, torch.bfloat16)
        original = _head(1000, 512, seed=77)
        layer.weight.data.copy_(original)
        method.process_weights_after_loading(layer)
        packed = method.packed_weight(layer)
        assert torch.equal(
            unpack_bf16_weight(packed).view(torch.int16), original.view(torch.int16)
        )
        method.declare_layer_kernels(layer)
        x = (torch.randn(9, 512, device=DEVICE) * 0.5).bfloat16()
        assert _close_to_fp32(project_head(layer, x), x, original)



def _nvfp4_layer(weight: torch.Tensor):
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        FLOAT4_E2M1_MAX,
        dequantize_to_dtype,
    )

    float8_e4m3_max = torch.finfo(torch.float8_e4m3fn).max
    amax = weight.abs().amax().float().clamp_min(1e-8)
    global_scale = (FLOAT4_E2M1_MAX * float8_e4m3_max) / amax
    packed, scales = scaled_fp4_quant(
        weight.contiguous(), global_scale, is_sf_swizzled_layout=False
    )
    weight_global_scale = global_scale.reciprocal().reshape(1).to(torch.float32)
    dequant = dequantize_to_dtype(
        packed,
        scales.view(torch.float8_e4m3fn),
        weight_global_scale,
        dtype=torch.float32,
        block_size=16,
        swizzle=False,
    )
    layer = SimpleNamespace(
        weight=packed,
        weight_scale=scales,
        weight_global_scale=weight_global_scale,
        nvfp4_in_features=int(weight.shape[1]),
    )
    return layer, dequant


@pytest.mark.parametrize("n,k", [(1000, 256), (32320, 256), (1000, 512)])
def test_nvfp4_head_matches_dequantized_reference_and_is_batch_invariant(n, k):
    generator = torch.Generator(device=DEVICE).manual_seed(n + k + 23)
    weight = (torch.randn(n, k, generator=generator, device=DEVICE) * 0.05).bfloat16()
    layer, dequant_weight = _nvfp4_layer(weight)
    prepare_nvfp4_head(layer)
    assert set(layer.tilelang_vocab) == set(ROW_TILES)
    x = (torch.randn(130, k, generator=generator, device=DEVICE) * 0.5).bfloat16()
    full = project_head(layer, x)
    assert _close_to_fp32(full, x, dequant_weight)
    for rows in (1, 5, 16, 17, 33, 64, 96, 128):
        assert torch.equal(project_head(layer, x[:rows].contiguous()), full[:rows]), rows
    assert torch.equal(project_head(layer, x[70:71].contiguous()), full[70:71])


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_nvfp4_head_greedy_pairs_match_greedy_shard_pair(out_dtype):
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
        vocab_parallel_pairs,
        vocab_parallel_pairs_from_head,
    )

    n, k = 1000, 256
    generator = torch.Generator(device=DEVICE).manual_seed(109)
    weight = (torch.randn(n, k, generator=generator, device=DEVICE) * 0.05).bfloat16()
    weight[100] = weight[5]
    weight[500] = weight[5]
    layer, _ = _nvfp4_layer(weight)
    prepare_nvfp4_head(layer, out_dtype)
    assert set(layer.tilelang_vocab_greedy) == set(ROW_TILES)
    x = (torch.randn(96, k, generator=generator, device=DEVICE) * 0.5).bfloat16()
    for vocab, vocab_start in ((n, 0), (n - 37, 32320)):
        for rows in (1, 6, 24, 48, 65, 96):
            part = x[:rows].contiguous()
            logits = project_head(layer, part)
            expected = vocab_parallel_pairs(logits, vocab, vocab_start, tp_size=1)
            actual = project_head_greedy_pairs(
                layer, part, vocab=vocab, vocab_start=vocab_start
            )
            from_head = vocab_parallel_pairs_from_head(
                layer, part, vocab, vocab_start, tp_size=1
            )
            assert torch.equal(actual[:, :2], expected[:, :2]), (rows, vocab)
            assert torch.equal(from_head[:, :2], expected[:, :2]), (rows, vocab)


def test_nvfp4_online_linear_method_tilelang_without_b12x(monkeypatch):
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear.nvfp4.b12x import B12xNvFp4LinearKernel
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.quantization.online.nvfp4 import (
        Nvfp4OnlineLinearMethod,
    )

    monkeypatch.setattr(
        B12xNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, *_args, **_kwargs: (False, "b12x disabled")),
    )
    config = VllmConfig(kernel_config=KernelConfig(linear_backend="tilelang"))
    with set_current_vllm_config(config):
        method = Nvfp4OnlineLinearMethod()
        assert method.tilelang and method.kernel is None
        layer = torch.nn.Module()
        generator = torch.Generator(device=DEVICE).manual_seed(83)
        original = (
            torch.randn(1000, 256, generator=generator, device=DEVICE) * 0.05
        ).bfloat16()
        layer.register_parameter(
            "weight", torch.nn.Parameter(original.clone(), requires_grad=False)
        )
        method.process_weights_after_loading(layer)
        method.declare_layer_kernels(layer)

        _, dequant = _nvfp4_layer(original)
        x = (torch.randn(2, 5, 256, generator=generator, device=DEVICE) * 0.5).bfloat16()
        out = method.apply(layer, x)
        assert out.shape == (2, 5, 1000) and out.dtype == torch.bfloat16
        assert _close_to_fp32(out.reshape(-1, 1000), x.reshape(-1, 256), dequant)

        processor = LogitsProcessor.__new__(LogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.head_dtype = None
        processor.use_tilelang_vocab_projection = True
        logits = processor._apply_head(layer, x, None)
        assert torch.equal(logits, out)


@pytest.mark.parametrize("n", [32320, 43092])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_packed_head_markov_greedy_pairs_match_greedy_markov_shard_pair(n, out_dtype):
    from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
        greedy_markov_shard_pair,
        scratch_shape,
    )

    k = 256
    weight = _head(n, k, seed=n + 303)
    weight[100] = weight[5]
    weight[500] = weight[5]
    layer = _packed_layer(weight)
    layer.tilelang_max_rows = 64
    layer.tilelang_markov_head = True
    prepare_packed_head(layer, out_dtype)
    assert hasattr(layer, "tilelang_vocab_markov_greedy")
    generator = torch.Generator(device=DEVICE).manual_seed(n + 404)
    x = torch.randn(64, k, generator=generator, device=DEVICE, dtype=torch.bfloat16)
    base_3d = torch.randn(
        64, 5, n, generator=generator, device=DEVICE, dtype=out_dtype
    )
    for vocab, vocab_start in ((n, 0), (n - 37, 32320)):
        for rows in (1, 5, 16, 17, 33, 64):
            part = x[:rows].contiguous()
            base_step = base_3d[:rows, 2]
            bias = project_head(layer, part)
            expected = torch.empty((rows, 4), dtype=torch.float32, device=DEVICE)
            shape = scratch_shape(rows, vocab)
            values = torch.empty(shape, dtype=torch.float32, device=DEVICE)
            indices = torch.empty(shape, dtype=torch.int32, device=DEVICE)
            greedy_markov_shard_pair(
                base_step[:, :vocab],
                bias[:, :vocab],
                vocab_start,
                expected,
                values,
                indices,
            )
            actual = torch.empty((rows, 4), dtype=torch.float32, device=DEVICE)
            project_head_markov_greedy_pairs(
                layer,
                part,
                base_step,
                vocab=vocab,
                vocab_start=vocab_start,
                out=actual,
            )
            assert torch.equal(actual[:, :2], expected[:, :2]), (rows, vocab, out_dtype)
