# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TileLang attention layer's staging, binding and context KV writes."""

from types import SimpleNamespace

import pytest
import torch


def _tilelang_layer(layer_id):
    """A TileLang attention layer with the geometry its bind reads (no weights)."""
    from vllm.models.deepseek_v4_1.tilelang.attention import (
        DeepseekV41TileLangAttention,
    )

    module = DeepseekV41TileLangAttention.__new__(DeepseekV41TileLangAttention)
    torch.nn.Module.__init__(module)
    module.prefix = f"model.layers.{layer_id}.attn"
    module.layer_id, module.candidate_source_layer = layer_id, 3
    module.capacity, module.swa_width, module.compress_ratio = 300, 128, 1
    module._main_page = module._index_page = 64
    module._main_width = module._index_width = 16
    module.is_ced_decoder, module.is_index_source = False, True
    module.topk_indices_buffer = None
    module.indexer = SimpleNamespace(heads=32)
    module.compressor = None
    module.config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        speculative_config=None,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=24),
    )
    module.rotary_emb = SimpleNamespace(cos_sin_cache=torch.zeros(16, 64))
    module._ready = False
    return module


def test_attention_offers_b12x_preparation_nothing():
    # B12X's preparation used to compile and launch helper plans for this layer
    # while memory profiling; the layer declares TileLang kernels only.
    module = _tilelang_layer(layer_id=3)
    assert getattr(module, "b12x_preparation_provider", None) is None
    assert not hasattr(module, "get_b12x_preparation_units")
    assert not module._ready


@pytest.mark.parametrize(
    "layer_id,staging",
    [
        (2, ["topk_indices_buffer", "_index_pages", "_active", "_short_index_pages"]),
        (
            3,
            [
                "topk_indices_buffer",
                "_index_pages",
                "_active",
                "_candidates",
                "_candidate_lens",
            ],
        ),
    ],
)
def test_tilelang_binding_prepares_the_attention_staging(
    monkeypatch, layer_id, staging
):
    # The layer's kernel binding allocates the staging, before memory profiling.
    module = _tilelang_layer(layer_id)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    module._build_tilelang_plans = lambda: "plans"
    module._attention_geometry()  # declare_kernels() runs it before binding
    with torch.inference_mode():  # the worker prepares kernels inside it
        module._bind_plans()
    assert module._ready and module._tilelang_plans == "plans"
    assert [name for name, _, _ in module._staging_specs] == staging
    for name, shape, dtype in module._staging_specs:
        tensor = getattr(module, name)
        assert tuple(tensor.shape) == shape and tensor.dtype == dtype
        assert not tensor.is_inference()  # updated in place while serving
    assert module._active.tolist() == [16 * 64]


@pytest.mark.parametrize(
    "num_verification_tokens,decode_capacity",
    [
        (5, 4 * 11),  # parallel drafting's two spans
        (16, 4 * 17),  # copy drafts: every request at the reorder threshold
    ],
)
def test_decode_arena_holds_every_request_at_the_reorder_threshold(
    num_verification_tokens, decode_capacity
):
    # A step of short requests (16 of 13-14 tokens under copy drafts' threshold
    # of 17) is all decode rows; its metadata must not be cut to fewer rows.
    module = _tilelang_layer(layer_id=3)
    module.config.speculative_config = SimpleNamespace(
        num_speculative_tokens=5,
        parallel_drafting=True,
        num_verification_tokens=num_verification_tokens,
    )
    module._attention_geometry()
    assert module._decode_capacity == decode_capacity
    swa_spec = module._attention_workspace_specs["decode"][0]
    assert swa_spec == ((decode_capacity, module.swa_width), torch.int32)


def test_context_kv_write_uses_the_tilelang_writer(monkeypatch):
    from vllm.models.deepseek_v4_1.tilelang import cache_writer, rope

    module = _tilelang_layer(layer_id=3)
    writes = []
    monkeypatch.setattr(rope, "rope_", lambda *args: None)
    monkeypatch.setattr(
        cache_writer, "write_cache", lambda *args, **kwargs: writes.append(kwargs)
    )
    module.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.empty(2, 8, dtype=torch.uint8), block_size=64
    )
    module._ready = True
    kv = torch.zeros(1, 512, dtype=torch.bfloat16)
    slots = torch.zeros(1, dtype=torch.int64)
    module.insert_context_kv(kv, slots, slots)
    assert writes == [dict(page_size=64, cache_kind="swa")]


def test_global_preparation_populates_caches_before_forward():
    from vllm.models.deepseek_v4_1 import attention

    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = torch.arange(16, dtype=torch.float32, device=device).view(4, 4)
    positions = torch.arange(4, device=device, dtype=torch.int64)
    backing = torch.zeros((8, 4), device=device)

    class CacheConsumer(torch.nn.Module):
        prepare_global_kv = attention.DeepseekV4Attention.prepare_global_kv
        forward = attention.DeepseekV4Attention.forward

        def __init__(self):
            super().__init__()
            self.prefix = "ced_cache_consumer"
            self.compressor = SimpleNamespace(state_cache=None)
            self.kv_cache = backing[:4]
            self.indexer = SimpleNamespace(
                k_cache=SimpleNamespace(kv_cache=backing[4:])
            )

        def _prepare_global_kv(self, positions, hidden):
            self.kv_cache.copy_(hidden * 2)
            self.indexer.k_cache.kv_cache.copy_(hidden * 3)

        def _forward(self, positions, hidden):
            return self.kv_cache + self.indexer.k_cache.kv_cache

    layer = CacheConsumer()

    def run(hidden):
        layer.prepare_global_kv(positions, hidden)
        return layer(positions, hidden)

    for factor in (1, -2):
        hidden = source * factor
        output = run(hidden)
        torch.testing.assert_close(output, hidden * 5)
        torch.testing.assert_close(backing, torch.cat((hidden * 2, hidden * 3)))


def test_front_is_one_kernel_across_row_counts(monkeypatch):
    """With cache pages, the front after the fused projection is ``attention_front``'s
    one launch across decode and prefill row counts, at the SWA slots and page size,
    and Q-B gets its cast; memory profiling (no cache pages) and breakable graphs run
    the separate kernels."""
    from vllm.models.deepseek_v4_1 import attention
    from vllm.models.deepseek_v4_1.tilelang import front

    module = _tilelang_layer(layer_id=3)
    module.q_lora_rank = 1280
    module.q_norm = SimpleNamespace(weight=torch.ones(1280), variance_epsilon=1e-6)
    module.kv_norm = SimpleNamespace(weight=torch.ones(512), variance_epsilon=1e-6)
    module.swa_cache_layer = SimpleNamespace(
        prefix="swa", kv_cache=torch.empty(2, 8, dtype=torch.uint8), block_size=128
    )
    launches = []
    monkeypatch.setattr(front, "attention_front", lambda *args: launches.append(args))
    slots = torch.arange(600, dtype=torch.int64)
    metadata = {"swa": SimpleNamespace(slot_mapping=slots)}
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    for rows in (6, 65, 512):
        positions = torch.zeros(rows, dtype=torch.int64)
        qr, kv, cast = module._front(torch.zeros(rows, 1792), positions)
        args = launches[-1]
        assert args[7].tolist() == list(range(rows)) and args[8] == 128
        assert args[9] is qr and args[12] is kv
        assert cast.rows is qr and args[10] is cast.values and args[11] is cast.scales
        assert qr.shape == (rows, 1280) and kv.shape == (rows, 512)
    assert module._front(torch.zeros(0, 1792), torch.zeros(0, dtype=torch.int64)) is None
    metadata = None
    assert module._front(torch.zeros(6, 1792), torch.zeros(6, dtype=torch.int64)) is None
    assert len(launches) == 3


def test_index_query_reuses_q_cast_and_fuses_rope_quantize(monkeypatch):
    """``_index_query`` feeds ``indexer.wq_b`` from ``q_cast`` (no second ``cast_mxfp8``)
    and passes ``positions[first:last]`` straight into ``_quantize_index_query`` without
    a separate ``_rope`` call."""
    from vllm.models.deepseek_v4_1 import kernels
    from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows

    module = _tilelang_layer(layer_id=3)
    seen_provided = []
    quant_calls = []

    def fake_wqb(x):
        seen_provided.append(x.shape)
        return torch.zeros((x.shape[0], 32 * 128), dtype=torch.bfloat16)

    module.indexer = SimpleNamespace(
        heads=32,
        wq_b=fake_wqb,
        weights_proj=lambda h: torch.ones((h.shape[0], 32), dtype=torch.bfloat16),
    )
    monkeypatch.setattr(
        kernels,
        "cast_mxfp8",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected cast_mxfp8")),
    )
    module._rope = lambda *args: (_ for _ in ()).throw(
        AssertionError("unexpected separate _rope")
    )
    module._quantize_index_query = (
        lambda mode, iq, data, scale, positions=None: quant_calls.append(
            (mode, tuple(iq.shape), positions.tolist())
        )
    )
    qr = torch.zeros((12, 1280), dtype=torch.bfloat16)
    q_cast = MXFP8Rows(
        qr,
        torch.zeros((64, 1280), dtype=torch.float8_e4m3fn),
        torch.zeros((64, 40), dtype=torch.uint8),
    )
    positions = torch.arange(100, 112, dtype=torch.int64)
    hidden = torch.zeros((12, 5120), dtype=torch.bfloat16)
    swa = SimpleNamespace(is_decode=False)
    module._index_query(qr, positions, hidden, None, 4, 9, swa, q_cast)
    assert seen_provided == [(5, 1280)]
    assert quant_calls == [("prefill", (5, 32, 128), [104, 105, 106, 107, 108])]


def test_prepare_global_kv_fuses_index_key_and_latent_writes(monkeypatch):
    from vllm.models.deepseek_v4_1 import attention
    from vllm.models.deepseek_v4_1.tilelang import attention as tl_attention
    from vllm.models.deepseek_v4_1.tilelang import cache_writer

    module = _tilelang_layer(layer_id=0)
    module._ready = True
    module._owner = lambda: module
    module.kv_cache = torch.empty(2, 8, dtype=torch.uint8)
    latent = torch.zeros(4, 512, dtype=torch.bfloat16)
    slots = torch.arange(4, dtype=torch.int64)
    module.compressor = lambda hidden, main, state: (latent, slots)
    module.compressor.state_cache = None
    raw_key = torch.zeros(4, 128, dtype=torch.bfloat16)
    module.indexer = SimpleNamespace(
        heads=32,
        wk=lambda x: raw_key,
        k_norm=SimpleNamespace(weight=torch.ones(128), variance_epsilon=1e-6),
        k_cache=SimpleNamespace(
            prefix="k_cache", kv_cache=torch.empty(2, 8, dtype=torch.uint8)
        ),
    )
    idx_slots = torch.arange(10, 14, dtype=torch.int64)
    metadata = {
        module.prefix: SimpleNamespace(),
        "k_cache": SimpleNamespace(slot_mapping=idx_slots),
    }
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    key_writes, cache_writes = [], []
    monkeypatch.setattr(
        tl_attention,
        "norm_rope_write_index_keys",
        lambda *args: key_writes.append(args),
    )
    monkeypatch.setattr(
        cache_writer,
        "write_cache",
        lambda *args, **kwargs: cache_writes.append((args, kwargs)),
    )
    positions = torch.arange(4, dtype=torch.int64)
    module._prepare_global_kv(positions, torch.zeros(4, 5120, dtype=torch.bfloat16))
    assert len(key_writes) == 1
    assert key_writes[0][0] is raw_key and key_writes[0][3] is positions
    assert key_writes[0][6].tolist() == [10, 11, 12, 13]
    assert len(cache_writes) == 1
    assert cache_writes[0][0][0] is latent
    assert cache_writes[0][1]["positions"] is positions
    assert cache_writes[0][1]["cache_kind"] == "indexed"


def test_attention_forward_releases_intermediates_before_o_proj(monkeypatch):
    """``_forward`` releases ``qr_kv``, ``q_cast``, ``qr``, ``q``, ``kv`` and
    ``_index_query``'s ``iq`` before ``_o_proj`` allocates the output projection."""
    import weakref

    from vllm.models.deepseek_v4_1 import attention, kernels
    from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows

    module = _tilelang_layer(layer_id=3)
    module._ready = True
    module.q_lora_rank = 16
    module.n_local_heads = 2
    module._l2pf_wo = None
    module._l2pf_comp = None
    module.swa_cache_layer = SimpleNamespace(prefix="swa")
    refs = {}

    def track(name, tensor):
        refs[name] = weakref.ref(tensor)
        return tensor

    module.fused_wqa_wkv = lambda h: (track("qr_kv", torch.ones(4, 528)), None)
    module._front = lambda qr_kv, pos: None
    module.q_norm = lambda x: track("qr", x.clone())
    module.kv_norm = lambda x: track("kv", x.clone())
    monkeypatch.setattr(
        kernels,
        "cast_mxfp8",
        lambda rows: MXFP8Rows(
            rows,
            track("q_cast", torch.ones(4, 16, dtype=torch.uint8)),
            torch.ones(4, 4, dtype=torch.uint8),
        ),
    )
    module.wq_b = lambda qr: track("q", torch.ones(4, 2 * 512))
    module._rope = lambda x, pos, role: x
    module._cache_context_kv = lambda kv, pos: None
    module.indexer = SimpleNamespace(
        heads=4,
        wq_b=lambda qr: track("iq", torch.ones(4, 4 * 128)),
        weights_proj=lambda h: (
            # iq must already be released when weights_proj runs
            assert_dead("iq"),
            torch.ones(4, 4),
        )[1],
    )
    module._quantize_index_query = (
        lambda mode, iq, data, scale, positions=None: None
    )

    def assert_dead(*names):
        for name in names:
            assert refs[name]() is None, f"{name} is still alive"

    def check_forward_mqa(q, kv, pos, output, *, index_query=None):
        assert_dead("qr_kv", "q_cast", "qr", "iq", "hidden")
        output.zero_()

    def check_o_proj(output, sp_rows=None):
        assert_dead("qr_kv", "q_cast", "qr", "iq", "hidden", "q", "kv")
        return torch.zeros(4, 32)

    module.forward_mqa = check_forward_mqa
    module._o_proj = check_o_proj
    metadata = {"swa": SimpleNamespace(is_decode=False)}
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    positions = torch.arange(4)
    out = module._forward(positions, track("hidden", torch.ones(4, 32)))
    assert out.shape == (4, 32)


def test_forward_mqa_shares_index_pages_without_mixed_step_clobber(monkeypatch):
    """Single-chunk index page tables are shared across layers sharing ``im``,
    and mixed decode+prefill steps never clobber the cached decode slice."""
    from vllm.models.deepseek_v4_1 import attention
    from vllm.models.deepseek_v4_1.tilelang import metadata as tl_metadata

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def make_layer(layer_id):
        layer = _tilelang_layer(layer_id=layer_id)
        layer.compress_ratio = 0
        layer.candidate_source_layer = 99
        layer._attention_geometry()
        layer._short_index_shape = None
        layer.topk_indices_buffer = torch.zeros(128, 512, dtype=torch.int32)
        layer.INDEX_CHUNK = 64
        layer._index_pages = torch.full((64, 16), -1, dtype=torch.int32)
        layer._active = torch.tensor([16 * 64], dtype=torch.int32)
        layer._ready = True
        layer.swa_cache_layer = SimpleNamespace(prefix="swa")
        layer.indexer = SimpleNamespace(
            heads=4,
            k_cache=SimpleNamespace(prefix="idx"),
        )
        layer._check_attention_rows = lambda *args, **kwargs: None
        layer._attend_split = lambda *args, **kwargs: None
        return layer

    l0, l1 = make_layer(0), make_layer(1)
    im = SimpleNamespace(
        req_id_per_token=torch.arange(128, dtype=torch.int32),
        cache_lengths=torch.full((128,), 128, dtype=torch.int32),
        block_table=torch.arange(128 * 16, dtype=torch.int32).view(128, 16),
        max_seq_len=32768,
    )

    pages_calls = []

    def fake_pages(reqs, lengths, table, out, offset, rows, read_width, *, page):
        pages_calls.append((offset, rows))
        out[:rows] = table[offset : offset + rows, : out.shape[1]]

    monkeypatch.setattr(tl_metadata, "pages", fake_pages)

    seen = {0: [], 1: []}
    for lid, layer in ((0, l0), (1, l1)):
        layer._index_chunk = (
            lambda mode, count, *, page_table, _lid=lid, **kw: seen[_lid].append(
                (mode, count, page_table.clone())
            )
        )

    # Case 1: Mixed step with total rows (6) <= INDEX_CHUNK (64).
    # Decode uses _index_pages[:2] and rest uses _index_pages[2:6]; both are cached
    # on l0 and reused on l1 without clobbering.
    ctx = SimpleNamespace(
        attn_metadata={
            "swa": SimpleNamespace(is_decode=False, num_decode_tokens=2),
            "idx": im,
        }
    )
    monkeypatch.setattr(attention, "get_forward_context", lambda: ctx)
    q = torch.zeros(6, 2, 512)
    iq = (
        torch.zeros(6, 4, 64, dtype=torch.uint8),
        torch.zeros(6, 4, 4, dtype=torch.uint8),
        torch.zeros(6, 4),
        None,
        None,
    )
    l0.forward_mqa(q, None, torch.arange(6), q.clone(), index_query=iq)
    l1.forward_mqa(q, None, torch.arange(6), q.clone(), index_query=iq)
    assert pages_calls == [(0, 2), (2, 4)]
    assert torch.equal(seen[0][0][2], im.block_table[0:2])
    assert torch.equal(seen[0][1][2], im.block_table[2:6])
    assert torch.equal(seen[1][0][2], im.block_table[0:2])
    assert torch.equal(seen[1][1][2], im.block_table[2:6])

    # Case 2: Mixed step with total rows (70) > INDEX_CHUNK (64) and shared buffer.
    # Decode slice must not be cached across rest multi-chunk overwrite of _index_pages.
    pages_calls.clear()
    seen[0].clear()
    seen[1].clear()
    ctx2 = SimpleNamespace(
        attn_metadata={
            "swa": SimpleNamespace(is_decode=False, num_decode_tokens=2),
            "idx": im,
        }
    )
    monkeypatch.setattr(attention, "get_forward_context", lambda: ctx2)
    q70 = torch.zeros(70, 2, 512)
    iq70 = (
        torch.zeros(70, 4, 64, dtype=torch.uint8),
        torch.zeros(70, 4, 4, dtype=torch.uint8),
        torch.zeros(70, 4),
        None,
        None,
    )
    l0.forward_mqa(q70, None, torch.arange(70), q70.clone(), index_query=iq70)
    l1.forward_mqa(q70, None, torch.arange(70), q70.clone(), index_query=iq70)
    assert torch.equal(seen[0][0][2], im.block_table[0:2])
    assert torch.equal(seen[1][0][2], im.block_table[0:2])


def test_prepare_global_kv_fuses_compressor_indexed_kv_and_wk_partials(monkeypatch):
    """When ``self.compressor`` is a ``DeepseekCompressor`` with a full-width
    ``kv_cache`` and ``indexer.wk`` exposes ``tilelang_partials``,
    ``_prepare_global_kv`` fuses the ``indexed`` KV write into ``compressor(...)``
    and passes 3D split-K partials from ``project_bf16_partials`` to
    ``_write_index_keys`` without launching ``_write_v41_cache``."""
    from vllm.models.deepseek_v4_1 import attention
    from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
    from vllm.models.deepseek_v4_1.tilelang import attention as tl_attention
    from vllm.models.deepseek_v4_1.tilelang import linear as tl_linear

    module = _tilelang_layer(layer_id=0)
    module._ready = True
    module._owner = lambda: module
    module._main_page = 64
    module.kv_cache = torch.empty(2, 64 * 288, dtype=torch.uint8)
    latent = torch.zeros(4, 512, dtype=torch.bfloat16)
    slots = torch.arange(4, dtype=torch.int64)
    comp_calls = []

    comp = DeepseekCompressor.__new__(DeepseekCompressor)
    torch.nn.Module.__init__(comp)
    comp.state_cache = None
    monkeypatch.setattr(
        DeepseekCompressor,
        "forward",
        lambda self, hidden, main, state=None, **kw: (
            comp_calls.append(kw),
            (latent, slots),
        )[1],
    )
    module.compressor = comp

    partials = torch.zeros(4, 4, 128, dtype=torch.float32)
    monkeypatch.setattr(
        tl_linear,
        "project_bf16_partials",
        lambda layer, x: partials,
    )
    wk = SimpleNamespace(tilelang_partials={16: object()})
    module.indexer = SimpleNamespace(
        heads=32,
        wk=wk,
        k_norm=SimpleNamespace(weight=torch.ones(128), variance_epsilon=1e-6),
        k_cache=SimpleNamespace(
            prefix="k_cache", kv_cache=torch.empty(2, 8, dtype=torch.uint8)
        ),
    )
    idx_slots = torch.arange(10, 14, dtype=torch.int64)
    metadata = {
        module.prefix: SimpleNamespace(),
        "k_cache": SimpleNamespace(slot_mapping=idx_slots),
    }
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    key_writes = []
    monkeypatch.setattr(
        tl_attention,
        "norm_rope_write_index_keys",
        lambda *args: key_writes.append(args),
    )
    monkeypatch.setattr(
        attention.DeepseekV4Attention,
        "_write_v41_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected separate _write_v41_cache")
        ),
    )
    positions = torch.arange(4, dtype=torch.int64)
    module._prepare_global_kv(positions, torch.zeros(4, 5120, dtype=torch.bfloat16))
    assert len(comp_calls) == 1
    assert comp_calls[0]["kv_cache"] is module.kv_cache
    assert comp_calls[0]["token_positions"] is positions
    assert comp_calls[0]["kv_page_size"] == 64
    assert len(key_writes) == 1
    assert key_writes[0][0] is partials


@pytest.mark.parametrize("tp_size,expected_part", [(1, 1792), (3, 608), (4, 448)])
def test_padded_merged_column_parallel_linear_shards_contiguously(
    monkeypatch, tp_size, expected_part
):
    """``_PaddedMergedColumnParallelLinear`` stacks ``[wq_a (1280), wkv (512)]`` into one
    contiguous 1792-row logical matrix, pads to a multiple of ``tp_size * 32`` (1824 at
    TP3 with zero weights and 0x7F unit UE8M0 scales in the trailing 32 rows), and
    all-gathers + slices back to 1792 columns."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers import linear as linear_mod
    from vllm.models.deepseek_v4_1.attention import _PaddedMergedColumnParallelLinear
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV41FP8Config

    quant_config = DeepseekV41FP8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[32, 32]
    )
    k = 64
    wq_a = torch.arange(1280 * k, dtype=torch.uint8).reshape(1280, k).view(
        torch.float8_e4m3fn
    )
    wkv = torch.arange(512 * k, dtype=torch.uint8).reshape(512, k).view(
        torch.float8_e4m3fn
    )
    sq_a = (
        torch.arange(40 * 2, dtype=torch.uint8)
        .reshape(40, 2)
        .clamp(1, 250)
        .view(torch.float8_e8m0fnu)
    )
    skv = (
        torch.arange(16 * 2, dtype=torch.uint8)
        .reshape(16, 2)
        .clamp(1, 250)
        .view(torch.float8_e8m0fnu)
    )
    full_w = torch.cat([wq_a.view(torch.uint8), wkv.view(torch.uint8)], dim=0)
    full_s = torch.cat([sq_a.view(torch.uint8), skv.view(torch.uint8)], dim=0)

    ranks = []
    with set_current_vllm_config(VllmConfig()):
        for rank in range(tp_size):
            monkeypatch.setattr(
                parallel_state,
                "_TP",
                SimpleNamespace(rank_in_group=rank, world_size=tp_size),
            )
            layer = _PaddedMergedColumnParallelLinear(
                k,
                [1280, 512],
                bias=False,
                gather_output=True,
                quant_config=quant_config,
                tp_rank=rank,
                tp_size=tp_size,
                prefix=f"test.r{rank}.fused_wqa_wkv",
            )
            assert layer.output_size_per_partition == expected_part
            layer.weight.weight_loader(layer.weight, wq_a, 0)
            layer.weight.weight_loader(layer.weight, wkv, 1)
            layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, sq_a, 0)
            layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, skv, 1)
            ranks.append(layer)

    cat_w = torch.cat([r.shard_weight.view(torch.uint8) for r in ranks], dim=0)
    cat_s = torch.cat([r.shard_weight_scale_inv.view(torch.uint8) for r in ranks], dim=0)
    assert torch.equal(cat_w[:1792], full_w)
    assert torch.equal(cat_s[:56], full_s)
    for r in ranks:
        assert torch.equal(r.full_weight.view(torch.uint8), full_w)
        assert torch.equal(r.full_weight_scale_inv.view(torch.uint8), full_s)
        assert r.shard_weight.data_ptr() == (
            r.weight.data_ptr() + r.tp_rank * expected_part * k
        )
    if tp_size == 3:
        assert not cat_w[1792:].any()
        assert (cat_s[56:] == 0x7F).all()

    for r in ranks:
        r.quant_method = SimpleNamespace(
            apply=lambda lyr, x, bias=None: (
                x
                @ (
                    lyr.shard_weight
                    if x.shape[0] <= 64
                    else lyr.full_weight
                )
                .view(torch.uint8)
                .float()
                .T
            )
        )
    gather_calls = []
    monkeypatch.setattr(
        linear_mod,
        "tensor_model_parallel_all_gather",
        lambda local_out: (
            gather_calls.append(local_out.shape[0]),
            torch.cat(
                [
                    r.quant_method.apply(r, torch.ones(local_out.shape[0], k))
                    for r in ranks
                ],
                dim=-1,
            ),
        )[1],
    )
    out, bias = ranks[0](torch.ones(5, k))
    assert bias is None
    assert out.shape == (5, 1792)
    assert gather_calls == ([5] if tp_size > 1 else [])
    if tp_size == 3:
        assert out.stride() == (1824, 1)
    expected = torch.ones(5, k) @ full_w.float().T
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    # Prefill / wide-decode (rows > 64) runs the full view locally without all-gather.
    gather_calls.clear()
    out_prefill, _ = ranks[0](torch.ones(65, k))
    assert out_prefill.shape == (65, 1792)
    assert gather_calls == []
    torch.testing.assert_close(
        out_prefill, torch.ones(65, k) @ full_w.float().T, rtol=0, atol=0
    )


def test_sp_prefill_with_sharded_fused_wqa_wkv_gathers_mxfp8(monkeypatch):
    """When ``fused_wqa_wkv`` is column-sharded across TP (``tp_size > 1``), SP prefill
    without a compressor gathers the already-cast local ``MXFP8Rows`` and local
    ``indexer.weights_proj`` in one ``gather_mxfp8`` call, skipping ``sp_rows.gather``
    and a second ``cast_mxfp8`` on ``hidden_states``."""
    from vllm.models.deepseek_v4_1 import attention, kernels, sp_prefill
    from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows, provided_mxfp8

    module = _tilelang_layer(layer_id=5)
    module._ready = True
    module.q_lora_rank = 16
    module.n_local_heads = 2
    module._l2pf_wo = None
    module._l2pf_comp = None
    module.swa_cache_layer = SimpleNamespace(prefix="swa")

    local_x = torch.ones((4, 32), dtype=torch.bfloat16)
    local_q = MXFP8Rows(
        rows=local_x,
        values=torch.ones((64, 32), dtype=torch.uint8),
        scales=torch.ones((64, 1), dtype=torch.uint8),
    )
    gathered_x = torch.ones((10, 32), dtype=torch.bfloat16)
    gathered_q = MXFP8Rows(
        rows=gathered_x,
        values=torch.ones((64, 32), dtype=torch.uint8),
        scales=torch.ones((64, 1), dtype=torch.uint8),
    )
    gathered_w = torch.full((10, 4), 3.0, dtype=torch.bfloat16)
    events = []

    class _FakeSPRows:
        num_tokens = 10
        start = 4
        real_rows = 4

        def gather(self, _):
            raise AssertionError("BF16 gather should not run when local MXFP8 is provided")

        def gather_mxfp8(self, x_q, *extra):
            events.append(("gather_mxfp8", x_q is local_q, len(extra)))
            return (gathered_q, gathered_w)

    def sharded_wqa_wkv(h):
        events.append(("fused_wqa_wkv", h is gathered_x, provided_mxfp8(h) is gathered_q))
        return torch.ones((10, 16 + 512), dtype=torch.bfloat16), None

    sharded_wqa_wkv.tp_size = 4
    module.fused_wqa_wkv = sharded_wqa_wkv
    module._front = lambda qr_kv, pos: None
    module.q_norm = lambda x: x.contiguous()
    module.kv_norm = lambda x: x.contiguous()
    monkeypatch.setattr(
        kernels,
        "cast_mxfp8",
        lambda rows: (
            events.append(("cast_mxfp8", tuple(rows.shape))),
            MXFP8Rows(
                rows,
                torch.ones((64, rows.shape[1]), dtype=torch.uint8),
                torch.ones((64, 1), dtype=torch.uint8),
            ),
        )[1],
    )
    module.wq_b = lambda qr: torch.ones((qr.shape[0], 2 * 512), dtype=torch.bfloat16)
    module._rope = lambda x, pos, role: x
    module._cache_context_kv = lambda kv, pos: None
    module.indexer = SimpleNamespace(
        heads=4,
        wq_b=lambda qr: torch.ones((qr.shape[0], 4 * 128), dtype=torch.bfloat16),
        weights_proj=lambda h: (
            events.append(("weights_proj", tuple(h.shape))),
            torch.ones((h.shape[0], 4), dtype=torch.bfloat16),
        )[1],
    )
    module._quantize_index_query = lambda mode, iq, data, scale, positions=None: None
    seen_index_weights = []

    def check_forward_mqa(q, kv, pos, output, *, index_query=None):
        seen_index_weights.append(index_query[2])
        output.zero_()

    module.forward_mqa = check_forward_mqa
    module._o_proj = lambda output, sp_rows=None: torch.zeros((4, 32), dtype=torch.bfloat16)
    metadata = {"swa": SimpleNamespace(is_decode=False)}
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=metadata),
    )
    monkeypatch.setattr(sp_prefill, "current", lambda: _FakeSPRows())
    with kernels.mxfp8_inputs(local_q):
        out = module._forward(torch.arange(10), local_x)
    assert out.shape == (4, 32)
    assert events == [
        ("weights_proj", (4, 32)),
        ("gather_mxfp8", True, 1),
        ("fused_wqa_wkv", True, True),
        ("cast_mxfp8", (10, 16)),
    ]
    assert len(seen_index_weights) == 1
    torch.testing.assert_close(seen_index_weights[0], gathered_w[4:8], rtol=0, atol=0)


def test_tp3_rank2_trims_zero_padded_output_group(monkeypatch):
    """On TP3 (8 output groups padded to 9), rank 2 trims its zero-padded 3rd group
    (heads 16..23) in ``setup_wo_projection`` before kernel declaration, while ranks
    0 and 1 keep all 3 real groups (24 heads)."""
    from torch import nn

    from vllm.models.deepseek_v4_1 import attention
    from vllm.models.deepseek_v4_1.tilelang import wo

    monkeypatch.setattr(
        wo,
        "TileLangWOProjection",
        lambda wo_a, wo_b, groups: SimpleNamespace(
            wo_a=wo_a, wo_b=wo_b, groups=groups
        ),
    )

    def make_layer():
        module = _tilelang_layer(layer_id=0)
        module.config.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(original_o_groups=8, o_groups=9)
        )
        module.n_groups = 9
        module.n_local_groups = 3
        module.n_local_heads = 24
        module.head_dim = 512
        module.o_lora_rank = 1024
        module.hidden_size = 5120
        module.attn_sink = nn.Parameter(
            torch.arange(24, dtype=torch.float32), requires_grad=False
        )
        module.wq_b = nn.Module()
        module.wq_b.weight = nn.Parameter(
            torch.zeros((24 * 512, 1280), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        module.wq_b.weight_scale_inv = nn.Parameter(
            torch.ones((24 * 512 // 32, 1280 // 32), dtype=torch.uint8),
            requires_grad=False,
        )
        module.wo_a = nn.Module()
        module.wo_a.weight = nn.Parameter(
            torch.zeros((3 * 1024, 4096), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        module.wo_a.weight_scale_inv = nn.Parameter(
            torch.ones((3 * 1024 // 32, 4096 // 32), dtype=torch.uint8),
            requires_grad=False,
        )
        module.wo_b = nn.Module()
        module.wo_b.weight = nn.Parameter(
            torch.zeros((5120, 3 * 1024), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        module.wo_b.weight_scale_inv = nn.Parameter(
            torch.ones((5120 // 32, 3 * 1024 // 32), dtype=torch.uint8),
            requires_grad=False,
        )
        return module

    monkeypatch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: 3)

    for rank in (0, 1):
        monkeypatch.setattr(attention, "get_tensor_model_parallel_rank", lambda: rank)
        layer = make_layer()
        layer.setup_wo_projection()
        assert layer.n_local_groups == 3
        assert layer.n_local_heads == 24
        assert layer.wq_b.weight.shape == (12288, 1280)
        assert layer._wo_projection_weights.groups == 3

    monkeypatch.setattr(attention, "get_tensor_model_parallel_rank", lambda: 2)
    layer_r2 = make_layer()
    layer_r2.setup_wo_projection()
    layer_r2.setup_wo_projection()  # idempotent
    assert layer_r2.n_local_groups == 2
    assert layer_r2.n_local_heads == 16
    assert layer_r2.attn_sink.shape == (16,)
    assert layer_r2.wq_b.weight.shape == (8192, 1280)
    assert layer_r2.wq_b.weight_scale_inv.shape == (8192 // 32, 1280 // 32)
    assert layer_r2.wo_a.weight.shape == (2048, 4096)
    assert layer_r2.wo_a.weight_scale_inv.shape == (2048 // 32, 4096 // 32)
    assert layer_r2.wo_b.weight.shape == (5120, 2048)
    assert layer_r2.wo_b.weight_scale_inv.shape == (5120 // 32, 2048 // 32)
    assert layer_r2._wo_projection_weights.groups == 2
