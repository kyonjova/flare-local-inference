"""MiMo host placement, channel capacities, and per-checkpoint template assets."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.recipes.mimo_v2 import bootstrap, recipe
from ignition.recipes.mimo_v2.vision import PATCH_CAPACITIES

ASSETS = Path(bootstrap.__file__).with_name("assets")


def test_two_hosts_may_each_use_their_own_gpu_zero():
    config = bootstrap.PreparationConfig(devices=(0,) * 2, hosts=("maxwell", "hertz"))
    assert config.devices == (0,) * 2
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 2)
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 2, hosts=("maxwell",) * 2)
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0, 1, 2))


def test_ring_requires_hosts():
    with pytest.raises(ValueError):
        bootstrap.PreparationConfig(devices=(0, 1), roce_ring=True)


@pytest.mark.parametrize("remote", [False, True])
def test_recipe_dispatches_the_complete_plan_to_the_rank_group(monkeypatch, remote):
    from ignition.preparation import rank_group

    # Placement arithmetic only: the qualified checkpoint needs more than two Sparks of memory.
    gpus = tuple(planning.Gpu(i, "GB10", f"GPU-{i}", 512 << 30, 500 << 30, (12, 1)) for i in range(2))
    plan = planning.fit(recipe.RECIPE, recipe.RECIPE.checkpoints[0], gpus, concurrency=8,
                        reserve_bytes=4 << 30, speculation=recipe.RECIPE.speculation)
    ranks = SimpleNamespace(hosts=("maxwell", "hertz")) if remote else None
    calls, result, storage = [], object(), object()
    monkeypatch.setattr(rank_group, "prepare_group", lambda *args, **kwargs: calls.append((args, kwargs)) or result)
    assert recipe.prepare("checkpoint", plan, compile_forwards=True, kernel_cache=None, progress=None,
                          package_storage=storage, ranks=ranks) is result
    (snapshot, config, forward), kwargs = calls[0]
    assert snapshot == "checkpoint" and forward is bootstrap.prepare_rank
    assert config.devices == ((0, 0) if remote else (0, 1))
    assert config.hosts == (ranks.hosts if remote else None)
    assert kwargs["ranks"] is ranks and kwargs["checkpoint"] is plan.checkpoint
    assert kwargs["package_storage"] is storage


@pytest.mark.parametrize("hidden", [4096, 6144])
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("dflash", [False, True])
def test_remote_channels_cover_every_collective_without_pcie(monkeypatch, dflash, world, hidden):
    config = bootstrap.PreparationConfig(devices=(0,) * world, hosts=("a", "b", "c", "d")[:world], max_active=16,
                                         dflash=dflash, speculative_token_capacity=128)
    calls, runtime, storage = [], object(), object()
    monkeypatch.setattr(bootstrap, "create_roce_runtime",
        lambda group, device, **kwargs: calls.append(kwargs) or runtime)
    for name in ("create_all_reduce_runtime", "create_all_gather_runtime", "create_dma_all_reduce_runtime"):
        monkeypatch.setattr(bootstrap, name, lambda *args, **kwargs: pytest.fail("host ranks cannot use CUDA IPC"))
    draft_block = 8 if dflash else 0
    channels = bootstrap._collective_runtimes(config, None, None, hidden=hidden, draft_block=draft_block,
        coordinator=SimpleNamespace(spans_hosts=True), package_storage=storage)
    assert channels == (runtime, runtime, runtime, runtime if dflash else None, runtime)
    (call,) = calls
    assert call["package_storage"] is storage and call["hidden_size"] == hidden
    slots = config.request_capacity
    tokens = max(config.prefill_chunk_size, config.decode_tokens, slots * draft_block)
    pipe_bytes = call["pipe_tokens"] * hidden * 2
    # Prefill hidden rows, the vision blocks' 1280-wide patch rows, and the merged image features.
    assert pipe_bytes >= tokens * hidden * 2
    assert pipe_bytes >= PATCH_CAPACITIES[-1] * 1280 * 2
    assert pipe_bytes >= PATCH_CAPACITIES[-1] // 4 * hidden * 2
    assert call["max_tokens"] >= config.decode_tokens
    vocabulary_rows = max(config.decode_samples, slots * draft_block)
    assert call["gather_bytes"] >= vocabulary_rows * (152576 // world) * 2
    if dflash:
        assert call["gather_bytes"] >= tokens * (hidden // world) * 2
    assert call["gather_bytes"] % 16 == 0


@pytest.mark.parametrize("hidden", [4096, 6144])
def test_local_channels_use_the_checkpoint_hidden_width(monkeypatch, hidden):
    config = bootstrap.PreparationConfig(devices=(0, 1), max_active=16)
    reductions, gathers = [], []
    monkeypatch.setattr(bootstrap, "create_all_reduce_runtime",
        lambda group, device, **kwargs: reductions.append(kwargs) or object())
    monkeypatch.setattr(bootstrap, "create_dma_all_reduce_runtime",
        lambda group, device, **kwargs: reductions.append(kwargs) or object())
    monkeypatch.setattr(bootstrap, "create_all_gather_runtime",
        lambda group, device, **kwargs: gathers.append(kwargs) or object())
    monkeypatch.setattr(bootstrap, "create_roce_runtime",
        lambda *args, **kwargs: pytest.fail("local ranks share CUDA IPC"))
    channels = bootstrap._collective_runtimes(config, None, None, hidden=hidden, draft_block=8,
        coordinator=SimpleNamespace(spans_hosts=False), package_storage=SimpleNamespace(peer_storage=object()))
    assert len({id(channel) for channel in channels}) == 5
    assert [r["hidden_size"] for r in reductions] == [hidden] * 3
    assert reductions[-1]["max_tokens"] == PATCH_CAPACITIES[-1] // 4
    assert [g["columns"] for g in gathers] == [152576, hidden]


def _render(template, messages, **options):
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    # Transformers' chat-template filter signature.
    environment.filters["tojson"] = lambda value, ensure_ascii=False, indent=None, separators=None, sort_keys=False: (
        json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys))
    return environment.from_string(template.replace("{%- mark -%}", "")).render(messages=messages, **options)


def test_each_checkpoint_template_renders_like_the_qualified_one():
    pro, flash = (bootstrap.chat_template_path(hidden).read_text() for hidden in (6144, 4096))
    assert flash.count("{%- mark -%}") == pro.count("{%- mark -%}") == 2
    tools = [{"type": "function", "function": {"name": "search", "description": "Look up",
              "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
    conversations = [
        [{"role": "user", "content": "Hello"}],
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi\n\n  there"},
         {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"},
         {"role": "user", "content": [{"type": "text", "text": "Describe"}, {"type": "image"}]}],
        [{"role": "user", "content": "Find it"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"type": "function", "function": {"name": "search", "arguments": {"q": "x"}}}]},
         {"role": "tool", "content": "result"}],
    ]
    for messages in conversations:
        for options in ({"add_generation_prompt": True}, {"add_generation_prompt": False, "tools": tools}):
            assert _render(flash, messages, **options) == _render(pro, messages, **options)


def test_unknown_template_width_is_rejected():
    with pytest.raises(ValueError, match="hidden size"):
        bootstrap.chat_template_path(5120)
