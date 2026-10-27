"""DeepSeek V4 TP2/TP4 host placement and channels preserve the local preparation contract."""

from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.recipes.deepseek4 import bootstrap, recipe


def test_two_hosts_may_each_use_their_own_gpu_zero():
    config = bootstrap.PreparationConfig(devices=(0,) * 2, hosts=("maxwell", "hertz"))
    assert config.devices == (0,) * 2
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 2)
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 2, hosts=("maxwell",) * 2)
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 2, hosts=("maxwell", "hertz", "faraday"))


def test_ring_requires_hosts():
    with pytest.raises(ValueError):
        bootstrap.PreparationConfig(devices=(0, 1), roce_ring=True)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("remote", [False, True])
def test_recipe_dispatches_the_complete_plan_to_the_rank_group(monkeypatch, remote, world):
    from ignition.preparation import rank_group

    gpus = tuple(planning.Gpu(i, "GB10", f"GPU-{i}", 128 << 30, 120 << 30, (12, 1)) for i in range(world))
    plan = planning.fit(recipe.RECIPE, recipe.RECIPE.checkpoints[0], gpus, concurrency=8,
                        reserve_bytes=4 << 30, speculation=recipe.RECIPE.speculation)
    ranks = SimpleNamespace(hosts=("maxwell", "hertz", "faraday", "ampere")[:world]) if remote else None
    calls, result, storage = [], object(), object()
    monkeypatch.setattr(rank_group, "prepare_group", lambda *args, **kwargs: calls.append((args, kwargs)) or result)
    assert recipe.prepare("checkpoint", plan, compile_forwards=True, kernel_cache=None, progress=None,
                          package_storage=storage, ranks=ranks) is result
    (snapshot, config, forward), kwargs = calls[0]
    assert snapshot == "checkpoint" and forward is bootstrap.prepare_rank
    assert config.devices == ((0,) * world if remote else tuple(range(world)))
    assert config.hosts == (ranks.hosts if remote else None)
    assert config.max_active == 8 and config.dspark
    assert kwargs["ranks"] is ranks and kwargs["checkpoint"] is plan.checkpoint
    assert kwargs["package_storage"] is storage


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("dspark", [False, True])
def test_remote_channels_cover_vocabulary_and_markov_gathers_without_pcie(monkeypatch, dspark, world):
    hosts = ("a", "b", "c", "d")[:world]
    config = bootstrap.PreparationConfig(devices=(0,) * world, hosts=hosts, max_active=16, dspark=dspark,
                                         speculative_widths=(1, 2, 3, 4, 5, 6), speculative_token_capacity=128)
    calls, runtime, storage = [], object(), object()
    monkeypatch.setattr(bootstrap, "create_roce_runtime",
        lambda group, device, **kwargs: calls.append(kwargs) or runtime)
    for name in ("create_all_reduce_runtime", "create_all_gather_runtime", "create_dma_all_reduce_runtime"):
        monkeypatch.setattr(bootstrap, name, lambda *args, **kwargs: pytest.fail("host ranks cannot use CUDA IPC"))
    channels = bootstrap._collective_runtimes(config, None, None,
        coordinator=SimpleNamespace(spans_hosts=True), package_storage=storage)
    assert channels == (runtime, runtime, runtime, runtime if dspark else None)
    assert len(calls) == 1
    (call,) = calls
    assert call["package_storage"] is storage and call["hidden_size"] == 4096
    slots = config.request_capacity
    assert call["pipe_tokens"] >= max(config.prefill_chunk_size, config.decode_tokens, slots * config.draft_tokens)
    assert call["max_tokens"] >= config.decode_tokens
    # BF16 vocabulary partitions for every sampled row.
    vocabulary_rows = max(config.decode_samples, slots * config.draft_tokens)
    assert call["gather_bytes"] >= vocabulary_rows * (129280 // world) * 2
    if dspark:
        # FP32 Markov logits gathered as two BF16 columns each.
        assert call["gather_bytes"] >= slots * (2 * 129280 // world) * 2
    assert call["gather_bytes"] % 16 == 0


def test_local_channels_are_unchanged(monkeypatch):
    config = bootstrap.PreparationConfig(devices=(0, 1), max_active=16)
    gathers = []
    monkeypatch.setattr(bootstrap, "create_all_gather_runtime",
        lambda group, device, **kwargs: gathers.append(kwargs) or object())
    for name in ("create_all_reduce_runtime", "create_dma_all_reduce_runtime"):
        monkeypatch.setattr(bootstrap, name, lambda *args, **kwargs: object())
    monkeypatch.setattr(bootstrap, "create_roce_runtime",
        lambda *args, **kwargs: pytest.fail("local ranks share CUDA IPC"))
    collective, prefill, vocabulary, markov = bootstrap._collective_runtimes(config, None, None,
        coordinator=SimpleNamespace(spans_hosts=False), package_storage=SimpleNamespace(peer_storage=object()))
    assert len({id(collective), id(prefill), id(vocabulary), id(markov)}) == 4
    assert gathers[0]["columns"] == 129280
    assert gathers[0]["max_rows"] == max(config.decode_samples, config.request_capacity * config.draft_tokens)
    assert gathers[1] == {"max_rows": config.request_capacity, "columns": 2 * 129280, "ipc": gathers[1]["ipc"]}
