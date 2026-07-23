from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest

from meta_archive.api.people_routes import (
    _RevisionBoundPeopleProjectionCache,
    _WeightedProjection,
)


def test_people_projection_cache_is_bounded_thread_safe_and_opaque() -> None:
    cache = _RevisionBoundPeopleProjectionCache(max_entries=3, max_rows=4, max_bytes=20)
    calls = 0

    def load_first() -> _WeightedProjection[int]:
        nonlocal calls
        calls += 1
        return _WeightedProjection(value=1, row_count=1, byte_count=8)

    assert (
        cache.get_or_compute(
            namespace="private-people-projection",
            scope="private contact filter",
            revision="revision-one",
            revision_is_current=lambda: True,
            compute=load_first,
        )
        == 1
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _index: cache.get_or_compute(
                    namespace="private-people-projection",
                    scope="private contact filter",
                    revision="revision-one",
                    revision_is_current=lambda: True,
                    compute=load_first,
                ),
                range(32),
            )
        )
    assert results == [1] * 32
    assert calls == 1
    assert all(isinstance(key, bytes) and len(key) == 32 for key in cache._entries)
    assert all(b"private" not in key for key in cache._entries)

    for scope, value in (("second", 2), ("third", 3)):
        assert (
            cache.get_or_compute(
                namespace="bounded",
                scope=scope,
                revision="revision-one",
                revision_is_current=lambda: True,
                compute=lambda value=value: _WeightedProjection[int](
                    value=value,
                    row_count=2,
                    byte_count=8,
                ),
            )
            == value
        )
    assert len(cache) == 2
    assert cache.retained_rows == 4
    assert cache.retained_bytes == 16

    # A projection that cannot fit is returned to the request but never retained.
    assert (
        cache.get_or_compute(
            namespace="bounded",
            scope="oversized",
            revision="revision-one",
            revision_is_current=lambda: True,
            compute=lambda: _WeightedProjection(value=4, row_count=1, byte_count=21),
        )
        == 4
    )
    assert len(cache) == 2
    assert cache.retained_rows <= 4
    assert cache.retained_bytes <= 20


def test_people_projection_cache_rejects_bad_weights_and_stale_publication() -> None:
    cache = _RevisionBoundPeopleProjectionCache(max_entries=2, max_rows=2, max_bytes=20)
    with pytest.raises(ValueError, match="weights cannot be negative"):
        cache.get_or_compute(
            namespace="contacts",
            scope="all",
            revision="revision-one",
            revision_is_current=lambda: True,
            compute=lambda: _WeightedProjection(value=1, row_count=-1, byte_count=1),
        )

    current = True

    def change_revision_during_compute() -> _WeightedProjection[int]:
        nonlocal current
        current = False
        return _WeightedProjection(value=10, row_count=1, byte_count=1)

    assert (
        cache.get_or_compute(
            namespace="contacts",
            scope="all",
            revision="before-change",
            revision_is_current=lambda: current,
            compute=change_revision_during_compute,
        )
        == 10
    )
    assert len(cache) == 0
    current = True
    assert (
        cache.get_or_compute(
            namespace="contacts",
            scope="all",
            revision="before-change",
            revision_is_current=lambda: current,
            compute=lambda: _WeightedProjection(value=11, row_count=1, byte_count=1),
        )
        == 11
    )
    assert len(cache) == 1


def test_people_projection_cache_cold_concurrency_keeps_eviction_accounting_bounded() -> None:
    workers = 8
    cache = _RevisionBoundPeopleProjectionCache(
        max_entries=8,
        max_rows=3,
        max_bytes=18,
    )
    duplicate_barrier = Barrier(workers)
    call_lock = Lock()
    duplicate_calls = 0

    def load_shared() -> _WeightedProjection[int]:
        nonlocal duplicate_calls
        with call_lock:
            duplicate_calls += 1
        # Every caller has already observed the initial miss before any result can be published.
        duplicate_barrier.wait()
        return _WeightedProjection(value=7, row_count=1, byte_count=6)

    def request_shared(_index: int) -> int:
        return cache.get_or_compute(
            namespace="contacts",
            scope="shared-cold-key",
            revision="revision-one",
            revision_is_current=lambda: True,
            compute=load_shared,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        shared_results = list(executor.map(request_shared, range(workers)))
    assert shared_results == [7] * workers
    # The cache deliberately does not promise single-flight cold loading. All duplicate computes
    # completed, but only one weighted entry was accounted for and retained.
    assert duplicate_calls == workers
    assert len(cache) == 1
    assert cache.retained_rows == 1
    assert cache.retained_bytes == 6

    distinct_barrier = Barrier(workers)

    def request_distinct(index: int) -> int:
        def load() -> _WeightedProjection[int]:
            distinct_barrier.wait()
            return _WeightedProjection(value=index, row_count=1, byte_count=6)

        return cache.get_or_compute(
            namespace="contacts",
            scope=f"distinct-{index}",
            revision="revision-one",
            revision_is_current=lambda: True,
            compute=load,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        distinct_results = list(executor.map(request_distinct, range(workers)))
    assert sorted(distinct_results) == list(range(workers))
    assert len(cache) == 3
    assert cache.retained_rows == 3
    assert cache.retained_bytes == 18
