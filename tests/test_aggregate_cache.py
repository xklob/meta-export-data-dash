from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from meta_archive.api.aggregate_cache import RevisionBoundAggregateCache


def test_revision_bound_aggregate_cache_is_bounded_thread_safe_and_opaque() -> None:
    cache = RevisionBoundAggregateCache(max_entries=2)
    calls = 0

    def load_one() -> int:
        nonlocal calls
        calls += 1
        return 1

    first = cache.get_or_compute(
        namespace="private-filter",
        scope="private search text",
        revision="revision-one",
        revision_is_current=lambda: True,
        compute=load_one,
    )
    second = cache.get_or_compute(
        namespace="private-filter",
        scope="private search text",
        revision="revision-one",
        revision_is_current=lambda: True,
        compute=load_one,
    )
    assert (first, second, calls) == (1, 1, 1)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _index: cache.get_or_compute(
                    namespace="private-filter",
                    scope="private search text",
                    revision="revision-one",
                    revision_is_current=lambda: True,
                    compute=load_one,
                ),
                range(32),
            )
        )
    assert results == [1] * 32
    assert calls == 1

    for scope, value in (("second", 2), ("third", 3)):
        assert (
            cache.get_or_compute(
                namespace="bounded",
                scope=scope,
                revision="revision-one",
                revision_is_current=lambda: True,
                compute=lambda value=value: value,
            )
            == value
        )
    assert len(cache) == 2


def test_revision_guards_reject_stale_hits_and_mid_compute_publication() -> None:
    cache = RevisionBoundAggregateCache(max_entries=4)
    assert (
        cache.get_or_compute(
            namespace="totals",
            scope="all",
            revision="old-revision",
            revision_is_current=lambda: True,
            compute=lambda: 10,
        )
        == 10
    )

    # A revision change after lookup rejects the old hit and prevents the replacement value from
    # being published under that old revision.
    assert (
        cache.get_or_compute(
            namespace="totals",
            scope="all",
            revision="old-revision",
            revision_is_current=lambda: False,
            compute=lambda: 11,
        )
        == 11
    )
    assert (
        cache.get_or_compute(
            namespace="totals",
            scope="all",
            revision="old-revision",
            revision_is_current=lambda: True,
            compute=lambda: 99,
        )
        == 10
    )

    current = True

    def change_revision_during_compute() -> int:
        nonlocal current
        current = False
        return 20

    assert (
        cache.get_or_compute(
            namespace="totals",
            scope="filtered",
            revision="before-compute",
            revision_is_current=lambda: current,
            compute=change_revision_during_compute,
        )
        == 20
    )
    current = True
    assert (
        cache.get_or_compute(
            namespace="totals",
            scope="filtered",
            revision="before-compute",
            revision_is_current=lambda: current,
            compute=lambda: 21,
        )
        == 21
    )
