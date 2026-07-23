from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from meta_archive.api import search_routes
from meta_archive.models import SemanticIndexConfig
from meta_archive.services import search as search_service
from meta_archive.services.search_index import (
    AUTO_EMBEDDING_MODEL_ALIAS,
    CLS_POOLING,
    TRANSFORMERS_POOLING,
    TRANSFORMERS_PROVIDER,
    EmbeddingModelSpec,
    TransformersTextEmbedder,
    WindowBuildSummary,
    _validated_embedding_batch_size,
    embedding_implementation_identity,
    embedding_model_spec,
)


def test_gte_modernbert_spec_is_exact_and_remains_a_nondefault_candidate() -> None:
    spec = embedding_model_spec("gte-modernbert-base")

    assert spec.model_name == "Alibaba-NLP/gte-modernbert-base"
    assert spec.model_revision == "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
    assert spec.dimensions == 768
    assert spec.document_prefix == ""
    assert spec.query_prefix == ""
    assert spec.language_scope == "English"
    assert spec.provider == TRANSFORMERS_PROVIDER
    assert spec.pooling == CLS_POOLING
    assert spec.max_sequence_tokens == 512
    assert spec.precision == "float32"
    assert spec.batch_order == "stable-token-length-ascending-restored"
    assert WindowBuildSummary(generation="fixture").recommended_model == (
        AUTO_EMBEDDING_MODEL_ALIAS
    )
    assert spec.alias != AUTO_EMBEDDING_MODEL_ALIAS
    assert embedding_implementation_identity(spec) == {
        "pipeline_version": "semantic-text-embedding-v1",
        "document_prefix": "",
        "query_prefix": "",
        "normalization": "l2",
        "provider": "transformers",
        "device": "cpu",
        "precision": "float32",
        "pooling": "cls",
        "max_sequence_tokens": 512,
        "truncation": True,
        "document_batching": {
            "microbatch_size": 16,
            "sort_bucket_size": 128,
            "sort_order": "stable-token-length-ascending-restored",
            "restore_input_order": True,
        },
    }


def test_gte_index_batch_size_preserves_fixed_sort_bucket_boundaries() -> None:
    spec = embedding_model_spec("gte-modernbert-base")

    assert _validated_embedding_batch_size(spec, 128) == 128
    assert _validated_embedding_batch_size(spec, 512) == 512
    with pytest.raises(ValueError, match="must be divisible by 128"):
        _validated_embedding_batch_size(spec, 100)
    with pytest.raises(ValueError, match="must be positive"):
        _validated_embedding_batch_size(spec, 0)


def test_multilingual_embedding_identity_remains_legacy_compatible() -> None:
    spec = embedding_model_spec("multilingual-e5-small")

    assert embedding_implementation_identity(spec) == {
        "pipeline_version": "semantic-text-embedding-v1",
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "normalization": "l2",
        "provider": "transformers",
        "device": "cpu",
        "precision": "float32",
        "pooling": TRANSFORMERS_POOLING,
        "max_sequence_tokens": 512,
        "truncation": True,
    }


def test_injected_embedding_identity_remains_legacy_compatible() -> None:
    spec = EmbeddingModelSpec(
        alias="fixture",
        model_name="fixture/embedding",
        model_revision="fixture-v1",
        dimensions=4,
        document_prefix="passage: ",
        query_prefix="query: ",
        language_scope="Fixture",
        approximate_download_gb=0,
    )

    assert embedding_implementation_identity(spec) == {
        "pipeline_version": "semantic-text-embedding-v1",
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "normalization": "l2",
        "provider": "injected-text-embedder-contract",
        "precision": "provider-declared",
        "pooling": "provider-declared",
    }


def test_transformers_embedder_uses_cls_pooling_and_the_spec_token_limit() -> None:
    torch = pytest.importorskip("torch")
    spec = embedding_model_spec("gte-modernbert-base")
    tokenizer_calls: list[dict[str, Any]] = []

    class Tokenizer:
        def __call__(self, texts: list[str], **options: Any) -> dict[str, Any]:
            tokenizer_calls.append({"texts": texts, **options})
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

    class Model:
        def __call__(self, **_encoded: Any) -> SimpleNamespace:
            return SimpleNamespace(
                last_hidden_state=torch.tensor(
                    [[[3.0, 4.0], [100.0, 0.0], [0.0, 100.0]]],
                    dtype=torch.float32,
                )
            )

    embedder = object.__new__(TransformersTextEmbedder)
    embedder.spec = spec
    embedder._tokenizer = Tokenizer()
    embedder._model = Model()

    assert embedder.embed_query("argument") == pytest.approx([0.6, 0.8])
    assert tokenizer_calls == [
        {
            "texts": ["argument"],
            "max_length": 512,
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
    ]


def test_gte_length_sorted_buckets_restore_order_and_are_outer_batch_invariant() -> None:
    torch = pytest.importorskip("torch")
    spec = embedding_model_spec("gte-modernbert-base")
    texts = [f"item-{index}" for index in range(256)]
    lengths = {text: ((index * 37) % 127) + 1 for index, text in enumerate(texts)}

    class Tokenizer:
        def __init__(self) -> None:
            self.length_batches: list[list[int]] = []
            self.embedding_batches: list[list[int]] = []

        def __call__(self, batch: list[str], **options: Any) -> dict[str, Any]:
            indexes = [int(text.removeprefix("item-")) for text in batch]
            if options.get("return_length") is True:
                self.length_batches.append(indexes)
                return {"length": [lengths[text] for text in batch]}
            self.embedding_batches.append(indexes)
            input_ids = torch.tensor([[index + 1] for index in indexes])
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

    class Model:
        def __call__(self, *, input_ids: Any, **_encoded: Any) -> SimpleNamespace:
            values = input_ids[:, 0].to(dtype=torch.float32)
            cls = torch.stack((values, torch.ones_like(values)), dim=1)
            return SimpleNamespace(last_hidden_state=cls.unsqueeze(1))

    def make_embedder() -> tuple[TransformersTextEmbedder, Tokenizer]:
        tokenizer = Tokenizer()
        embedder = object.__new__(TransformersTextEmbedder)
        embedder.spec = spec
        embedder._tokenizer = tokenizer
        embedder._model = Model()
        return embedder, tokenizer

    combined_embedder, combined_tokenizer = make_embedder()
    combined = combined_embedder.embed_documents(texts)
    split_embedder, split_tokenizer = make_embedder()
    split = split_embedder.embed_documents(texts[:128]) + split_embedder.embed_documents(
        texts[128:]
    )

    assert combined == split
    assert combined_tokenizer.length_batches == [list(range(128)), list(range(128, 256))]
    assert split_tokenizer.length_batches == [list(range(128)), list(range(128, 256))]
    expected_first_microbatch = sorted(
        range(128),
        key=lambda index: lengths[texts[index]],
    )[:16]
    assert combined_tokenizer.embedding_batches[0] == expected_first_microbatch
    for index in (0, 31, 127, 128, 255):
        value = float(index + 1)
        norm = (value * value + 1.0) ** 0.5
        assert combined[index] == pytest.approx([value / norm, 1.0 / norm])


def test_search_config_requires_exact_registered_model_identity() -> None:
    registered = embedding_model_spec("gte-modernbert-base")
    exact = SemanticIndexConfig(
        model_name=registered.model_name,
        model_revision=registered.model_revision,
        dimensions=registered.dimensions,
        document_prefix=registered.document_prefix,
        query_prefix=registered.query_prefix,
        backend="lancedb-transformers",
        audit_json={},
    )

    assert search_service._spec_for_config(exact) is registered

    historical = SemanticIndexConfig(
        model_name=registered.model_name,
        model_revision="historical-unregistered-revision",
        dimensions=registered.dimensions,
        document_prefix=registered.document_prefix,
        query_prefix=registered.query_prefix,
        backend="lancedb-transformers",
        audit_json={
            "embedding_implementation": {
                "provider": "transformers",
                "pooling": "attention-mask-mean",
                "max_sequence_tokens": 384,
                "precision": "float32",
                "document_batching": {
                    "microbatch_size": 16,
                    "sort_bucket_size": 128,
                    "sort_order": "stable-token-length-ascending-restored",
                    "restore_input_order": True,
                },
            }
        },
    )
    reconstructed = search_service._spec_for_config(historical)

    assert reconstructed is not registered
    assert reconstructed.model_revision == "historical-unregistered-revision"
    assert reconstructed.provider == TRANSFORMERS_PROVIDER
    assert reconstructed.pooling == TRANSFORMERS_POOLING
    assert reconstructed.max_sequence_tokens == 384
    assert reconstructed.batch_order == "stable-token-length-ascending-restored"


def test_search_embedder_cache_keys_every_vector_affecting_spec_field() -> None:
    calls: list[object] = []

    def factory(spec: object) -> object:
        sentinel = object()
        calls.append((spec, sentinel))
        return sentinel

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                search_embedder_factory=factory,
            )
        )
    )
    get_embedder = search_routes._cached_search_embedder_factory(request)
    original = embedding_model_spec("gte-modernbert-base")
    changed_pooling = replace(original, pooling=TRANSFORMERS_POOLING)

    first = get_embedder(original)
    second = get_embedder(changed_pooling)

    assert first is not second
    assert get_embedder(original) is first
    assert get_embedder(changed_pooling) is second
    assert [spec for spec, _sentinel in calls] == [original, changed_pooling]
