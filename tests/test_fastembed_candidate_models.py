from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from fastembed.common.model_description import PoolingType
from pydantic import ValidationError

from meta_archive.api.schemas import SemanticIndexRequest
from meta_archive.models import SemanticIndexConfig
from meta_archive.services import search as search_service
from meta_archive.services.search_index import (
    CLS_POOLING,
    FASTEMBED_PROVIDER,
    PINNED_FASTEMBED_ARTIFACTS,
    FastEmbedTextEmbedder,
    embedding_config_hash,
    embedding_implementation_identity,
    embedding_model_spec,
)

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CANDIDATE_CONTRACTS: dict[str, dict[str, Any]] = {
    "bge-base-en-v1.5": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "model_revision": "738cad1c108e2f23649db9e44b2eab988626493b",
        "fastembed_source": "qdrant/bge-base-en-v1.5-onnx-q",
        "model_file": "model_optimized.onnx",
        "dimensions": 768,
        "download_gb": 0.22,
        "fastembed_declared_gb": 0.21,
        "precision": "float16-onnx",
    },
    "e5-base-v2": {
        "model_name": "intfloat/e5-base-v2",
        "model_revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        "fastembed_source": "intfloat/e5-base-v2",
        "model_file": "onnx/model.onnx",
        "dimensions": 768,
        "download_gb": 0.44,
        "precision": "float32-onnx",
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "custom_pooling": "MEAN",
        "custom_normalization": True,
    },
    "e5-small-v2": {
        "model_name": "intfloat/e5-small-v2",
        "model_revision": "ffb93f3bd4047442299a41ebb6fa998a38507c52",
        "fastembed_source": "intfloat/e5-small-v2",
        "model_file": "model.onnx",
        "dimensions": 384,
        "download_gb": 0.134,
        "precision": "float32-onnx",
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "custom_pooling": "MEAN",
        "custom_normalization": True,
    },
    "mxbai-embed-large-v1": {
        "model_name": "mixedbread-ai/mxbai-embed-large-v1",
        "model_revision": "b33106f585b9ce46904ad7443a3b52b7a63e231c",
        "fastembed_source": "mixedbread-ai/mxbai-embed-large-v1",
        "model_file": "onnx/model.onnx",
        "dimensions": 1024,
        "download_gb": 1.34,
        "fastembed_declared_gb": 0.64,
        "precision": "float32-onnx",
    },
    "nomic-embed-text-v1.5": {
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "model_revision": "e9b6763023c676ca8431644204f50c2b100d9aab",
        "fastembed_source": "nomic-ai/nomic-embed-text-v1.5",
        "model_file": "onnx/model.onnx",
        "dimensions": 768,
        "download_gb": 0.55,
        "fastembed_declared_gb": 0.52,
        "precision": "float32-onnx",
        "document_prefix": "search_document: ",
        "query_prefix": "search_query: ",
        "pooling": "attention-mask-mean",
        "application_max_tokens": 512,
    },
    "snowflake-arctic-embed-l": {
        "model_name": "snowflake/snowflake-arctic-embed-l",
        "model_revision": "d8fb21ca8d905d2832ee8b96c894d3298964346b",
        "fastembed_source": "snowflake/snowflake-arctic-embed-l",
        "model_file": "onnx/model.onnx",
        "dimensions": 1024,
        "download_gb": 1.34,
        "fastembed_declared_gb": 1.02,
        "precision": "float32-onnx",
    },
}
BUILTIN_FASTEMBED_ALIASES = (
    "bge-base-en-v1.5",
    "mxbai-embed-large-v1",
    "nomic-embed-text-v1.5",
    "snowflake-arctic-embed-l",
)
API_EXPLICIT_CANDIDATE_ALIASES = (
    "bge-base-en-v1.5",
    "e5-base-v2",
    "mxbai-embed-large-v1",
    "snowflake-arctic-embed-l",
)
CUSTOM_E5_ALIASES = ("e5-base-v2", "e5-small-v2")


class _FakeTokenizer:
    def __init__(self) -> None:
        self.truncation: dict[str, int] | None = None

    def enable_truncation(self, *, max_length: int) -> None:
        self.truncation = {"max_length": max_length}


def _create_pinned_snapshot(cache_dir: Path, alias: str) -> Path:
    artifact = PINNED_FASTEMBED_ARTIFACTS[alias]
    snapshot = (
        cache_dir
        / f"models--{artifact.repository.replace('/', '--')}"
        / "snapshots"
        / artifact.revision
    )
    for required in artifact.required_files:
        target = snapshot / required
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"{alias}:{required}".encode())
    return snapshot


def _tree_state(root: Path) -> list[tuple[str, bytes]]:
    return [
        (path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _fastembed_registry_entry(alias: str) -> dict[str, Any]:
    contract = CANDIDATE_CONTRACTS[alias]
    return {
        "model": contract["model_name"],
        "sources": {"hf": contract["fastembed_source"]},
        "model_file": contract["model_file"],
        "dim": contract["dimensions"],
    }


@pytest.mark.parametrize(("alias", "contract"), CANDIDATE_CONTRACTS.items())
def test_fastembed_candidate_model_contracts(
    alias: str,
    contract: dict[str, str | int | float],
) -> None:
    spec = embedding_model_spec(alias)

    assert spec.alias == alias
    assert spec.model_name == contract["model_name"]
    assert spec.model_revision == contract["model_revision"]
    assert spec.dimensions == contract["dimensions"]
    document_prefix = str(contract.get("document_prefix", ""))
    query_prefix = str(contract.get("query_prefix", QUERY_PREFIX))
    assert spec.document_prefix == document_prefix
    assert spec.query_prefix == query_prefix
    assert spec.language_scope == "English"
    assert spec.approximate_download_gb == contract["download_gb"]
    assert spec.provider == FASTEMBED_PROVIDER
    pooling = str(contract.get("pooling", CLS_POOLING))
    if "custom_pooling" in contract:
        pooling = "attention-mask-mean"
    assert spec.pooling == pooling
    assert spec.max_sequence_tokens == 512
    assert spec.precision == contract["precision"]
    artifact = PINNED_FASTEMBED_ARTIFACTS[alias]
    assert artifact.repository == contract["fastembed_source"]
    assert artifact.revision == spec.model_revision
    assert artifact.model_file == contract["model_file"]
    assert embedding_implementation_identity(spec) == {
        "pipeline_version": "semantic-text-embedding-v1",
        "document_prefix": document_prefix,
        "query_prefix": query_prefix,
        "normalization": "l2",
        "provider": "fastembed",
        "precision": contract["precision"],
        "pooling": pooling,
        "artifact_source": {
            "repository": contract["fastembed_source"],
            "revision": contract["model_revision"],
            "model_file": contract["model_file"],
            "required_files": list(artifact.required_files),
            "kind": "huggingface_snapshot",
            "fingerprint_schema": "hf-logical-snapshot-v1",
            **(
                {
                    "fastembed_custom_model": {
                        "pooling": contract["custom_pooling"],
                        "normalization": contract["custom_normalization"],
                    }
                }
                if "custom_pooling" in contract
                else {}
            ),
            **(
                {"application_max_tokens": contract["application_max_tokens"]}
                if "application_max_tokens" in contract
                else {}
            ),
        },
        "max_sequence_tokens": 512,
        "truncation": True,
    }


@pytest.mark.parametrize(
    ("alias", "contract"),
    [(alias, CANDIDATE_CONTRACTS[alias]) for alias in BUILTIN_FASTEMBED_ALIASES],
)
def test_fastembed_registry_resolves_exact_candidate_artifact(
    alias: str,
    contract: dict[str, str | int | float],
) -> None:
    fastembed = pytest.importorskip("fastembed")
    supported = {
        candidate["model"]: candidate
        for candidate in fastembed.TextEmbedding.list_supported_models()
    }
    candidate = supported[str(contract["model_name"])]

    assert candidate["sources"]["hf"] == contract["fastembed_source"]
    assert candidate["model_file"] == contract["model_file"]
    assert candidate["dim"] == contract["dimensions"]
    assert candidate["size_in_GB"] == contract["fastembed_declared_gb"]
    assert "English" in candidate["description"]
    expected_registry_tokens = 8192 if alias == "nomic-embed-text-v1.5" else 512
    assert f"{expected_registry_tokens} input tokens truncation" in candidate["description"]


@pytest.mark.parametrize("alias", CANDIDATE_CONTRACTS)
def test_fastembed_candidate_config_identity_is_exact(alias: str) -> None:
    spec = embedding_model_spec(alias)
    config = SemanticIndexConfig(
        model_name=spec.model_name,
        model_revision=spec.model_revision,
        dimensions=spec.dimensions,
        document_prefix=spec.document_prefix,
        query_prefix=spec.query_prefix,
        backend="lancedb-fastembed",
        audit_json={"embedding_implementation": embedding_implementation_identity(spec)},
    )

    assert search_service._spec_for_config(config) is spec
    fingerprint = "sha256:" + "a" * 64
    exact_hash = embedding_config_hash(
        spec,
        corpus_revision="candidate-bakeoff-v1",
        artifact_fingerprint=fingerprint,
    )
    assert exact_hash == embedding_config_hash(
        spec,
        corpus_revision="candidate-bakeoff-v1",
        artifact_fingerprint=fingerprint,
    )
    assert exact_hash != embedding_config_hash(
        replace(spec, query_prefix=""),
        corpus_revision="candidate-bakeoff-v1",
        artifact_fingerprint=fingerprint,
    )
    assert exact_hash != embedding_config_hash(
        spec,
        corpus_revision="candidate-bakeoff-v1",
        artifact_fingerprint="sha256:" + "b" * 64,
    )
    identity = embedding_implementation_identity(spec)
    assert json.loads(json.dumps(identity)) == identity


@pytest.mark.parametrize("alias", CANDIDATE_CONTRACTS)
def test_fastembed_candidate_source_change_invalidates_config_identity(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    spec = embedding_model_spec(alias)
    artifact = PINNED_FASTEMBED_ARTIFACTS[alias]
    fingerprint = "sha256:" + "a" * 64
    original_identity = embedding_implementation_identity(spec)
    original_hash = embedding_config_hash(
        spec,
        corpus_revision="candidate-bakeoff-v1",
        artifact_fingerprint=fingerprint,
    )

    monkeypatch.setitem(
        PINNED_FASTEMBED_ARTIFACTS,
        alias,
        replace(artifact, repository="fixture/changed-artifact-source"),
    )

    assert embedding_implementation_identity(spec) != original_identity
    assert (
        embedding_config_hash(
            spec,
            corpus_revision="candidate-bakeoff-v1",
            artifact_fingerprint=fingerprint,
        )
        != original_hash
    )


@pytest.mark.parametrize("alias", BUILTIN_FASTEMBED_ALIASES)
def test_fastembed_candidate_read_only_mode_loads_exact_existing_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    calls: list[dict[str, Any]] = []
    tokenizers: list[_FakeTokenizer] = []

    class FakeTextEmbedding:
        @staticmethod
        def list_supported_models() -> list[dict[str, Any]]:
            return [_fastembed_registry_entry(alias)]

        def __init__(self, **options: Any) -> None:
            calls.append(options)
            tokenizer = _FakeTokenizer() if alias == "nomic-embed-text-v1.5" else None
            if tokenizer is not None:
                tokenizers.append(tokenizer)
            self.model = SimpleNamespace(
                _model_dir=options["specific_model_path"],
                tokenizer=tokenizer,
            )

    fastembed = ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    cache_dir = tmp_path / alias
    snapshot = _create_pinned_snapshot(cache_dir, alias)
    before = _tree_state(tmp_path)

    embedder = FastEmbedTextEmbedder(
        embedding_model_spec(alias),
        cache_dir=str(cache_dir),
        local_files_only=True,
    )

    assert calls == [
        {
            "model_name": CANDIDATE_CONTRACTS[alias]["model_name"],
            "cache_dir": str(cache_dir),
            "threads": 8,
            "specific_model_path": str(snapshot),
            "local_files_only": True,
        }
    ]
    assert embedder.artifact_fingerprint().startswith("sha256:")
    if alias == "nomic-embed-text-v1.5":
        assert tokenizers[0].truncation == {"max_length": 512}
    else:
        assert tokenizers == []
    assert _tree_state(tmp_path) == before


@pytest.mark.parametrize("alias", BUILTIN_FASTEMBED_ALIASES)
def test_fastembed_candidate_read_only_missing_snapshot_fails_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    calls: list[dict[str, Any]] = []

    class ForbiddenTextEmbedding:
        @staticmethod
        def list_supported_models() -> list[dict[str, Any]]:
            return [_fastembed_registry_entry(alias)]

        def __init__(self, **options: Any) -> None:
            calls.append(options)

    fastembed = ModuleType("fastembed")
    fastembed.TextEmbedding = ForbiddenTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    cache_dir = tmp_path / alias
    before = _tree_state(tmp_path)

    with pytest.raises(RuntimeError, match="pinned local model snapshot is unavailable"):
        FastEmbedTextEmbedder(
            embedding_model_spec(alias),
            cache_dir=str(cache_dir),
            local_files_only=True,
        )

    assert calls == []
    assert not cache_dir.exists()
    assert _tree_state(tmp_path) == before


@pytest.mark.parametrize("alias", BUILTIN_FASTEMBED_ALIASES)
def test_fastembed_candidate_online_load_pins_revision_and_specific_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    import huggingface_hub

    cache_dir = tmp_path / alias
    snapshot = _create_pinned_snapshot(cache_dir, alias)
    download_calls: list[dict[str, Any]] = []
    backend_calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**options: Any) -> str:
        download_calls.append(options)
        return str(snapshot)

    class FakeTextEmbedding:
        @staticmethod
        def list_supported_models() -> list[dict[str, Any]]:
            return [_fastembed_registry_entry(alias)]

        def __init__(self, **options: Any) -> None:
            backend_calls.append(options)
            tokenizer = _FakeTokenizer() if alias == "nomic-embed-text-v1.5" else None
            self.model = SimpleNamespace(
                _model_dir=options["specific_model_path"],
                tokenizer=tokenizer,
            )

    fastembed = ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    artifact = PINNED_FASTEMBED_ARTIFACTS[alias]

    FastEmbedTextEmbedder(
        embedding_model_spec(alias),
        cache_dir=str(cache_dir),
    )

    assert download_calls == [
        {
            "repo_id": artifact.repository,
            "revision": artifact.revision,
            "cache_dir": cache_dir,
            "allow_patterns": list(artifact.required_files),
        }
    ]
    assert backend_calls[0]["specific_model_path"] == str(snapshot)
    assert backend_calls[0]["local_files_only"] is True


def test_fastembed_candidate_rejects_installed_registry_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = "bge-base-en-v1.5"
    backend_calls: list[dict[str, Any]] = []

    class DriftedTextEmbedding:
        @staticmethod
        def list_supported_models() -> list[dict[str, Any]]:
            entry = _fastembed_registry_entry(alias)
            entry["sources"] = {"hf": "fixture/drifted-source"}
            return [entry]

        def __init__(self, **options: Any) -> None:
            backend_calls.append(options)

    fastembed = ModuleType("fastembed")
    fastembed.TextEmbedding = DriftedTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)

    with pytest.raises(RuntimeError, match="artifact declaration does not match"):
        FastEmbedTextEmbedder(
            embedding_model_spec(alias),
            cache_dir=str(tmp_path / "cache"),
        )

    assert backend_calls == []
    assert not (tmp_path / "cache").exists()


def test_fastembed_candidate_rejects_snapshot_change_during_backend_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = "snowflake-arctic-embed-l"
    cache_dir = tmp_path / "cache"
    snapshot = _create_pinned_snapshot(cache_dir, alias)

    class MutatingTextEmbedding:
        @staticmethod
        def list_supported_models() -> list[dict[str, Any]]:
            return [_fastembed_registry_entry(alias)]

        def __init__(self, **options: Any) -> None:
            (snapshot / "config.json").write_bytes(b"changed-during-load")
            self.model = SimpleNamespace(_model_dir=options["specific_model_path"])

    fastembed = ModuleType("fastembed")
    fastembed.TextEmbedding = MutatingTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)

    with pytest.raises(RuntimeError, match="snapshot changed while loading"):
        FastEmbedTextEmbedder(
            embedding_model_spec(alias),
            cache_dir=str(cache_dir),
            local_files_only=True,
        )


@pytest.mark.parametrize("alias", CUSTOM_E5_ALIASES)
def test_e5_custom_model_is_registered_once_and_uses_exact_pinned_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    import huggingface_hub

    cache_dir = tmp_path / "cache"
    snapshot = _create_pinned_snapshot(cache_dir, alias)
    registry: list[dict[str, Any]] = []
    registration_calls: list[dict[str, Any]] = []
    backend_calls: list[dict[str, Any]] = []
    download_calls: list[dict[str, Any]] = []

    class FakeTextEmbedding:
        @classmethod
        def list_supported_models(cls) -> list[dict[str, Any]]:
            return list(registry)

        @classmethod
        def add_custom_model(cls, **options: Any) -> None:
            registration_calls.append(options)
            registry.append(
                {
                    "model": options["model"],
                    "sources": {"hf": options["sources"].hf},
                    "model_file": options["model_file"],
                    "dim": options["dim"],
                }
            )

        def __init__(self, **options: Any) -> None:
            backend_calls.append(options)
            self.model = SimpleNamespace(_model_dir=options["specific_model_path"])

    def fake_snapshot_download(**options: Any) -> str:
        download_calls.append(options)
        return str(snapshot)

    fastembed = ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    spec = embedding_model_spec(alias)
    artifact = PINNED_FASTEMBED_ARTIFACTS[alias]
    before = _tree_state(tmp_path)

    online = FastEmbedTextEmbedder(spec, cache_dir=str(cache_dir))
    read_only = FastEmbedTextEmbedder(
        spec,
        cache_dir=str(cache_dir),
        local_files_only=True,
    )

    assert len(registration_calls) == 1
    registration = registration_calls[0]
    contract = CANDIDATE_CONTRACTS[alias]
    assert registration["model"] == contract["model_name"]
    assert registration["pooling"] is PoolingType.MEAN
    assert registration["normalization"] is True
    assert registration["sources"].hf == contract["fastembed_source"]
    assert registration["dim"] == contract["dimensions"]
    assert registration["model_file"] == contract["model_file"]
    assert registration["description"] == f"English {alias} retrieval embeddings"
    assert registration["license"] == "mit"
    assert registration["size_in_gb"] == contract["download_gb"]
    assert download_calls == [
        {
            "repo_id": artifact.repository,
            "revision": artifact.revision,
            "cache_dir": cache_dir,
            "allow_patterns": list(artifact.required_files),
        }
    ]
    assert len(backend_calls) == 2
    assert all(call["specific_model_path"] == str(snapshot) for call in backend_calls)
    assert all(call["local_files_only"] is True for call in backend_calls)
    assert online.artifact_fingerprint() == read_only.artifact_fingerprint()
    assert _tree_state(tmp_path) == before


@pytest.mark.parametrize("alias", CUSTOM_E5_ALIASES)
def test_e5_custom_model_read_only_missing_snapshot_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    registry: list[dict[str, Any]] = []
    backend_calls: list[dict[str, Any]] = []

    class FakeTextEmbedding:
        @classmethod
        def list_supported_models(cls) -> list[dict[str, Any]]:
            return list(registry)

        @classmethod
        def add_custom_model(cls, **options: Any) -> None:
            registry.append(
                {
                    "model": options["model"],
                    "sources": {"hf": options["sources"].hf},
                    "model_file": options["model_file"],
                    "dim": options["dim"],
                }
            )

        def __init__(self, **options: Any) -> None:
            backend_calls.append(options)

    fastembed = ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    cache_dir = tmp_path / "missing"

    with pytest.raises(RuntimeError, match="pinned local model snapshot is unavailable"):
        FastEmbedTextEmbedder(
            embedding_model_spec(alias),
            cache_dir=str(cache_dir),
            local_files_only=True,
        )

    assert backend_calls == []
    assert not cache_dir.exists()


@pytest.mark.parametrize("alias", API_EXPLICIT_CANDIDATE_ALIASES)
def test_semantic_index_request_accepts_fastembed_candidate_alias(alias: str) -> None:
    request = SemanticIndexRequest.model_validate({"model": alias, "rebuild_windows": False})

    assert request.model == alias


def test_semantic_index_request_still_rejects_unregistered_alias() -> None:
    with pytest.raises(ValidationError):
        SemanticIndexRequest(model="not-a-model")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        SemanticIndexRequest(model="e5-small-v2")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        SemanticIndexRequest(model="nomic-embed-text-v1.5")  # type: ignore[arg-type]
