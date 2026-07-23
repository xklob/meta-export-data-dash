from __future__ import annotations

import hashlib
import stat
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from meta_archive.api import enrichment_routes, search_routes
from meta_archive.paths import ArchiveLayout
from meta_archive.services import nlp_annotations as nlp_annotations_service
from meta_archive.services import search_index as search_index_service
from meta_archive.services.conflict_scoring import (
    DEFAULT_CONFLICT_SPEC,
    ModernBertNliConflictScorer,
)
from meta_archive.services.model_cache import (
    pinned_transformers_artifact,
    pinned_transformers_snapshot,
)
from meta_archive.services.nlp_annotations import MODEL_SPECS as NLP_MODEL_SPECS
from meta_archive.services.nlp_annotations import TransformersClassifier
from meta_archive.services.reranking import (
    DEFAULT_RERANKER_SPEC,
    TransformersCrossEncoderReranker,
)
from meta_archive.services.search_index import (
    FastEmbedTextEmbedder,
    TransformersTextEmbedder,
    embedding_model_spec,
)

_FLOAT32 = object()


def _tree_state(root: Path) -> tuple[tuple[str, int, int, int, str | None, str | None], ...]:
    def entry(path: Path) -> tuple[str, int, int, int, str | None, str | None]:
        value = path.lstat()
        return (
            path.relative_to(root).as_posix(),
            stat.S_IFMT(value.st_mode),
            stat.S_IMODE(value.st_mode),
            value.st_size,
            str(path.readlink()) if stat.S_ISLNK(value.st_mode) else None,
            hashlib.sha256(path.read_bytes()).hexdigest() if stat.S_ISREG(value.st_mode) else None,
        )

    return tuple(sorted(entry(path) for path in root.rglob("*")))


def _create_pinned_snapshot(cache_dir: Path, model_name: str, revision: str) -> Path:
    snapshot = cache_dir / f"models--{model_name.replace('/', '--')}" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    return snapshot


class _FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(id2label={0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"})
        self.to_calls: list[str] = []
        self.float_calls = 0
        self.eval_calls = 0

    def to(self, device: str) -> _FakeModel:
        self.to_calls.append(device)
        return self

    def float(self) -> _FakeModel:
        self.float_calls += 1
        return self

    def eval(self) -> _FakeModel:
        self.eval_calls += 1
        return self


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    list[tuple[str | Path, dict[str, object]]],
    list[tuple[str | Path, dict[str, object]]],
    object,
    _FakeModel,
]:
    tokenizer = object()
    model = _FakeModel()
    tokenizer_calls: list[tuple[str | Path, dict[str, object]]] = []
    model_calls: list[tuple[str | Path, dict[str, object]]] = []

    def load_tokenizer(source: str | Path, **options: object) -> object:
        tokenizer_calls.append((source, options))
        return tokenizer

    def load_model(source: str | Path, **options: object) -> _FakeModel:
        model_calls.append((source, options))
        return model

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=load_tokenizer)  # type: ignore[attr-defined]
    transformers.AutoModel = SimpleNamespace(from_pretrained=load_model)  # type: ignore[attr-defined]
    transformers.AutoModelForSequenceClassification = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=load_model
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return tokenizer_calls, model_calls, tokenizer, model


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[object, set[object], object]]:
    quantize_calls: list[tuple[object, set[object], object]] = []
    linear_type = object()
    qint8 = object()

    def get_num_threads() -> int:
        return 4

    def set_num_threads(_threads: int) -> None:
        return None

    def quantize_dynamic(model: object, layers: set[object], *, dtype: object) -> object:
        quantize_calls.append((model, layers, dtype))
        return model

    torch = ModuleType("torch")
    torch.get_num_threads = get_num_threads  # type: ignore[attr-defined]
    torch.set_num_threads = set_num_threads  # type: ignore[attr-defined]
    torch.nn = SimpleNamespace(Linear=linear_type)  # type: ignore[attr-defined]
    torch.qint8 = qint8  # type: ignore[attr-defined]
    torch.float32 = _FLOAT32  # type: ignore[attr-defined]
    torch.ao = SimpleNamespace(  # type: ignore[attr-defined]
        quantization=SimpleNamespace(quantize_dynamic=quantize_dynamic)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return quantize_calls


def _forbid_mkdir(
    _path: Path,
    mode: int = 0o777,
    parents: bool = False,
    exist_ok: bool = False,
) -> None:
    del mode, parents, exist_ok
    raise AssertionError("local-only model loading must not create directories")


def test_pinned_snapshot_resolver_refuses_missing_path_without_creating_anything(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "missing-cache"
    before = _tree_state(tmp_path)

    with pytest.raises(RuntimeError, match="pinned local model snapshot is unavailable"):
        pinned_transformers_snapshot(
            cache_dir,
            model_name="owner/model",
            model_revision="exact-revision",
        )

    assert not cache_dir.exists()
    assert _tree_state(tmp_path) == before


def test_pinned_snapshot_resolver_refuses_snapshot_symlink_without_mutation(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    target = tmp_path / "outside-snapshot"
    target.mkdir()
    snapshot = cache_dir / "models--owner--model" / "snapshots" / "exact-revision"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(target, target_is_directory=True)
    before = _tree_state(tmp_path)

    with pytest.raises(RuntimeError, match="pinned local model snapshot is unsafe"):
        pinned_transformers_snapshot(
            cache_dir,
            model_name="owner/model",
            model_revision="exact-revision",
        )

    assert snapshot.is_symlink()
    assert _tree_state(tmp_path) == before


def test_pinned_snapshot_resolver_refuses_external_leaf_symlink_without_mutation(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    snapshot = _create_pinned_snapshot(cache_dir, "owner/model", "exact-revision")
    outside = tmp_path / "outside-model.bin"
    outside.write_bytes(b"private outside bytes")
    (snapshot / "model.bin").symlink_to(outside)
    before = _tree_state(tmp_path)

    with pytest.raises(RuntimeError, match="pinned local model snapshot is unsafe"):
        pinned_transformers_snapshot(
            cache_dir,
            model_name="owner/model",
            model_revision="exact-revision",
        )

    assert _tree_state(tmp_path) == before


def test_pinned_snapshot_resolver_accepts_hugging_face_blob_symlink_without_mutation(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    snapshot = _create_pinned_snapshot(cache_dir, "owner/model", "exact-revision")
    model_cache = cache_dir / "models--owner--model"
    blob = model_cache / "blobs" / "content-hash"
    blob.parent.mkdir()
    blob.write_bytes(b"model bytes")
    (snapshot / "model.bin").symlink_to(Path("../../blobs/content-hash"))
    before = _tree_state(tmp_path)

    resolved = pinned_transformers_snapshot(
        cache_dir,
        model_name="owner/model",
        model_revision="exact-revision",
    )

    assert resolved == snapshot
    assert _tree_state(tmp_path) == before


def test_pinned_snapshot_artifact_hashes_exact_blob_bytes_and_only_requested_snapshot(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    snapshot = _create_pinned_snapshot(cache_dir, "owner/model", "exact-revision")
    model_cache = cache_dir / "models--owner--model"
    blob = model_cache / "blobs" / "filename-is-not-trusted"
    blob.parent.mkdir()
    blob.write_bytes(b"MODEL_BYTES")
    (snapshot / "model.bin").symlink_to(Path("../../blobs/filename-is-not-trusted"))
    (snapshot / "config.json").write_bytes(b"{}")

    first = pinned_transformers_artifact(
        cache_dir,
        model_name="owner/model",
        model_revision="exact-revision",
    )
    unrelated = _create_pinned_snapshot(cache_dir, "owner/model", "other-revision")
    (unrelated / "model.bin").write_bytes(b"UNRELATED")
    second = pinned_transformers_artifact(
        cache_dir,
        model_name="owner/model",
        model_revision="exact-revision",
    )

    assert first == second
    assert first.file_count == 2
    assert first.total_bytes == len(b"MODEL_BYTES{}")
    assert len(first.fingerprint_sha256) == 64

    blob.write_bytes(b"OTHER_BYTES")
    changed = pinned_transformers_artifact(
        cache_dir,
        model_name="owner/model",
        model_revision="exact-revision",
    )
    assert changed.fingerprint_sha256 != first.fingerprint_sha256


def test_pinned_snapshot_resolver_refuses_chained_blob_symlink_without_mutation(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    snapshot = _create_pinned_snapshot(cache_dir, "owner/model", "exact-revision")
    model_cache = cache_dir / "models--owner--model"
    outside = tmp_path / "outside-model.bin"
    outside.write_bytes(b"private outside bytes")
    blob = model_cache / "blobs" / "content-hash"
    blob.parent.mkdir()
    blob.symlink_to(outside)
    (snapshot / "model.bin").symlink_to(Path("../../blobs/content-hash"))
    before = _tree_state(tmp_path)

    with pytest.raises(RuntimeError, match="pinned local model snapshot is unsafe"):
        pinned_transformers_snapshot(
            cache_dir,
            model_name="owner/model",
            model_revision="exact-revision",
        )

    assert _tree_state(tmp_path) == before


@pytest.mark.parametrize("model_alias", ["multilingual-e5-small", "gte-modernbert-base"])
def test_transformers_embedder_loads_exact_local_snapshot_without_mkdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_alias: str,
) -> None:
    spec = embedding_model_spec(model_alias)
    cache_dir = tmp_path / "transformers"
    snapshot = _create_pinned_snapshot(cache_dir, spec.model_name, spec.model_revision)
    before = _tree_state(tmp_path)
    tokenizer_calls, model_calls, tokenizer, model = _install_fake_transformers(monkeypatch)
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(Path, "mkdir", _forbid_mkdir)

    embedder = TransformersTextEmbedder(
        spec,
        cache_dir=str(cache_dir),
        local_files_only=True,
    )

    assert embedder.spec == spec
    assert embedder._tokenizer is tokenizer
    assert embedder._model is model
    assert tokenizer_calls == [(snapshot, {"local_files_only": True})]
    assert model_calls == [(snapshot, {"dtype": _FLOAT32, "local_files_only": True})]
    assert model.to_calls == ["cpu"]
    assert model.float_calls == 1
    assert model.eval_calls == 1
    assert _tree_state(tmp_path) == before


def test_nlp_classifier_fingerprints_and_loads_exact_local_snapshot_without_mkdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = NLP_MODEL_SPECS["sentiment"]
    cache_root = tmp_path / "nlp"
    location = cache_root / spec.task / spec.model_revision
    snapshot = _create_pinned_snapshot(location, spec.model_name, spec.model_revision)
    (snapshot / "config.json").write_bytes(b"{}")
    before = _tree_state(tmp_path)
    tokenizer_calls, model_calls, _tokenizer, model = _install_fake_transformers(monkeypatch)
    monkeypatch.setattr(Path, "mkdir", _forbid_mkdir)

    classifier = TransformersClassifier(cache_root)
    classifier.prepare(spec)
    configuration = classifier.inference_configuration(spec)

    assert tokenizer_calls == [(snapshot, {"local_files_only": True})]
    assert model_calls == [(snapshot, {"local_files_only": True})]
    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1
    assert configuration.payload["model_artifact"]["file_count"] == 1
    assert len(configuration.payload["model_artifact"]["sha256"]) == 64
    assert configuration.payload["implementation"]["provider"] == (
        "transformers-auto-sequence-classification"
    )
    assert len(configuration.sha256) == 64
    assert _tree_state(tmp_path) == before


def test_nlp_classifier_refuses_snapshot_fingerprint_change_during_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = NLP_MODEL_SPECS["sentiment"]
    cache_root = tmp_path / "nlp"
    location = cache_root / spec.task / spec.model_revision
    snapshot = _create_pinned_snapshot(location, spec.model_name, spec.model_revision)
    (snapshot / "config.json").write_bytes(b"{}")
    artifact = pinned_transformers_artifact(
        location,
        model_name=spec.model_name,
        model_revision=spec.model_revision,
    )
    artifacts = iter(
        (
            artifact,
            replace(artifact, fingerprint_sha256="0" * 64),
        )
    )
    _install_fake_transformers(monkeypatch)
    monkeypatch.setattr(
        nlp_annotations_service,
        "pinned_transformers_artifact",
        lambda *_args, **_kwargs: next(artifacts),
    )

    with pytest.raises(RuntimeError, match="changed during model loading"):
        TransformersClassifier(cache_root).prepare(spec)


def test_reranker_loads_exact_local_snapshot_without_mkdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "reranker"
    snapshot = _create_pinned_snapshot(
        cache_dir,
        DEFAULT_RERANKER_SPEC.model_name,
        DEFAULT_RERANKER_SPEC.model_revision,
    )
    before = _tree_state(tmp_path)
    tokenizer_calls, model_calls, tokenizer, model = _install_fake_transformers(monkeypatch)
    monkeypatch.setattr(Path, "mkdir", _forbid_mkdir)
    reranker = TransformersCrossEncoderReranker(cache_dir, local_files_only=True)

    loaded_tokenizer, loaded_model = reranker._load()

    assert loaded_tokenizer is tokenizer
    assert loaded_model is model
    assert tokenizer_calls == [(snapshot, {"local_files_only": True})]
    assert model_calls == [(snapshot, {"local_files_only": True})]
    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1
    assert _tree_state(tmp_path) == before


def test_reranker_local_only_missing_cache_fails_closed_without_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "missing-reranker"
    tokenizer_calls, model_calls, _, _ = _install_fake_transformers(monkeypatch)
    before = _tree_state(tmp_path)

    with pytest.raises(RuntimeError, match="could not be loaded at its pinned revision"):
        TransformersCrossEncoderReranker(cache_dir, local_files_only=True)._load()

    assert tokenizer_calls == []
    assert model_calls == []
    assert not cache_dir.exists()
    assert _tree_state(tmp_path) == before


def test_conflict_scorer_loads_exact_local_snapshot_without_mkdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "conflict"
    snapshot = _create_pinned_snapshot(
        cache_dir,
        DEFAULT_CONFLICT_SPEC.model_name,
        DEFAULT_CONFLICT_SPEC.model_revision,
    )
    before = _tree_state(tmp_path)
    tokenizer_calls, model_calls, tokenizer, model = _install_fake_transformers(monkeypatch)
    quantize_calls = _install_fake_torch(monkeypatch)
    monkeypatch.setattr(Path, "mkdir", _forbid_mkdir)
    scorer = ModernBertNliConflictScorer(cache_dir, local_files_only=True)

    loaded_tokenizer, loaded_model, entailment_index = scorer._load()

    assert loaded_tokenizer is tokenizer
    assert loaded_model is model
    assert entailment_index == 1
    assert tokenizer_calls == [(snapshot, {"local_files_only": True})]
    assert model_calls == [(snapshot, {"local_files_only": True})]
    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 2
    assert quantize_calls == [(model, {sys.modules["torch"].nn.Linear}, sys.modules["torch"].qint8)]  # type: ignore[attr-defined]
    assert _tree_state(tmp_path) == before


def test_conflict_scorer_local_only_missing_cache_fails_closed_without_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "missing-conflict"
    tokenizer_calls, model_calls, _, _ = _install_fake_transformers(monkeypatch)
    _install_fake_torch(monkeypatch)
    before = _tree_state(tmp_path)

    with pytest.raises(RuntimeError, match="could not be loaded"):
        ModernBertNliConflictScorer(cache_dir, local_files_only=True)._load()

    assert tokenizer_calls == []
    assert model_calls == []
    assert not cache_dir.exists()
    assert _tree_state(tmp_path) == before


def test_fastembed_normal_mode_constructs_backend_without_local_only_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "fastembed"
    cache_dir.mkdir()
    calls: list[dict[str, Any]] = []

    class FakeTextEmbedding:
        def __init__(self, **options: Any) -> None:
            calls.append(options)
            self.model = SimpleNamespace(_model_dir=cache_dir)

    fastembed = ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    spec = embedding_model_spec("bge-small-en-v1.5")

    embedder = FastEmbedTextEmbedder(
        spec,
        cache_dir=str(cache_dir),
        local_files_only=False,
    )

    assert embedder.spec == spec
    assert calls == [
        {
            "model_name": spec.model_name,
            "cache_dir": str(cache_dir),
            "threads": 8,
        }
    ]


def test_fastembed_local_only_refuses_existing_cache_before_backend_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class ForbiddenTextEmbedding:
        def __init__(self, **options: Any) -> None:
            calls.append(options)

    fastembed = ModuleType("fastembed")
    fastembed.TextEmbedding = ForbiddenTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    cache_dir = tmp_path / "existing-fastembed"
    cache_dir.mkdir()
    before = _tree_state(tmp_path)

    with pytest.raises(RuntimeError, match="cannot safely initialize a FastEmbed cache"):
        FastEmbedTextEmbedder(
            embedding_model_spec("bge-small-en-v1.5"),
            cache_dir=str(cache_dir),
            local_files_only=True,
        )

    assert calls == []
    assert _tree_state(tmp_path) == before


def test_fastembed_local_only_missing_cache_fails_before_backend_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class ForbiddenTextEmbedding:
        def __init__(self, **options: Any) -> None:
            calls.append(options)

    fastembed = ModuleType("fastembed")
    fastembed.TextEmbedding = ForbiddenTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    cache_dir = tmp_path / "missing-fastembed"
    before = _tree_state(tmp_path)

    with pytest.raises(RuntimeError, match="cannot safely initialize a FastEmbed cache"):
        FastEmbedTextEmbedder(
            embedding_model_spec("bge-small-en-v1.5"),
            cache_dir=str(cache_dir),
            local_files_only=True,
        )

    assert calls == []
    assert not cache_dir.exists()
    assert _tree_state(tmp_path) == before


@pytest.mark.parametrize("local_files_only", [False, True])
def test_default_embedder_forwards_local_only_mode_to_fastembed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_files_only: bool,
) -> None:
    layout = ArchiveLayout(tmp_path / "archive")
    spec = embedding_model_spec("bge-small-en-v1.5")
    calls: list[tuple[object, str, bool]] = []
    sentinel = object()

    def fake_fastembed(
        received_spec: object,
        *,
        cache_dir: str,
        local_files_only: bool = False,
    ) -> object:
        calls.append((received_spec, cache_dir, local_files_only))
        return sentinel

    monkeypatch.setattr(search_index_service, "FastEmbedTextEmbedder", fake_fastembed)

    assert (
        search_index_service.default_text_embedder(
            layout,
            spec,
            local_files_only=local_files_only,
        )
        is sentinel
    )
    assert calls == [(spec, str(layout.models / "fastembed"), local_files_only)]


@pytest.mark.parametrize("acceptance_read_only", [False, True])
def test_search_api_factories_forward_acceptance_local_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    acceptance_read_only: bool,
) -> None:
    layout = ArchiveLayout(tmp_path / f"archive-{acceptance_read_only}")
    state = SimpleNamespace(
        layout=layout,
        acceptance_read_only=acceptance_read_only,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    embedder_calls: list[tuple[ArchiveLayout, object, bool]] = []
    reranker_calls: list[tuple[Path, object, bool]] = []
    embedder_sentinel = object()
    reranker_sentinel = object()

    def fake_embedder(
        received_layout: ArchiveLayout,
        spec: object,
        *,
        local_files_only: bool,
    ) -> object:
        embedder_calls.append((received_layout, spec, local_files_only))
        return embedder_sentinel

    def fake_reranker(
        cache_dir: Path,
        spec: object,
        *,
        local_files_only: bool,
    ) -> object:
        reranker_calls.append((cache_dir, spec, local_files_only))
        return reranker_sentinel

    monkeypatch.setattr(search_routes, "default_text_embedder", fake_embedder)
    monkeypatch.setattr(search_routes, "TransformersCrossEncoderReranker", fake_reranker)
    embedding_spec = embedding_model_spec("multilingual-e5-small")

    assert (
        search_routes._cached_search_embedder_factory(request)(embedding_spec) is embedder_sentinel
    )
    assert (
        search_routes._cached_search_reranker_factory(request)(DEFAULT_RERANKER_SPEC)
        is reranker_sentinel
    )
    assert embedder_calls == [(layout, embedding_spec, acceptance_read_only)]
    assert reranker_calls == [
        (
            layout.models / "reranker" / DEFAULT_RERANKER_SPEC.model_revision,
            DEFAULT_RERANKER_SPEC,
            acceptance_read_only,
        )
    ]


@pytest.mark.parametrize("acceptance_read_only", [False, True])
def test_enrichment_api_factory_forwards_acceptance_local_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    acceptance_read_only: bool,
) -> None:
    layout = ArchiveLayout(tmp_path / f"archive-{acceptance_read_only}")
    state = SimpleNamespace(
        layout=layout,
        acceptance_read_only=acceptance_read_only,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    calls: list[tuple[Path, object, bool]] = []
    sentinel = object()

    def fake_scorer(
        cache_dir: Path,
        spec: object,
        *,
        local_files_only: bool,
    ) -> object:
        calls.append((cache_dir, spec, local_files_only))
        return sentinel

    monkeypatch.setattr(enrichment_routes, "ModernBertNliConflictScorer", fake_scorer)

    assert (
        enrichment_routes._cached_conflict_scorer_factory(request)(DEFAULT_CONFLICT_SPEC)
        is sentinel
    )
    assert calls == [
        (
            layout.models / "conflict" / DEFAULT_CONFLICT_SPEC.model_revision,
            DEFAULT_CONFLICT_SPEC,
            acceptance_read_only,
        )
    ]
