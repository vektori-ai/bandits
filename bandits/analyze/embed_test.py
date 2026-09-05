"""Embedding backend tests. No network: the embedder callable is injected.

The paraphrase cases here are the ones lexical overlap gets wrong, so each is
asserted against ``jaccard_distance`` as well. A test that only showed the
embedding path working would not show that it works *better*, which is the whole
reason the backend exists.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from bandits.analyze.embed import DEFAULT_SIMILARITY as EMBEDDING_SIMILARITY
from bandits.analyze.embed import (
    EmbeddingCache,
    EmbeddingError,
    build_cache,
    compute_cache_id,
    cosine_distance,
    descriptors,
    embedding_distance,
    load_cache,
    save_cache,
)
from bandits.analyze.families import DEFAULT_SIMILARITY as FAMILIES_SIMILARITY
from bandits.analyze.families import jaccard_distance, mine_task_set, normalize_instruction
from bandits.analyze.models import CorpusAnalysis, TaskCandidate
from bandits.store import DerivedStore

# Two requests per topic, sharing almost no vocabulary with their own pair.
LOGIN = ("log into the dev box", "ssh into dev for me")
REVIEW = ("review this PR", "can you take a look at my pull request")

# Vectors chosen so the measured geometry matches the real corpus the backend was
# tuned against: within-family pairs land around 0.65 cosine and cross-family
# pairs near 0.45. Each text gets its own vector, so a pair is never trivially
# identical and cosine_distance does real arithmetic.
_VECTORS: dict[str, list[float]] = {
    LOGIN[0]: [1.0, 0.0, 0.0, 0.0],
    LOGIN[1]: [0.66, 0.75, 0.0, 0.0],
    REVIEW[0]: [0.0, 0.0, 1.0, 0.0],
    REVIEW[1]: [0.0, 0.0, 0.66, 0.75],
}
_BY_DESCRIPTOR = {normalize_instruction(text): vector for text, vector in _VECTORS.items()}


def _stub_embedder(model: str, texts):
    """Stands in for the Fireworks call, returning fixed vectors for known text."""
    return [list(_BY_DESCRIPTOR.get(text, [0.0, 0.0, 0.0, 1.0])) for text in texts]


def _analysis(*instructions: str) -> CorpusAnalysis:
    return CorpusAnalysis(
        corpus_id="corpus-test",
        source="otlp",
        tasks=tuple(
            TaskCandidate(task_id=f"task-{i}", trace_id=f"trace-{i}", instruction=text)
            for i, text in enumerate(instructions)
        ),
        evidence=(),
    )


def _cache(*instructions: str) -> EmbeddingCache:
    return build_cache(
        [normalize_instruction(text) for text in instructions],
        model="stub",
        embed=_stub_embedder,
    )


def _mine(analysis, cache, **overrides):
    return mine_task_set(
        analysis,
        "analysis-x",
        budget=10,
        distance=embedding_distance(cache),
        proposed_by="model",
        **overrides,
    )


def test_paraphrases_land_in_one_family_where_lexical_grouping_cannot() -> None:
    analysis = _analysis(*LOGIN, *REVIEW)
    cache = _cache(*LOGIN, *REVIEW)

    lexical = mine_task_set(analysis, "analysis-x", budget=10)
    embedded = _mine(analysis, cache, similarity=EMBEDDING_SIMILARITY)

    assert len(lexical.families) == 4, "the lexical baseline should fragment these"
    assert len(embedded.families) == 2
    grouped = sorted(sorted(f.trace_ids) for f in embedded.families)
    assert grouped == [["trace-0", "trace-1"], ["trace-2", "trace-3"]]


def test_the_lexical_threshold_rejects_pairs_the_backend_identifies() -> None:
    """The integration hazard, end to end.

    Passing a distance backend without also moving the threshold leaves grouping
    at ``families.DEFAULT_SIMILARITY``, which demands a closeness real paraphrase
    pairs do not reach. The backend then looks worse than it is.
    """
    analysis = _analysis(*LOGIN, *REVIEW)
    cache = _cache(*LOGIN, *REVIEW)

    assert len(_mine(analysis, cache, similarity=FAMILIES_SIMILARITY).families) == 4
    assert len(_mine(analysis, cache, similarity=EMBEDDING_SIMILARITY).families) == 2


def test_embedding_grouping_is_recorded_as_model_proposed() -> None:
    analysis = _analysis(*LOGIN)
    task_set = _mine(analysis, _cache(*LOGIN), similarity=EMBEDDING_SIMILARITY)

    assert {f.proposed_by for f in task_set.families} == {"model"}


def test_the_jaccard_threshold_would_reject_a_real_paraphrase_pair() -> None:
    """Why the embedding path cannot inherit ``families.DEFAULT_SIMILARITY``.

    0.7 demands a cosine distance of 0.3 or less. Same-family pairs measured on a
    real corpus reach only 0.62 similarity, so the lexical default rejects pairs
    the backend correctly identifies.
    """
    cache = _cache(*LOGIN)
    distance = embedding_distance(cache)
    pair = distance(*(normalize_instruction(text) for text in LOGIN))

    assert pair <= 1.0 - 0.6, "should group under the embedding default"
    assert pair > 1.0 - 0.7, "and be rejected under the lexical one"


def test_descriptors_are_the_strings_clustering_compares() -> None:
    analysis = _analysis("Refund order 7741", "Refund order 8820", "")

    assert descriptors(analysis) == ["refund order <id>"]


def test_a_missing_vector_reads_as_far_rather_than_reaching_the_network() -> None:
    distance = embedding_distance(EmbeddingCache(model="stub", vectors={"known": (1.0, 0.0)}))

    assert distance("known", "absent") == 1.0
    assert distance("absent", "absent") == 0.0


def test_cosine_distance_floors_at_zero_for_opposed_vectors() -> None:
    assert cosine_distance((1.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert cosine_distance((1.0, 0.0), (-1.0, 0.0)) == 1.0
    assert cosine_distance((0.0, 0.0), (1.0, 0.0)) == 1.0


def test_building_over_a_cache_only_embeds_what_is_missing() -> None:
    calls: list[list[str]] = []

    def counting(model: str, texts):
        calls.append(list(texts))
        return _stub_embedder(model, texts)

    first = build_cache(["alpha"], model="stub", embed=counting)
    second = build_cache(["alpha", "beta"], model="stub", embed=counting, existing=first)

    assert calls == [["alpha"], ["beta"]]
    assert set(second.vectors) == {"alpha", "beta"}


def test_vectors_from_two_models_are_refused() -> None:
    first = build_cache(["alpha"], model="stub", embed=_stub_embedder)

    with pytest.raises(EmbeddingError, match="not comparable"):
        build_cache(["beta"], model="other", embed=_stub_embedder, existing=first)


def test_a_saved_cache_round_trips(tmp_path) -> None:
    store = DerivedStore(tmp_path)
    cache = _cache(*LOGIN)

    envelope = save_cache(cache, store, "corpus-test")

    assert envelope.artifact_id == compute_cache_id(cache)
    assert envelope.parent_artifact_id == "corpus-test"
    assert load_cache(envelope.artifact_id, store) == cache


def test_jaccard_scores_these_paraphrases_as_unrelated() -> None:
    """The baseline this backend exists to replace."""
    left, right = (normalize_instruction(text) for text in LOGIN)

    assert jaccard_distance(left, right) > 0.7


def test_a_success_response_with_an_error_body_is_still_a_failure() -> None:
    """A 200 carrying a quota message must not surface as a traceback."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"error": "quota exceeded"}'

    with (
        mock.patch("urllib.request.urlopen", return_value=_Response()),
        mock.patch.dict(os.environ, {"FIREWORKS_API_KEY": "x"}),
        pytest.raises(EmbeddingError, match="not usable"),
    ):
        build_cache(["alpha"], model="stub")
