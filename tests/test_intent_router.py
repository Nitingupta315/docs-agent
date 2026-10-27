"""Version matching in mcp-server/intent_router.py."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs-agent-mcp" / "mcp-server"))

from intent_router import (  # noqa: E402
    entity_matches_version,
    pick_search_plan,
    rerank_for_version_match,
)


@pytest.mark.parametrize(
    "entity",
    [
        {"version": "1.1"},
        {"version": "v1.1"},
        {"version": "1.1.0"},
        {"content_text": "Kubeflow v1.1 was released in 2020."},
        {"file_path": "content/en/docs/releases/kubeflow-1.1.md"},
        {"file_path": "content/en/docs/releases/kubeflow_1.1.md"},
    ],
)
def test_matches_same_version(entity):
    assert entity_matches_version(entity, ["1.1"])


@pytest.mark.parametrize(
    "entity",
    [
        {"version": "1.10"},
        {"version": "1.11.0"},
        {"version": "21.1"},
        {"content_text": "Kubeflow 1.10 release notes"},
        {"file_path": "content/en/docs/releases/kubeflow-1.10.md"},
    ],
)
def test_does_not_match_longer_version_with_same_prefix(entity):
    assert not entity_matches_version(entity, ["1.1"])


def test_api_version_does_not_match_longer_api_version():
    assert entity_matches_version({"content_text": "apiVersion: kubeflow.org/v1beta1"}, ["v1beta1"])
    assert not entity_matches_version({"content_text": "apiVersion: kubeflow.org/v1beta10"}, ["v1beta1"])


def test_release_date_question_keeps_only_the_asked_version():
    query = "When was Kubeflow 1.1 released?"
    hits = [
        {"distance": 0.9, "entity": {"version": "1.10", "content_text": "Kubeflow 1.10 release notes"}},
        {"distance": 0.8, "entity": {"version": "1.11", "content_text": "Kubeflow 1.11 release notes"}},
        {"distance": 0.7, "entity": {"version": "1.1", "content_text": "Kubeflow 1.1 release notes"}},
    ]

    assert pick_search_plan(query).prefer_version_match
    reranked = rerank_for_version_match(hits, query, top_k=5)

    assert [hit["entity"]["version"] for hit in reranked] == ["1.1"]
