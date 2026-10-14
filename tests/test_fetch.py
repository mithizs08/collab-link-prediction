"""fetch_openalex: cursor paging, caching and retries, with a fake HTTP session."""

import json

import pytest

from linkpred.data import fetch_openalex as fo


class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params, timeout):
        self.calls.append(params)
        return self.responses.pop(0)


def page(results, next_cursor):
    return FakeResp(200, {"meta": {"next_cursor": next_cursor, "count": 3}, "results": results})


def test_paginates_and_caches(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.setenv("OPENALEX_MAILTO", "someone@example.org")
    s = FakeSession([page([{"id": 1}, {"id": 2}], "c2"), page([{"id": 3}], "c3"), page([], None)])
    n = fo.fetch(tmp_path, "I1", 2015, 2016, session=s, sleep=lambda _: None)
    assert n == 3
    assert [c["cursor"] for c in s.calls] == ["*", "c2", "c3"]
    assert s.calls[0]["mailto"] == "someone@example.org"
    # the manifest must not contain credentials
    assert "mailto" not in json.loads((tmp_path / "manifest.json").read_text())

    # Rerun: everything comes from cache, no HTTP calls.
    s2 = FakeSession([])
    assert fo.fetch(tmp_path, "I1", 2015, 2016, session=s2, sleep=lambda _: None) == 3
    assert s2.calls == []


def test_refuses_mixing_queries(tmp_path):
    fo.fetch(tmp_path, "I1", 2015, 2016, session=FakeSession([page([], None)]), sleep=lambda _: None)
    with pytest.raises(RuntimeError):
        fo.fetch(tmp_path, "I1", 2015, 2020, session=FakeSession([]), sleep=lambda _: None)


def test_retries_on_429_then_succeeds():
    waits = []
    s = FakeSession([FakeResp(429, headers={"Retry-After": "2"}), FakeResp(503), page([], None)])
    data = fo.get_with_retries(s, {}, sleep=waits.append)
    assert data["results"] == [] and waits == [2.0, 2.0]   # Retry-After, then backoff 1*2**1


def test_gives_up_after_max_retries():
    s = FakeSession([FakeResp(500)] * 3)
    with pytest.raises(RuntimeError):
        fo.get_with_retries(s, {}, max_retries=2, sleep=lambda _: None)
