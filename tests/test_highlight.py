"""Unit tests for claim parsing, span finding, score normalization, rendering."""

from highlight import _esc, find_span, normalize_scores, parse_claims, render_html


def test_parse_claims_valid_json():
    assert parse_claims('["alpha", "beta"]', "fallback") == ["alpha", "beta"]


def test_parse_claims_json_embedded_in_noise():
    raw = 'Here are the claims: ["a", "b", "c"] -- done.'
    assert parse_claims(raw, "fallback") == ["a", "b", "c"]


def test_parse_claims_strips_and_drops_empty():
    assert parse_claims('["  x  ", "", "y"]', "fb") == ["x", "y"]


def test_parse_claims_fallback_on_invalid_json():
    assert parse_claims("not json at all", "One fact. Two facts!") == ["One fact.", "Two facts!"]


def test_parse_claims_fallback_on_non_string_list():
    assert parse_claims("[1, 2, 3]", "A. B.") == ["A.", "B."]


def test_normalize_scores_probabilities_unchanged():
    assert normalize_scores([0.1, 0.5, 0.9]) == [0.1, 0.5, 0.9]


def test_normalize_scores_minmax_for_unbounded():
    assert normalize_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_normalize_scores_constant_is_neutral():
    assert normalize_scores([3.0, 3.0]) == [0.5, 0.5]


def test_normalize_scores_empty():
    assert normalize_scores([]) == []


def test_find_span_exact():
    answer = "The sky is blue today."
    start, end = find_span("sky is blue", answer)
    assert answer[start:end] == "sky is blue"


def test_find_span_case_insensitive():
    span = find_span("paris is the capital", "Paris Is The Capital.")
    assert span is not None and span[0] == 0


def test_find_span_no_match_returns_none():
    assert find_span("completely unrelated text here", "short answer") is None


def test_esc_escapes_html():
    assert _esc('<script>&"') == '&lt;script&gt;&amp;"'


def test_render_html_escapes_and_wraps():
    html = render_html("a<b", [{"claim": "a<b", "score": 0.5, "span": (0, 3)}])
    assert html.startswith("<p>") and html.endswith("</p>")
    assert "&lt;" in html
    assert "<script>" not in html


def test_render_html_unmatched_span_still_renders_text():
    html = render_html("hello", [{"claim": "x", "score": 0.9, "span": None}])
    assert "hello" in html
