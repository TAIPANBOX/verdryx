"""Tests for verdryx.pricing.

The `PriceBook.default()` numbers are a direct port of tokenfuse's
`crates/gateway/src/pricebook.rs` `default_price_book()`; several tests here
mirror that file's own `#[cfg(test)]` module (haiku's sane-range check, the
opus/fallback conservatism check, and the "resolves by exact match, not
fallback" check) so a units mistake or a copy-paste slip shows up on the
Python side the same way tokenfuse catches it on the Rust side.
"""

from __future__ import annotations

import pytest

from verdryx.pricing import ModelPrice, PriceBook

# ------------------------------------------------------------------
# ModelPrice.cost_usd
# ------------------------------------------------------------------


def test_model_price_cost_usd_sums_all_four_token_kinds() -> None:
    # Same figures as tokenfuse_core::pricing's own `sonnet()` test fixture:
    # input 3, output 15, cache read 0.3, cache write 3.75 (USD/Mtok).
    price = ModelPrice(3.0, 15.0, 0.30, 3.75)
    cost = price.cost_usd(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert cost == pytest.approx(22.05)


def test_model_price_cost_usd_defaults_cache_tokens_to_zero() -> None:
    price = ModelPrice(3.0, 15.0, 0.30, 3.75)
    assert price.cost_usd(input_tokens=1_000_000, output_tokens=0) == pytest.approx(3.0)


def test_model_price_cache_read_is_cheaper_than_fresh_input() -> None:
    price = ModelPrice(3.0, 15.0, 0.30, 3.75)
    cached = price.cost_usd(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000)
    fresh = price.cost_usd(input_tokens=1_000_000, output_tokens=0)
    assert cached < fresh
    assert cached == pytest.approx(0.30)


def test_model_price_cost_usd_large_token_counts_do_not_misbehave() -> None:
    price = ModelPrice(15.0, 75.0)
    # 5e9 tokens * $15/Mtok = $75,000.
    assert price.cost_usd(input_tokens=5_000_000_000, output_tokens=0) == pytest.approx(75_000.0)


# ------------------------------------------------------------------
# PriceBook: construction, exact match, fallback
# ------------------------------------------------------------------


def test_price_book_starts_empty() -> None:
    book = PriceBook()
    assert not book.is_known("anything")
    assert book.lookup("anything") is None


def test_price_book_with_price_returns_self_for_chaining() -> None:
    book = PriceBook()
    result = book.with_price("m", ModelPrice(1.0, 2.0))
    assert result is book


def test_price_book_with_fallback_returns_self_for_chaining() -> None:
    book = PriceBook()
    result = book.with_fallback(ModelPrice(1.0, 2.0))
    assert result is book


def test_price_book_exact_match_prices_a_known_model() -> None:
    book = PriceBook().with_price("m", ModelPrice(3.0, 15.0))
    assert book.is_known("m")
    # 1000 input * 3.0/1e6 + 500 output * 15.0/1e6 = 0.003 + 0.0075 = 0.0105
    assert book.price("m", input_tokens=1000, output_tokens=500) == pytest.approx(0.0105)


def test_price_book_unknown_model_without_fallback_raises() -> None:
    book = PriceBook().with_price("known", ModelPrice(1.0, 2.0))
    with pytest.raises(ValueError, match="no price for"):
        book.price("mystery-model", input_tokens=100, output_tokens=100)


def test_price_book_unknown_model_with_fallback_resolves_via_fallback() -> None:
    book = (
        PriceBook()
        .with_price("known", ModelPrice(1.0, 2.0))
        .with_fallback(ModelPrice(15.0, 75.0, 1.5, 18.75))
    )
    assert book.is_known("known")
    assert not book.is_known("mystery-model")
    # Fallback still resolves a price rather than raising.
    assert book.price("mystery-model", input_tokens=1_000_000, output_tokens=0) == pytest.approx(
        15.0
    )


# ------------------------------------------------------------------
# PriceBook.default(): mirrors tokenfuse's default_price_book() numbers
# ------------------------------------------------------------------

#: (model, input_per_mtok, output_per_mtok, cache_read_per_mtok,
#: cache_write_per_mtok), copied verbatim from tokenfuse's
#: crates/gateway/src/pricebook.rs `default_price_book()`.
_EXPECTED_DEFAULT_ENTRIES = [
    ("claude-sonnet", 3.0, 15.0, 0.30, 3.75),
    ("claude-haiku", 0.80, 4.0, 0.08, 1.0),
    ("gpt", 2.5, 10.0, 0.25, 3.125),
    ("claude-haiku-4-5", 1.00, 5.00, 0.10, 1.25),
    ("claude-haiku-4-5-20251001", 1.00, 5.00, 0.10, 1.25),
    ("claude-sonnet-4-5", 3.00, 15.00, 0.30, 3.75),
    ("claude-sonnet-4-5-20250929", 3.00, 15.00, 0.30, 3.75),
    ("claude-opus-4-5", 5.00, 25.00, 0.50, 6.25),
    ("claude-opus-4-5-20251101", 5.00, 25.00, 0.50, 6.25),
    ("gpt-4o", 2.50, 10.00, 1.25, 2.50),
    ("gpt-4o-mini", 0.15, 0.60, 0.075, 0.15),
    ("o1", 15.00, 60.00, 7.50, 15.00),
]


@pytest.mark.parametrize(
    "model,input_p,output_p,cache_read_p,cache_write_p", _EXPECTED_DEFAULT_ENTRIES
)
def test_price_book_default_entry_matches_tokenfuse_pricebook(
    model: str, input_p: float, output_p: float, cache_read_p: float, cache_write_p: float
) -> None:
    entry = PriceBook.default().lookup(model)
    assert entry == ModelPrice(input_p, output_p, cache_read_p, cache_write_p)


def test_price_book_default_new_2026_models_resolve_by_exact_match_not_fallback() -> None:
    book = PriceBook.default()
    for model in [
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-5",
        "claude-opus-4-5-20251101",
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
    ]:
        assert book.is_known(model), f"{model} should be an exact entry"
    # Still no exact entry for a genuinely unknown model -- it must fall
    # back rather than silently gaining a made-up price.
    assert not book.is_known("some-future-model-nobody-has-priced-yet")
    assert book.lookup("some-future-model-nobody-has-priced-yet") is not None


def test_price_book_default_generic_prefix_entries_are_distinct_from_dated_ones() -> None:
    """The illustrative generic entries ("claude-sonnet", "claude-haiku",
    "gpt") are their own rows, not aliases of the dated exact entries --
    verified by asserting claude-haiku's rate differs from
    claude-haiku-4-5's, matching tokenfuse's own price book."""
    book = PriceBook.default()
    assert book.lookup("claude-haiku") != book.lookup("claude-haiku-4-5")
    assert book.lookup("claude-sonnet") == book.lookup("claude-sonnet-4-5")


def test_price_book_default_haiku_4_5_sample_estimate_is_in_sane_milli_dollar_range() -> None:
    """Mirrors tokenfuse's own haiku_4_5_sample_estimate_is_in_sane_milli_dollar_range
    test: a 1000-input/500-output claude-haiku-4-5 call should land in the
    single-digit-to-low-tens-of-milli-dollars range, guarding against a
    units mistake (get the per-Mtok conversion wrong by 1e6 and this would
    report either sub-micro-dollar or multi-dollar costs instead)."""
    book = PriceBook.default()
    cost = book.price("claude-haiku-4-5", input_tokens=1000, output_tokens=500)
    # input: 1000 * $1.00/1e6 = $0.001; output: 500 * $5.00/1e6 = $0.0025
    # total = $0.0035 = 3.5 milli-dollars.
    assert cost == pytest.approx(0.0035)
    assert 0.0001 < cost < 1.0, f"cost {cost} is outside the sane milli-dollar range"


def test_price_book_default_fallback_stays_at_least_as_expensive_as_opus() -> None:
    """Mirrors tokenfuse's opus_4_5_is_the_most_expensive_entry_the_fallback_stays_conservative
    test (ADR-8): the fallback should remain at least as expensive as any
    known model, so an unrecognized model is never under-priced relative to
    what is actually known."""
    book = PriceBook.default()
    opus_cost = book.price("claude-opus-4-5", input_tokens=1_000_000, output_tokens=1_000_000)
    fallback_cost = book.price(
        "truly-unknown-model", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert fallback_cost >= opus_cost


def test_price_book_default_fallback_never_raises_for_unknown_model() -> None:
    # PriceBook.default() always configures a fallback, so an unknown model
    # resolves to a (conservative) price rather than raising ValueError.
    cost = PriceBook.default().price("anything-goes-here", input_tokens=1, output_tokens=1)
    assert cost > 0.0
