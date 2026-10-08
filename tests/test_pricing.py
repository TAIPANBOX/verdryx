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
#: cache_write_per_mtok), every row of tokenfuse's published price book,
#: `contracts/tokenfuse-constants.json` `price_book.models` at tokenfuse
#: commit ead13bc (main, 2026-10-07), in the artifact's own order (sorted by
#: model id). Converted from integer micro-USD per Mtok to USD per Mtok.
#: estate-gates' C3 compares the same four rates against the same artifact;
#: this list is the copy a verdryx test run can see without that repository.
_EXPECTED_DEFAULT_ENTRIES = [
    ("anthropic.claude-fable-5", 11.00, 55.00, 1.10, 13.75),
    ("anthropic.claude-fable-5-1", 11.00, 55.00, 0.275, 13.75),
    ("anthropic.claude-haiku-4-5", 1.10, 5.50, 0.11, 1.375),
    ("anthropic.claude-haiku-4-5-20251001-v1:0", 1.10, 5.50, 0.11, 1.375),
    ("anthropic.claude-opus-4-5-20251101-v1:0", 5.50, 27.50, 0.55, 6.875),
    ("anthropic.claude-opus-4-6-v1", 5.50, 27.50, 0.55, 6.875),
    ("anthropic.claude-opus-4-7", 5.50, 27.50, 0.55, 6.875),
    ("anthropic.claude-opus-4-8", 5.50, 27.50, 0.55, 6.875),
    ("anthropic.claude-opus-5", 5.50, 27.50, 0.55, 6.875),
    ("anthropic.claude-opus-5-5", 4.40, 22.00, 0.22, 5.50),
    ("anthropic.claude-sonnet-4-5-20250929-v1:0", 3.30, 16.50, 0.33, 4.125),
    ("anthropic.claude-sonnet-4-6", 3.30, 16.50, 0.33, 4.125),
    ("anthropic.claude-sonnet-5", 2.20, 11.00, 0.22, 2.75),
    ("anthropic.claude-sonnet-5-5", 2.20, 11.00, 0.22, 2.75),
    ("anthropic/claude-opus-4.8", 5.00, 25.00, 0.50, 6.25),
    ("anthropic/claude-opus-5", 5.00, 25.00, 0.50, 6.25),
    ("anthropic/claude-opus-5.5", 4.00, 20.00, 0.20, 5.00),
    ("anthropic/claude-sonnet-4.6", 3.00, 15.00, 0.30, 3.75),
    ("anthropic/claude-sonnet-5", 2.00, 10.00, 0.20, 2.50),
    ("anthropic/claude-sonnet-5.5", 2.00, 10.00, 0.20, 2.50),
    ("apac.anthropic.claude-opus-4-6-v1", 5.50, 27.50, 0.55, 6.875),
    ("claude-fable-5", 10.00, 50.00, 1.00, 12.50),
    ("claude-fable-5-1", 10.00, 50.00, 0.25, 12.50),
    ("claude-haiku", 0.80, 4.00, 0.08, 1.00),
    ("claude-haiku-4-5", 1.00, 5.00, 0.10, 1.25),
    ("claude-haiku-4-5-20251001", 1.00, 5.00, 0.10, 1.25),
    ("claude-haiku-4-5@20251001", 1.10, 5.50, 0.11, 1.375),
    ("claude-opus-4-5", 5.00, 25.00, 0.50, 6.25),
    ("claude-opus-4-5-20251101", 5.00, 25.00, 0.50, 6.25),
    ("claude-opus-4-5@20251101", 5.50, 27.50, 0.55, 6.875),
    ("claude-opus-4-6", 5.00, 25.00, 0.50, 6.25),
    ("claude-opus-4-7", 5.00, 25.00, 0.50, 6.25),
    ("claude-opus-4-8", 5.00, 25.00, 0.50, 6.25),
    ("claude-opus-5", 5.00, 25.00, 0.50, 6.25),
    ("claude-opus-5-5", 4.00, 20.00, 0.20, 5.00),
    ("claude-sonnet", 3.00, 15.00, 0.30, 3.75),
    ("claude-sonnet-4-5", 3.00, 15.00, 0.30, 3.75),
    ("claude-sonnet-4-5-20250929", 3.00, 15.00, 0.30, 3.75),
    ("claude-sonnet-4-5@20250929", 3.30, 16.50, 0.33, 4.125),
    ("claude-sonnet-4-6", 3.00, 15.00, 0.30, 3.75),
    ("claude-sonnet-5", 2.00, 10.00, 0.20, 2.50),
    ("claude-sonnet-5-5", 2.00, 10.00, 0.20, 2.50),
    ("eu.anthropic.claude-haiku-4-5-20251001-v1:0", 1.10, 5.50, 0.11, 1.375),
    ("eu.anthropic.claude-opus-4-5-20251101-v1:0", 5.50, 27.50, 0.55, 6.875),
    ("eu.anthropic.claude-opus-4-6-v1", 5.50, 27.50, 0.55, 6.875),
    ("eu.anthropic.claude-sonnet-4-5-20250929-v1:0", 3.30, 16.50, 0.33, 4.125),
    ("eu.anthropic.claude-sonnet-4-6", 3.30, 16.50, 0.33, 4.125),
    ("global.anthropic.claude-haiku-4-5-20251001-v1:0", 1.00, 5.00, 0.10, 1.25),
    ("global.anthropic.claude-opus-4-5-20251101-v1:0", 5.00, 25.00, 0.50, 6.25),
    ("global.anthropic.claude-opus-4-6-v1", 5.00, 25.00, 0.50, 6.25),
    ("global.anthropic.claude-sonnet-4-5-20250929-v1:0", 3.00, 15.00, 0.30, 3.75),
    ("global.anthropic.claude-sonnet-4-6", 3.00, 15.00, 0.30, 3.75),
    ("gpt", 2.50, 10.00, 0.25, 3.125),
    ("gpt-4o", 2.50, 10.00, 1.25, 2.50),
    ("gpt-4o-mini", 0.15, 0.60, 0.075, 0.15),
    ("jp.anthropic.claude-opus-4-6-v1", 5.50, 27.50, 0.55, 6.875),
    ("jp.anthropic.claude-sonnet-4-5-20250929-v1:0", 3.30, 16.50, 0.33, 4.125),
    ("jp.anthropic.claude-sonnet-4-6", 3.30, 16.50, 0.33, 4.125),
    ("o1", 15.00, 60.00, 7.50, 15.00),
    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", 1.10, 5.50, 0.11, 1.375),
    ("us.anthropic.claude-opus-4-5-20251101-v1:0", 5.50, 27.50, 0.55, 6.875),
    ("us.anthropic.claude-opus-4-6-v1", 5.50, 27.50, 0.55, 6.875),
    ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", 3.30, 16.50, 0.33, 4.125),
    ("us.anthropic.claude-sonnet-4-6", 3.30, 16.50, 0.33, 4.125),
]


@pytest.mark.parametrize(
    "model,input_p,output_p,cache_read_p,cache_write_p", _EXPECTED_DEFAULT_ENTRIES
)
def test_price_book_default_entry_matches_tokenfuse_pricebook(
    model: str, input_p: float, output_p: float, cache_read_p: float, cache_write_p: float
) -> None:
    entry = PriceBook.default().lookup(model)
    assert entry == ModelPrice(input_p, output_p, cache_read_p, cache_write_p)


def test_price_book_default_holds_exactly_the_pinned_models() -> None:
    """No row beyond the pinned list and none missing from it. The
    parametrised test above proves each pinned row is present and right; it
    cannot see a row verdryx holds that tokenfuse does not, which prices a
    model here that tokenfuse charges at the fallback rate."""
    book = PriceBook.default()
    assert set(book._prices) == {row[0] for row in _EXPECTED_DEFAULT_ENTRIES}


#: (id whose endpoint may be regional, the list-rate id of the same model).
#: Tokenfuse prices the first at list plus 10 percent (its `regional()`
#: helper in crates/gateway/src/pricebook.rs); a typo in one hand-written
#: rate shows here as a ratio that is not 1.1.
_REGIONAL_PAIRS = [
    ("anthropic.claude-fable-5-1", "claude-fable-5-1"),
    ("anthropic.claude-fable-5", "claude-fable-5"),
    ("anthropic.claude-opus-5-5", "claude-opus-5-5"),
    ("anthropic.claude-opus-5", "claude-opus-5"),
    ("anthropic.claude-opus-4-8", "claude-opus-4-8"),
    ("anthropic.claude-opus-4-7", "claude-opus-4-7"),
    ("anthropic.claude-opus-4-6-v1", "claude-opus-4-6"),
    ("us.anthropic.claude-opus-4-6-v1", "claude-opus-4-6"),
    ("eu.anthropic.claude-opus-4-6-v1", "claude-opus-4-6"),
    ("jp.anthropic.claude-opus-4-6-v1", "claude-opus-4-6"),
    ("apac.anthropic.claude-opus-4-6-v1", "claude-opus-4-6"),
    ("claude-opus-4-5@20251101", "claude-opus-4-5"),
    ("anthropic.claude-opus-4-5-20251101-v1:0", "claude-opus-4-5"),
    ("us.anthropic.claude-opus-4-5-20251101-v1:0", "claude-opus-4-5"),
    ("eu.anthropic.claude-opus-4-5-20251101-v1:0", "claude-opus-4-5"),
    ("anthropic.claude-sonnet-5-5", "claude-sonnet-5-5"),
    ("anthropic.claude-sonnet-5", "claude-sonnet-5"),
    ("anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("us.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("eu.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("jp.anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("claude-sonnet-4-5@20250929", "claude-sonnet-4-5"),
    ("anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
    ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
    ("eu.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
    ("jp.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
    ("claude-haiku-4-5@20251001", "claude-haiku-4-5"),
    ("anthropic.claude-haiku-4-5", "claude-haiku-4-5"),
    ("anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5"),
    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5"),
    ("eu.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5"),
]


@pytest.mark.parametrize("regional_id,list_id", _REGIONAL_PAIRS)
def test_price_book_default_regional_id_is_list_plus_ten_percent(
    regional_id: str, list_id: str
) -> None:
    book = PriceBook.default()
    assert book.is_known(regional_id), f"{regional_id} should be an exact entry"
    assert book.is_known(list_id), f"{list_id} should be an exact entry"
    regional = book.lookup(regional_id)
    listed = book.lookup(list_id)
    for got, base in [
        (regional.input_per_mtok_usd, listed.input_per_mtok_usd),
        (regional.output_per_mtok_usd, listed.output_per_mtok_usd),
        (regional.cache_read_per_mtok_usd, listed.cache_read_per_mtok_usd),
        (regional.cache_write_per_mtok_usd, listed.cache_write_per_mtok_usd),
    ]:
        assert got == pytest.approx(base * 1.1)


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
    # Opus 4.5 stopped being the dearest row: a regional Claude Fable id is
    # 11 / 55. Every row, so a dearer model added later fails here rather
    # than being under-priced by the fallback. Input and output only, which
    # is the claim tokenfuse's own price book makes ("nothing in the book is
    # dearer in input or output"): o1's cached input, 7.50, is above the
    # fallback's cache read, 1.50, in tokenfuse's book and so in this one.
    fallback = book.lookup("truly-unknown-model")
    for model, *_ in _EXPECTED_DEFAULT_ENTRIES:
        entry = book.lookup(model)
        assert fallback.input_per_mtok_usd >= entry.input_per_mtok_usd, model
        assert fallback.output_per_mtok_usd >= entry.output_per_mtok_usd, model


def test_price_book_default_fallback_never_raises_for_unknown_model() -> None:
    # PriceBook.default() always configures a fallback, so an unknown model
    # resolves to a (conservative) price rather than raising ValueError.
    cost = PriceBook.default().price("anything-goes-here", input_tokens=1, output_tokens=1)
    assert cost > 0.0
