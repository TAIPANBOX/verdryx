"""Token -> USD pricing, mirroring TokenFuse's default price book.

TokenFuse (the sibling FinOps-enforcement service in the TAIPANBOX stack)
ships a price book of USD-per-million-token rates for the models it proxies
(``crates/gateway/src/pricebook.rs``, built on ``tokenfuse_core::PriceBook``
in ``crates/core/src/pricing.rs``). Verdryx needs the same numbers for one
job: turning an LLM judge's token usage into a dollar cost, so
``Score.cost_usd`` no longer has to sit at 0.0 for a grader that actually
calls a model (see ``models.py``'s ``Score`` docstring). This module is a
small, dependency-free port of just the pricing piece Verdryx needs; it does
not attempt to replicate tokenfuse's budget/ledger/settlement machinery.

Lookup mirrors tokenfuse's ``PriceBook`` exactly: an unknown model resolves
via a flat, deliberately conservative fallback (never an under-estimate)
rather than raising, matching tokenfuse's ADR-8. Resolution is exact-match
only, same as ``tokenfuse_core::pricing::PriceBook::price`` (``self.prices.
get(model).copied().or(self.fallback)`` -- no prefix or fuzzy matching there
either), so a model string has to match one of the entries below verbatim to
avoid the fallback rate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """USD price for one model, per million tokens ("per Mtok"), matching
    how providers publish their rates.

    Mirrors ``tokenfuse_core::pricing::ModelPrice`` field-for-field and
    argument-order-for-argument-order with its ``per_mtok_usd(input,
    output, cache_read, cache_write)`` constructor, so the numbers copied
    from tokenfuse's price book need no reshaping. Verdryx's own caller
    (the LLM judge path in graders.py) only ever reports input and output
    tokens today, but cache_read/cache_write are carried here anyway for
    parity and for a future caller with cache-aware usage data.
    """

    input_per_mtok_usd: float
    output_per_mtok_usd: float
    cache_read_per_mtok_usd: float = 0.0
    cache_write_per_mtok_usd: float = 0.0

    def cost_usd(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Exact USD cost of one call's usage against this price."""
        return (
            input_tokens * self.input_per_mtok_usd
            + output_tokens * self.output_per_mtok_usd
            + cache_read_tokens * self.cache_read_per_mtok_usd
            + cache_write_tokens * self.cache_write_per_mtok_usd
        ) / 1_000_000


class PriceBook:
    """Model name -> ModelPrice lookup, with an optional conservative
    fallback for models with no exact entry.

    Empty until populated via `with_price`/`with_fallback` (mirroring
    tokenfuse_core::PriceBook's own empty-by-default, builder-style
    construction); `PriceBook.default()` returns the populated book Verdryx's
    built-in adapters use unless a caller injects their own, mirroring how
    tokenfuse's gateway builds its `default_price_book()` on top of the
    generic `PriceBook` container.
    """

    def __init__(self) -> None:
        self._prices: dict[str, ModelPrice] = {}
        self._fallback: ModelPrice | None = None

    def with_price(self, model: str, price: ModelPrice) -> PriceBook:
        """Register an exact-match entry. Returns self, for chaining."""
        self._prices[model] = price
        return self

    def with_fallback(self, price: ModelPrice) -> PriceBook:
        """Set the price used for a model with no exact entry. Returns
        self, for chaining."""
        self._fallback = price
        return self

    def is_known(self, model: str) -> bool:
        """Whether `model` resolves via an exact entry, as opposed to the
        fallback."""
        return model in self._prices

    def lookup(self, model: str) -> ModelPrice | None:
        """The ModelPrice for `model`: its exact entry if registered, else
        the fallback, else None if no fallback is configured."""
        return self._prices.get(model, self._fallback)

    def price(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """USD cost of one call against this book.

        Raises:
            ValueError: If `model` has no exact entry and no fallback is
                configured. `PriceBook.default()` always sets a fallback,
                so this only fires for a caller-built book that deliberately
                left one out.
        """
        entry = self.lookup(model)
        if entry is None:
            raise ValueError(f"no price for {model!r} and PriceBook has no fallback configured")
        return entry.cost_usd(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)

    @classmethod
    def default(cls) -> PriceBook:
        """The price book Verdryx's built-in adapters use unless a caller
        injects their own.

        Ported number-for-number from tokenfuse's
        `crates/gateway/src/pricebook.rs` `default_price_book()`. Prices as
        of 2026-07; verify against
        https://platform.claude.com/docs/en/about-claude/pricing and
        https://developers.openai.com/api/docs/pricing before relying on
        them for anything beyond a rough estimate, same disclaimer
        tokenfuse's own price book carries.
        """
        return (
            cls()
            # Illustrative generic entries (kept for callers that pass a
            # bare family name rather than a real dated provider model id,
            # the same role they play in tokenfuse's book).
            .with_price("claude-sonnet", ModelPrice(3.0, 15.0, 0.30, 3.75))
            .with_price("claude-haiku", ModelPrice(0.80, 4.0, 0.08, 1.0))
            .with_price("gpt", ModelPrice(2.5, 10.0, 0.25, 3.125))
            #
            # Anthropic, current lineup. Cache write/read follow Anthropic's
            # published multiplier off the input rate (5-minute TTL: write
            # = 1.25x input, read = 0.1x input). Both the "latest" alias and
            # the dated snapshot are entered so either resolves exactly.
            .with_price("claude-haiku-4-5", ModelPrice(1.00, 5.00, 0.10, 1.25))
            .with_price("claude-haiku-4-5-20251001", ModelPrice(1.00, 5.00, 0.10, 1.25))
            .with_price("claude-sonnet-4-5", ModelPrice(3.00, 15.00, 0.30, 3.75))
            .with_price("claude-sonnet-4-5-20250929", ModelPrice(3.00, 15.00, 0.30, 3.75))
            .with_price("claude-opus-4-5", ModelPrice(5.00, 25.00, 0.50, 6.25))
            .with_price("claude-opus-4-5-20251101", ModelPrice(5.00, 25.00, 0.50, 6.25))
            #
            # OpenAI, current lineup. No separate cache-write fee (the
            # first pass through is billed as ordinary input), so
            # cache_write is set equal to input; cached-read is a flat 50%
            # off input across these models.
            .with_price("gpt-4o", ModelPrice(2.50, 10.00, 1.25, 2.50))
            .with_price("gpt-4o-mini", ModelPrice(0.15, 0.60, 0.075, 0.15))
            .with_price("o1", ModelPrice(15.00, 60.00, 7.50, 15.00))
            #
            # Conservative fallback for anything not listed above (mirrors
            # tokenfuse's ADR-8): priced at 3x Opus 4.5, the most expensive
            # known model, so an unrecognized model is never under-priced
            # relative to what is actually known.
            .with_fallback(ModelPrice(15.0, 75.0, 1.5, 18.75))
        )
