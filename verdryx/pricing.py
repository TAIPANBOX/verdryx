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

        Ported number-for-number from tokenfuse's published price book,
        `contracts/tokenfuse-constants.json` (`price_book.models`), at
        tokenfuse commit ead13bc (main, 2026-10-07), which is generated from
        `crates/gateway/src/pricebook.rs` `default_price_book()`. Anthropic
        rates as tokenfuse read them on 2026-10-07; OpenAI rates as of
        2026-07. Verify against
        https://platform.claude.com/docs/en/about-claude/pricing and
        https://developers.openai.com/api/docs/pricing before relying on
        them for anything beyond a rough estimate, same disclaimer
        tokenfuse's own price book carries.

        Not mirrored: tokenfuse's fifth rate, the 1-hour cache write
        (`cache_write_1h_per_mtok`). `ModelPrice` here carries four rates
        and no caller reports a 1-hour subset, so it has nothing to price.
        estate-gates' C3 compares the four rates carried here.
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
            # Anthropic: every listed Claude id, as tokenfuse prices it. Ids
            # sold at the list rate (the Claude API, Google Cloud's dateless
            # ids, a Bedrock `global.` profile, OpenRouter) carry the list
            # rate; an id whose endpoint may be regional (a Bedrock `us.`,
            # `eu.`, `jp.` or `apac.` profile, a bare Bedrock `anthropic.*`
            # id, a Google `@`-dated id) carries list plus 10 percent, the
            # over-charging side. Tokenfuse builds these rows from one table
            # with a `regional()` helper; they are written out one by one
            # here because estate-gates' C3 reads this mirror as literal
            # `.with_price(...)` calls, and a loop would hide them from it.
            # The comment above each family is tokenfuse's own, verbatim.
            #
            # Claude Fable 5.1: 10 / 50, cache hits 0.25 (0.025x), writes 12.50 / 20.
            .with_price("claude-fable-5-1", ModelPrice(10.00, 50.00, 0.25, 12.50))
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("anthropic.claude-fable-5-1", ModelPrice(11.00, 55.00, 0.275, 13.75))
            #
            # Claude Fable 5: 10 / 50, cache hits 1, writes 12.50 / 20.
            .with_price("claude-fable-5", ModelPrice(10.00, 50.00, 1.00, 12.50))
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("anthropic.claude-fable-5", ModelPrice(11.00, 55.00, 1.10, 13.75))
            #
            # Claude Opus 5.5: 4 / 20, cache hits 0.20 (0.05x), writes 5 / 8.
            .with_price("claude-opus-5-5", ModelPrice(4.00, 20.00, 0.20, 5.00))
            .with_price("anthropic/claude-opus-5.5", ModelPrice(4.00, 20.00, 0.20, 5.00))
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("anthropic.claude-opus-5-5", ModelPrice(4.40, 22.00, 0.22, 5.50))
            #
            # Claude Opus 5: 5 / 25, cache hits 0.50, writes 6.25 / 10.
            .with_price("claude-opus-5", ModelPrice(5.00, 25.00, 0.50, 6.25))
            .with_price("anthropic/claude-opus-5", ModelPrice(5.00, 25.00, 0.50, 6.25))
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("anthropic.claude-opus-5", ModelPrice(5.50, 27.50, 0.55, 6.875))
            #
            # Claude Opus 4.8: as Opus 5.
            .with_price("claude-opus-4-8", ModelPrice(5.00, 25.00, 0.50, 6.25))
            .with_price("anthropic/claude-opus-4.8", ModelPrice(5.00, 25.00, 0.50, 6.25))
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("anthropic.claude-opus-4-8", ModelPrice(5.50, 27.50, 0.55, 6.875))
            #
            # Claude Opus 4.7: as Opus 5.
            .with_price("claude-opus-4-7", ModelPrice(5.00, 25.00, 0.50, 6.25))
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("anthropic.claude-opus-4-7", ModelPrice(5.50, 27.50, 0.55, 6.875))
            #
            # Claude Opus 4.6: as Opus 5. Bedrock profiles: global, us, eu, jp, apac.
            .with_price("claude-opus-4-6", ModelPrice(5.00, 25.00, 0.50, 6.25))
            .with_price("global.anthropic.claude-opus-4-6-v1", ModelPrice(5.00, 25.00, 0.50, 6.25))
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("anthropic.claude-opus-4-6-v1", ModelPrice(5.50, 27.50, 0.55, 6.875))
            .with_price("us.anthropic.claude-opus-4-6-v1", ModelPrice(5.50, 27.50, 0.55, 6.875))
            .with_price("eu.anthropic.claude-opus-4-6-v1", ModelPrice(5.50, 27.50, 0.55, 6.875))
            .with_price("jp.anthropic.claude-opus-4-6-v1", ModelPrice(5.50, 27.50, 0.55, 6.875))
            .with_price("apac.anthropic.claude-opus-4-6-v1", ModelPrice(5.50, 27.50, 0.55, 6.875))
            #
            # Claude Opus 4.5: as Opus 5 (a 67 percent cut from Opus 4.1's 15 / 75).
            # `claude-opus-4-5` is the Claude API alias of the dated snapshot; the
            # book has no alias mechanism, so both are rows. Bedrock: global, us, eu.
            .with_price("claude-opus-4-5", ModelPrice(5.00, 25.00, 0.50, 6.25))
            .with_price("claude-opus-4-5-20251101", ModelPrice(5.00, 25.00, 0.50, 6.25))
            .with_price(
                "global.anthropic.claude-opus-4-5-20251101-v1:0",
                ModelPrice(5.00, 25.00, 0.50, 6.25),
            )
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("claude-opus-4-5@20251101", ModelPrice(5.50, 27.50, 0.55, 6.875))
            .with_price(
                "anthropic.claude-opus-4-5-20251101-v1:0", ModelPrice(5.50, 27.50, 0.55, 6.875)
            )
            .with_price(
                "us.anthropic.claude-opus-4-5-20251101-v1:0", ModelPrice(5.50, 27.50, 0.55, 6.875)
            )
            .with_price(
                "eu.anthropic.claude-opus-4-5-20251101-v1:0", ModelPrice(5.50, 27.50, 0.55, 6.875)
            )
            #
            # Claude Sonnet 5.5: 2 / 10, cache hits 0.20, writes 2.50 / 4.
            .with_price("claude-sonnet-5-5", ModelPrice(2.00, 10.00, 0.20, 2.50))
            .with_price("anthropic/claude-sonnet-5.5", ModelPrice(2.00, 10.00, 0.20, 2.50))
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("anthropic.claude-sonnet-5-5", ModelPrice(2.20, 11.00, 0.22, 2.75))
            #
            # Claude Sonnet 5: 2 / 10, as Sonnet 5.5. The page's footnote: the 2 / 10
            # launch price "is now the standard price" and the increase to 3 / 15
            # scheduled for 2026-09-01 "will not occur". tokenfuse#305 assumed 3 / 15.
            .with_price("claude-sonnet-5", ModelPrice(2.00, 10.00, 0.20, 2.50))
            .with_price("anthropic/claude-sonnet-5", ModelPrice(2.00, 10.00, 0.20, 2.50))
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("anthropic.claude-sonnet-5", ModelPrice(2.20, 11.00, 0.22, 2.75))
            #
            # Claude Sonnet 4.6: 3 / 15, cache hits 0.30, writes 3.75 / 6. Bedrock:
            # global, us, eu, jp.
            .with_price("claude-sonnet-4-6", ModelPrice(3.00, 15.00, 0.30, 3.75))
            .with_price("global.anthropic.claude-sonnet-4-6", ModelPrice(3.00, 15.00, 0.30, 3.75))
            .with_price("anthropic/claude-sonnet-4.6", ModelPrice(3.00, 15.00, 0.30, 3.75))
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("anthropic.claude-sonnet-4-6", ModelPrice(3.30, 16.50, 0.33, 4.125))
            .with_price("us.anthropic.claude-sonnet-4-6", ModelPrice(3.30, 16.50, 0.33, 4.125))
            .with_price("eu.anthropic.claude-sonnet-4-6", ModelPrice(3.30, 16.50, 0.33, 4.125))
            .with_price("jp.anthropic.claude-sonnet-4-6", ModelPrice(3.30, 16.50, 0.33, 4.125))
            #
            # Claude Sonnet 4.5 (deprecated): as Sonnet 4.6. Bedrock: global, us, eu, jp.
            .with_price("claude-sonnet-4-5", ModelPrice(3.00, 15.00, 0.30, 3.75))
            .with_price("claude-sonnet-4-5-20250929", ModelPrice(3.00, 15.00, 0.30, 3.75))
            .with_price(
                "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                ModelPrice(3.00, 15.00, 0.30, 3.75),
            )
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("claude-sonnet-4-5@20250929", ModelPrice(3.30, 16.50, 0.33, 4.125))
            .with_price(
                "anthropic.claude-sonnet-4-5-20250929-v1:0", ModelPrice(3.30, 16.50, 0.33, 4.125)
            )
            .with_price(
                "us.anthropic.claude-sonnet-4-5-20250929-v1:0", ModelPrice(3.30, 16.50, 0.33, 4.125)
            )
            .with_price(
                "eu.anthropic.claude-sonnet-4-5-20250929-v1:0", ModelPrice(3.30, 16.50, 0.33, 4.125)
            )
            .with_price(
                "jp.anthropic.claude-sonnet-4-5-20250929-v1:0", ModelPrice(3.30, 16.50, 0.33, 4.125)
            )
            #
            # Claude Haiku 4.5: 1 / 5, cache hits 0.10, writes 1.25 / 2. Bedrock
            # Messages-API id `anthropic.claude-haiku-4-5`; InvokeModel profiles:
            # global, us, eu.
            .with_price("claude-haiku-4-5", ModelPrice(1.00, 5.00, 0.10, 1.25))
            .with_price("claude-haiku-4-5-20251001", ModelPrice(1.00, 5.00, 0.10, 1.25))
            .with_price(
                "global.anthropic.claude-haiku-4-5-20251001-v1:0",
                ModelPrice(1.00, 5.00, 0.10, 1.25),
            )
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("claude-haiku-4-5@20251001", ModelPrice(1.10, 5.50, 0.11, 1.375))
            .with_price("anthropic.claude-haiku-4-5", ModelPrice(1.10, 5.50, 0.11, 1.375))
            .with_price(
                "anthropic.claude-haiku-4-5-20251001-v1:0", ModelPrice(1.10, 5.50, 0.11, 1.375)
            )
            .with_price(
                "us.anthropic.claude-haiku-4-5-20251001-v1:0", ModelPrice(1.10, 5.50, 0.11, 1.375)
            )
            .with_price(
                "eu.anthropic.claude-haiku-4-5-20251001-v1:0", ModelPrice(1.10, 5.50, 0.11, 1.375)
            )
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
