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


@dataclass(frozen=True)
class LongContext:
    """A model's long-prompt rate: every part of a call whose prompt is
    longer than ``above_prompt_tokens`` is priced at ``price``.

    Mirrors tokenfuse's ``tokenfuse_core::pricing::LongContext`` (its
    invariant 99). The prompt is every input token, cache reads and cache
    writes included, which is how Anthropic counts Haiku 5.5's 100,000-token
    tier and the gross prompt OpenAI's 272K and Google's 200K tiers are
    measured on. A prompt exactly at the threshold is still the base rate.
    """

    above_prompt_tokens: int
    price: ModelPrice


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
        self._long: dict[str, LongContext] = {}
        self._fallback: ModelPrice | None = None

    def with_price(self, model: str, price: ModelPrice) -> PriceBook:
        """Register an exact-match entry. Returns self, for chaining."""
        self._prices[model] = price
        return self

    def with_long_context(
        self, model: str, above_prompt_tokens: int, price: ModelPrice
    ) -> PriceBook:
        """Give an exact entry a long-prompt rate (see `LongContext`).
        Returns self, for chaining. A model with no exact entry never has
        one: the fallback is a single rate."""
        self._long[model] = LongContext(above_prompt_tokens, price)
        return self

    def long_context(self, model: str) -> LongContext | None:
        """`model`'s long-prompt rate, or None when it has none."""
        return self._long.get(model)

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

        A model with a long-prompt rate (`with_long_context`) is priced
        whole at that rate when its prompt, input plus cache read plus
        cache write tokens, is above the threshold, as tokenfuse's
        ``PriceBook::cost`` does; `lookup` always answers the base rate.

        Raises:
            ValueError: If `model` has no exact entry and no fallback is
                configured. `PriceBook.default()` always sets a fallback,
                so this only fires for a caller-built book that deliberately
                left one out.
        """
        entry = self.lookup(model)
        if entry is None:
            raise ValueError(f"no price for {model!r} and PriceBook has no fallback configured")
        long = self._long.get(model) if model in self._prices else None
        prompt = input_tokens + cache_read_tokens + cache_write_tokens
        if long is not None and prompt > long.above_prompt_tokens:
            entry = long.price
        return entry.cost_usd(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)

    @classmethod
    def default(cls) -> PriceBook:
        """The price book Verdryx's built-in adapters use unless a caller
        injects their own.

        Ported number-for-number from tokenfuse's published price book,
        `contracts/tokenfuse-constants.json` (`price_book.models` and
        `price_book.long_context`), at tokenfuse commit f75c912 (main,
        2026-10-14), which is generated from `crates/gateway/src/pricebook.rs`
        `default_price_book()`. Every rate as tokenfuse read it from the
        vendor's own page on 2026-10-13 (Anthropic, OpenAI, Google's Vertex
        AI, DeepSeek, Moonshot, Z.ai), the GPT-4.1 family as read on
        2026-10-14, Gemini 2.5 Flash and Flash-Lite and the Gemini 3.x Flash
        rows as read on 2026-10-08. Verify against
        https://platform.claude.com/docs/en/about-claude/pricing,
        https://cloud.google.com/vertex-ai/generative-ai/pricing,
        https://developers.openai.com/api/docs/pricing,
        https://api-docs.deepseek.com/quick_start/pricing,
        https://platform.kimi.ai/docs/pricing/chat and
        https://docs.z.ai/guides/overview/pricing before relying on them for
        anything beyond a rough estimate, same disclaimer tokenfuse's own
        price book carries.

        Long-prompt tiers (tokenfuse's invariant 99) are mirrored with
        `with_long_context`: a call whose prompt (input + cache read + cache
        write tokens) is above the threshold is priced whole at the long
        rate. estate-gates' C3 reads only the `.with_price(...)` rows; the
        tiers are held by this repository's own pinned test.

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
            # Claude Sonnet 5.5: 2 / 10, cache hits 0.10 (0.05x), writes 2.50 / 4.
            .with_price("claude-sonnet-5-5", ModelPrice(2.00, 10.00, 0.10, 2.50))
            .with_price("anthropic/claude-sonnet-5.5", ModelPrice(2.00, 10.00, 0.10, 2.50))
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("anthropic.claude-sonnet-5-5", ModelPrice(2.20, 11.00, 0.11, 2.75))
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
            # Claude Haiku 5.5, priced by prompt length: up to 100,000 prompt
            # tokens 0.10 / 0.50, cache hits 0.01, writes 0.125 / 0.20; over
            # 100,000, 0.50 / 2.50, 0.05, 0.625 / 1. Bedrock
            # `anthropic.claude-haiku-5-5` at list plus 10 percent on both tiers.
            .with_price("claude-haiku-5-5", ModelPrice(0.10, 0.50, 0.01, 0.125))
            .with_long_context("claude-haiku-5-5", 100_000, ModelPrice(0.50, 2.50, 0.05, 0.625))
            .with_price("anthropic/claude-haiku-5.5", ModelPrice(0.10, 0.50, 0.01, 0.125))
            .with_long_context(
                "anthropic/claude-haiku-5.5", 100_000, ModelPrice(0.50, 2.50, 0.05, 0.625)
            )
            # Regional or endpoint-unnamed ids: list plus 10 percent.
            .with_price("anthropic.claude-haiku-5-5", ModelPrice(0.11, 0.55, 0.011, 0.1375))
            .with_long_context(
                "anthropic.claude-haiku-5-5", 100_000, ModelPrice(0.55, 2.75, 0.055, 0.6875)
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
            # Google: Gemini on Vertex AI, by the id Vertex AI's
            # OpenAI-compatible endpoint takes (`google/<model id>`), as
            # tokenfuse prices it (its `GEMINI` table, invariant 82). Rates
            # read by tokenfuse from
            # https://cloud.google.com/vertex-ai/generative-ai/pricing on
            # 2026-10-08. No cache-write fee, so cache_write is the input
            # rate. Each choice the page leaves open is taken on the
            # over-charging side: the Gemini 3 family at its non-global
            # rate (the id does not name the endpoint), Gemini 3.6, 3.7 and
            # 3.8 Flash at the standard rate, not the introductory credit.
            # Gemini 2.5 Pro and 3.1 Pro Preview carry their rate above 200K
            # prompt tokens as a long-prompt tier (read 2026-10-13); 3.1 Pro
            # Preview lists a Global row only.
            .with_price("google/gemini-3.8-flash", ModelPrice(1.65, 8.25, 0.165, 1.65))
            .with_price("google/gemini-3.8-flash-cyber", ModelPrice(1.65, 8.25, 0.165, 1.65))
            .with_price("google/gemini-3.7-flash", ModelPrice(1.65, 8.25, 0.165, 1.65))
            .with_price("google/gemini-3.6-flash", ModelPrice(1.65, 8.25, 0.165, 1.65))
            .with_price("google/gemini-3.5-flash", ModelPrice(1.65, 9.90, 0.165, 1.65))
            .with_price("google/gemini-3.5-flash-lite", ModelPrice(0.33, 2.75, 0.033, 0.33))
            .with_price("google/gemini-3.1-flash-lite", ModelPrice(0.275, 1.65, 0.0275, 0.275))
            .with_price("google/gemini-3.1-pro-preview", ModelPrice(2.00, 12.00, 0.20, 2.00))
            .with_long_context(
                "google/gemini-3.1-pro-preview", 200_000, ModelPrice(4.00, 18.00, 0.40, 4.00)
            )
            .with_price("google/gemini-2.5-pro", ModelPrice(1.25, 10.00, 0.125, 1.25))
            .with_long_context(
                "google/gemini-2.5-pro", 200_000, ModelPrice(2.50, 15.00, 0.25, 2.50)
            )
            .with_price("google/gemini-2.5-flash", ModelPrice(0.30, 2.50, 0.03, 0.30))
            .with_price("google/gemini-2.5-flash-lite", ModelPrice(0.10, 0.40, 0.01, 0.10))
            #
            # OpenAI, Standard tier, read by tokenfuse from
            # https://developers.openai.com/api/docs/pricing on 2026-10-13.
            # GPT-5.6 and later bill a cache write at 1.25x input; where the
            # page shows no cache-write price there is no write fee, so
            # cache_write is the input rate. The tiered models carry their
            # rate above 272K prompt tokens as a long-prompt tier.
            .with_price("gpt-6-astra", ModelPrice(10.00, 50.00, 1.00, 12.50))
            .with_long_context("gpt-6-astra", 272_000, ModelPrice(20.00, 75.00, 2.00, 25.00))
            .with_price("gpt-6.1-sol", ModelPrice(2.00, 10.00, 0.10, 2.50))
            .with_long_context("gpt-6.1-sol", 272_000, ModelPrice(4.00, 15.00, 0.20, 5.00))
            .with_price("gpt-6-sol", ModelPrice(2.00, 10.00, 0.20, 2.50))
            .with_long_context("gpt-6-sol", 272_000, ModelPrice(4.00, 15.00, 0.40, 5.00))
            .with_price("gpt-6-luna", ModelPrice(0.10, 0.50, 0.01, 0.125))
            .with_long_context("gpt-6-luna", 272_000, ModelPrice(0.20, 0.75, 0.02, 0.25))
            .with_price("gpt-5.6-sol", ModelPrice(4.00, 20.00, 0.40, 5.00))
            .with_long_context("gpt-5.6-sol", 272_000, ModelPrice(8.00, 30.00, 0.80, 10.00))
            .with_price("gpt-5.6-terra", ModelPrice(2.00, 12.00, 0.20, 2.50))
            .with_long_context("gpt-5.6-terra", 272_000, ModelPrice(4.00, 18.00, 0.40, 5.00))
            .with_price("gpt-5.6-luna", ModelPrice(0.20, 1.20, 0.02, 0.25))
            .with_long_context("gpt-5.6-luna", 272_000, ModelPrice(0.40, 1.80, 0.04, 0.50))
            .with_price("gpt-5.5", ModelPrice(5.00, 30.00, 0.50, 5.00))
            .with_long_context("gpt-5.5", 272_000, ModelPrice(10.00, 45.00, 1.00, 10.00))
            .with_price("gpt-5.4", ModelPrice(2.50, 15.00, 0.25, 2.50))
            .with_long_context("gpt-5.4", 272_000, ModelPrice(5.00, 22.50, 0.50, 5.00))
            .with_price("gpt-5.4-mini", ModelPrice(0.75, 4.50, 0.075, 0.75))
            .with_price("gpt-4o", ModelPrice(2.50, 10.00, 1.25, 2.50))
            .with_price("gpt-4o-mini", ModelPrice(0.15, 0.60, 0.075, 0.15))
            # The GPT-4.1 family, read by tokenfuse on 2026-10-14 from the same
            # page: 2.00 / 8.00, cached 0.50; mini 0.40 / 1.60, cached 0.10;
            # nano 0.10 / 0.40, cached 0.025. Cache writes show "-", so the
            # write rate is the input rate. No long-prompt tier is listed.
            .with_price("gpt-4.1", ModelPrice(2.00, 8.00, 0.50, 2.00))
            .with_price("gpt-4.1-mini", ModelPrice(0.40, 1.60, 0.10, 0.40))
            .with_price("gpt-4.1-nano", ModelPrice(0.10, 0.40, 0.025, 0.10))
            .with_price("o1", ModelPrice(15.00, 60.00, 7.50, 15.00))
            #
            # OpenRouter's `openai/<id>` names for every OpenAI id above, at the same
            # rates and with the same long-prompt tier. An `openai/` id names OpenAI's
            # own model sold at OpenAI's list rate, so the vendor's row is the honest
            # price (tokenfuse's invariant 107). An `openai/` id with no row above
            # stays at the fallback.
            .with_price("openai/gpt-4.1", ModelPrice(2.00, 8.00, 0.50, 2.00))
            .with_price("openai/gpt-4.1-mini", ModelPrice(0.40, 1.60, 0.10, 0.40))
            .with_price("openai/gpt-4.1-nano", ModelPrice(0.10, 0.40, 0.025, 0.10))
            .with_price("openai/gpt-4o", ModelPrice(2.50, 10.00, 1.25, 2.50))
            .with_price("openai/gpt-4o-mini", ModelPrice(0.15, 0.60, 0.075, 0.15))
            .with_price("openai/gpt-5.4", ModelPrice(2.50, 15.00, 0.25, 2.50))
            .with_long_context("openai/gpt-5.4", 272_000, ModelPrice(5.00, 22.50, 0.50, 5.00))
            .with_price("openai/gpt-5.4-mini", ModelPrice(0.75, 4.50, 0.075, 0.75))
            .with_price("openai/gpt-5.5", ModelPrice(5.00, 30.00, 0.50, 5.00))
            .with_long_context("openai/gpt-5.5", 272_000, ModelPrice(10.00, 45.00, 1.00, 10.00))
            .with_price("openai/gpt-5.6-luna", ModelPrice(0.20, 1.20, 0.02, 0.25))
            .with_long_context("openai/gpt-5.6-luna", 272_000, ModelPrice(0.40, 1.80, 0.04, 0.50))
            .with_price("openai/gpt-5.6-sol", ModelPrice(4.00, 20.00, 0.40, 5.00))
            .with_long_context("openai/gpt-5.6-sol", 272_000, ModelPrice(8.00, 30.00, 0.80, 10.00))
            .with_price("openai/gpt-5.6-terra", ModelPrice(2.00, 12.00, 0.20, 2.50))
            .with_long_context("openai/gpt-5.6-terra", 272_000, ModelPrice(4.00, 18.00, 0.40, 5.00))
            .with_price("openai/gpt-6-astra", ModelPrice(10.00, 50.00, 1.00, 12.50))
            .with_long_context("openai/gpt-6-astra", 272_000, ModelPrice(20.00, 75.00, 2.00, 25.00))
            .with_price("openai/gpt-6-luna", ModelPrice(0.10, 0.50, 0.01, 0.125))
            .with_long_context("openai/gpt-6-luna", 272_000, ModelPrice(0.20, 0.75, 0.02, 0.25))
            .with_price("openai/gpt-6-sol", ModelPrice(2.00, 10.00, 0.20, 2.50))
            .with_long_context("openai/gpt-6-sol", 272_000, ModelPrice(4.00, 15.00, 0.40, 5.00))
            .with_price("openai/gpt-6.1-sol", ModelPrice(2.00, 10.00, 0.10, 2.50))
            .with_long_context("openai/gpt-6.1-sol", 272_000, ModelPrice(4.00, 15.00, 0.20, 5.00))
            .with_price("openai/o1", ModelPrice(15.00, 60.00, 7.50, 15.00))
            #
            # First-party APIs at their default endpoint, read by tokenfuse
            # on 2026-10-13. DeepSeek at its PEAK rate (off-peak is half;
            # the book does not price by the clock), no write fee; its two
            # legacy names are billed at the Flash price. Moonshot's Kimi K3
            # (writes 3.00 for 5 minutes, 6.00 for an hour; the 1-hour rate
            # is not mirrored here). Z.ai's GLM-5.3 and GLM-5.1, no write fee.
            .with_price("deepseek-flash", ModelPrice(0.30, 1.20, 0.006, 0.30))
            .with_price("deepseek-v4-flash", ModelPrice(0.30, 1.20, 0.006, 0.30))
            .with_price("deepseek-v4-flash-vision-exp", ModelPrice(0.30, 1.20, 0.006, 0.30))
            .with_price("deepseek-v4-pro", ModelPrice(1.32, 3.96, 0.044, 1.32))
            .with_price("kimi-k3", ModelPrice(3.00, 15.00, 0.30, 3.00))
            .with_price("glm-5.3", ModelPrice(1.40, 4.40, 0.26, 1.40))
            .with_price("glm-5.1", ModelPrice(1.40, 4.40, 0.26, 1.40))
            #
            # Conservative fallback for anything not listed above (mirrors
            # tokenfuse's ADR-8): priced at 3x Opus 4.5, the most expensive
            # known model, so an unrecognized model is never under-priced
            # relative to what is actually known.
            .with_fallback(ModelPrice(15.0, 75.0, 1.5, 18.75))
        )
