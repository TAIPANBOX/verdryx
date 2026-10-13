Feature: The price book verdryx prices a judge with is tokenfuse's, tiers included

  @decided 2026-10-13: the built-in price tables are brought to the
  providers' current official prices in the same wave across the estate,
  and verdryx's copy of tokenfuse's price book follows tokenfuse's. That
  wave also made tokenfuse price a vendor's prompt-length tier for real
  (its invariant 99): a call whose prompt is above a model's threshold is
  priced whole at the long rate. A mirror that carried only the base rows
  would price every long judge call at the short rate.

  # @test:test_price_book_default_entry_matches_tokenfuse_pricebook
  Scenario: Every row is tokenfuse's published row
    Given tokenfuse's published price book at the commit this mirror names
    When verdryx looks up any model it lists
    Then it holds the same four rates

  # @test:test_price_book_default_long_context_matches_tokenfuse_pricebook
  Scenario: Every long-prompt tier is tokenfuse's published tier
    Given tokenfuse's published long-prompt tiers at the same commit
    When verdryx looks up a model's tier
    Then it holds the same threshold and the same four rates

  # @test:test_price_book_default_haiku_5_5_is_priced_by_prompt_length
  Scenario: A short and a long Claude Haiku 5.5 call are priced at their own rates
    Given a judge call on Claude Haiku 5.5 with 1000 input and 500 output tokens
    Then it costs 0.00035 USD, where it used to be priced at the fallback
    And a call with 150000 input and 1000 output tokens costs 0.0775 USD at the long rate

  # @test:test_cache_reads_and_writes_count_toward_the_prompt
  Scenario: Cache reads and writes count toward the prompt's length
    Given a tiered model with a threshold of 100000 prompt tokens
    When a call sends 40000 input, 40000 cache read and 20001 cache write tokens
    Then the whole call is priced at the long rate
