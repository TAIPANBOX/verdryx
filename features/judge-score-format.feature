Feature: The LLM judge's score is read from one declared line, or not at all

  @decided 2026-11-23: the judge prompt declares one exact score format, the
  parser accepts only that format, and every ambiguous reply (no score line,
  more than one, a number out of range, something that is not a number) is a
  refusal. A refused reply is never a pass and never a number: the case is
  counted apart, the same way an unanswered typed case already is. Paraphrase
  of the 2026-11-23 review that found a reply such as "1 of 5 criteria met,
  0.2" read as a perfect score, which then reached drift and the error budget.

  # @test:test_judge_prompt_declares_one_exact_score_line
  Scenario: The judge is told the one form its answer must take
    Given the judge's instructions and the message it is sent
    Then they ask for exactly one line, "SCORE: <number>", as the whole reply
    And they say a reply in any other form is refused

  # @test:test_judge_reads_the_declared_score_line
  Scenario: A reply that is the declared line is read as its number
    Given a judge reply that is "SCORE: 0.8", with or without whitespace around it
    When the score is read
    Then the score is 0.8

  # @test:test_a_reply_that_leads_with_a_count_is_refused_not_read_as_its_first_number
  Scenario: A count in prose is not taken for the score
    Given a judge reply "1 of 5 criteria met, 0.2"
    When the score is read
    Then it is refused because no line is a score line
    And it is not read as 1.0

  # @test:test_an_out_of_range_score_is_refused_not_clamped
  Scenario: A number outside zero to one is refused, not clamped
    Given a judge reply "SCORE: 7"
    When the score is read
    Then it is refused as out of range
    And it is not turned into a perfect 1.0

  # @test:test_a_refused_judge_reply_is_counted_apart_and_never_scored
  Scenario: A refused reply is counted apart and never reaches the mean
    Given an eval set with two judged cases
    And the judge answers the first with "SCORE: 0.4" and the second in prose
    When the eval runs
    Then the run holds one score, 0.4, and its mean is 0.4
    And the second case is counted as unanswered with the reason "judge_no_score_line"

  # @test:test_an_adapter_value_outside_zero_one_is_refused
  Scenario: A judge adapter that hands back NaN or a value outside zero to one is refused
    Given a judge adapter that returns NaN, infinity, 1.5 or -0.5
    When the case is graded
    Then the grade is refused with a reason, and no score is made

  # @test:test_eval_prints_and_stores_a_refused_judge_case_with_its_reason
  Scenario: The eval command stores a refused case apart and prints its reason
    Given an eval set whose one judged case the judge cannot score
    When "verdryx eval" runs it
    Then the mean is printed as unmeasured, not as zero
    And the refused case is printed and stored with its reason and no score

  # @test:test_hostile_judge_replies_are_refused_with_a_reason
  Scenario: Hostile replies are each refused with their own reason
    Given judge replies that are empty, carry two score lines, quote a score inside prose before a different final line, use non-ASCII digits, NaN or an exponent, or run to hundreds of thousands of characters
    When each score is read
    Then each is refused with the reason that fits it
    And the refusal message never carries a huge reply whole

  # @test:test_a_seed_sweep_never_reads_a_score_the_declared_line_did_not_state
  Scenario: Across hundreds of generated replies only the declared line alone is read
    Given 400 seeds, each generating a declared line, the same line with other text beside it, and prose with no score line
    Then the declared line alone reads back as exactly its value
    And every other reply is refused

  # @test:test_the_t_statistic_never_decides_the_verdict
  Scenario: Welch's t is reported beside the drift verdict and never decides it
    Given a drift comparison whose t-statistic is forced to an extreme in either direction
    When the verdict is computed
    Then the verdict is the one the flat threshold and the bootstrap interval give
