Feature: The judge bake-off is ready before the key exists

  @decided 2026-09-25: the bake-off between judges is prepared in full
  ahead of time, so that the day a Jev key exists the only steps left are
  putting the key in place and running one command. Nothing in it spends
  money without a separate yes given at run time.

  # @test:test_cmd_claude_estimate_cli_exits_1_without_confirmation
  # @test:test_require_claude_confirmation_never_opens_a_socket
  Scenario: The paid judge waits for a yes
    Given the bake-off is asked to include the priced LLM judge
    When no spend confirmation was given
    Then it prints what the run would cost and stops before any call

  # @test:test_arithmetic_false_cases_really_are_the_wrong_product
  # @test:test_routing_false_cases_answer_is_a_different_real_team
  # @test:test_format_false_cases_really_are_wrong
  Scenario: Every case knows its own truth
    Given the generated cases
    Then every answer marked wrong really is wrong and every answer marked right really is right

  # @test:test_grade_judge_typed_counts_unanswered_and_never_scores_it
  Scenario: A judge that cannot answer is counted, not scored
    Given a judge that leaves a case unanswered
    When the bake-off grades it
    Then the case is counted as unanswered with its reason and scores nothing

  # @test:test_report_states_n_beside_the_accuracy_ratio
  # @test:test_report_carries_no_verdict_word
  Scenario: The report gives numbers, not a verdict
    Given a finished bake-off
    Then every ratio in the report stands beside its n and no sentence declares a winner

  # @test:test_accuracy_at_empty_is_nan_not_zero
  # @test:test_cost_per_1000_answered_zero_answered_is_nan_not_zero
  Scenario: A number nobody measured is not a zero
    Given a judge that answered nothing in a family
    Then its accuracy and its cost per thousand read as not measured rather than as zero
