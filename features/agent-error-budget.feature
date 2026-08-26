Feature: When is an agent reliable enough

  The ask, 2026-08-26: SLOs for agents, so that an error budget is a number an
  operator can reason about rather than a feeling. `@yurii 2026-08-26`, on how
  far it goes: "лише вимірювання".

  Enterprises say quality, not cost, is what blocks agents from production, and
  the thing they cannot say is when one is reliable enough. This plane already
  scores answers and detects drift. Neither converts into an allowance somebody
  can be out of, and that conversion is what makes an objective arguable before
  it is breached rather than after.

  Nothing here is enforced. No agent is demoted, no policy is written, and no
  autonomy tier exists in the estate to write one into.

  Background:
    Given a tokenfuse trace of runs an operator's fleet produced
    And an objective for each indicator

  # @test:test_wilson_widens_on_a_small_sample
  Scenario: A ratio without its sample size is not evidence
    Given one agent with four runs and another with four hundred
    When both are measured
    Then the interval around the small sample is far wider
    And an operator can see which of the two numbers means anything

  # @test:test_the_error_budget_goes_negative_rather_than_clamping
  Scenario: Twice over budget does not read as exactly spent
    Given an agent failing twice as often as its objective allows
    When its budget is reported
    Then the figure is negative rather than zero
    And a budget exactly used up is a different number from one blown through

  # @test:test_escalation_is_a_failure_of_task_success_by_default
  Scenario: Handing the task to a person is not the agent succeeding
    Given a run that escalated to a human
    When task success is measured
    Then the run counts against the objective
    And an operator who disagrees can say so with one flag

  # @test:test_an_untagged_run_is_not_a_failed_run
  Scenario: A run nobody tagged is unmeasured, not failed
    Given runs that carry no outcome tag
    When task success is measured
    Then those runs are in neither the numerator nor the denominator
    And an unconfigured fleet reads as unmeasured rather than as broken

  # @test:test_containment_counts_a_wardryx_refusal_and_not_only_a_breaker_one
  Scenario: Something stopped the run, whichever plane stopped it
    Given a run the policy plane refused
    When containment is measured
    Then that run counts as uncontained
    And the answer does not depend on which of two vocabularies the refusal used

  # @test:test_cost_discipline_is_measured_against_the_fleet_not_the_agent
  Scenario: An agent is not graded against its own bad habit
    Given an agent that has been expensive since the day it shipped
    When cost discipline is measured
    Then it is compared with the rest of the fleet
    And it does not pass by being consistently expensive

  # @test:test_an_unattributed_run_is_counted_and_never_bucketed
  Scenario: A fleet does not score better for being unidentifiable
    Given runs that carry no identity at all
    When the report is produced
    Then those runs appear in no subject's numbers
    And the report says how many there were, beside the figures

  # @test:test_a_burn_rate_with_no_clock_is_unmeasured_and_not_zero
  Scenario: A warning that can never fire says so
    Given runs with no usable timestamp
    When the burn rate is computed
    Then the report says the rate could not be measured
    And it does not report the fleet as burning at zero

  # @test:test_a_breach_the_interval_does_not_establish_never_reaches_the_bus
  Scenario: Nobody is woken for a number that could be luck
    Given an agent below its objective on a sample too small to show it
    When events are prepared
    Then no event is sent
    And the report still shows the shortfall, so nobody has to guess why

  # @test:test_an_established_breach_does_reach_the_bus_with_its_evidence
  Scenario: A breach the evidence shows does reach a person
    Given an agent clearly below its objective over enough runs
    When events are prepared
    Then one event carries the measurement and the interval behind it
    And a consumer can see how many runs it rests on

  # @test:test_a_slow_burn_never_reaches_the_bus
  Scenario: A budget going by Friday is not the same alarm as one gone
    Given a fleet that was healthy and started failing this morning
    When the burn rate crosses the slower threshold
    Then the report shows it
    And no event is sent, because one type is one paging band

  # @test:test_quality_floor_says_why_it_could_not_be_measured
  Scenario: An absent indicator sends somebody to the right place
    Given no per-run scores were supplied
    When quality is measured
    Then the report says it was not measured and why
    And the reason names where the scores would have to come from

  # @test:test_an_eval_run_round_trips_its_subject
  Scenario: A run remembers whose run it was
    Given an eval run of an agent the operator named
    When the run is stored
    Then the store keeps the agent it was named with
    And a later report can tell whose scores those are

  # @test:test_a_store_written_before_the_subject_existed_still_opens
  Scenario: A store written last month still opens this month
    Given an eval store from a build before a run had a subject
    When this build opens it
    Then it opens, and every run and score is still there
    And nothing had to be exported and re-imported by hand

  # @test:test_a_row_written_before_the_column_reads_back_as_no_subject
  Scenario: A run from before nobody can attribute is left unattributed
    Given a run written before there was anywhere to record its subject
    When it is read back
    Then it reports no subject
    And no plausible-looking agent has been filled in for it

  # @test:test_quality_floor_computes_from_scores_supplied_beside_the_trace
  Scenario: The indicator that shipped dark now computes
    Given a trace of a fleet and the eval scores for the same agents
    When quality is measured
    Then the fraction of runs meeting the floor is reported
    And it rests on the same subject the rest of the report is grouped by

  # @test:test_a_score_with_no_subject_is_counted_and_never_bucketed
  Scenario: A fleet does not score better for an unlabelled evaluation
    Given some scores from runs nobody named an agent for
    When the report is produced
    Then those scores are in no subject's numbers
    And the report says how many there were

  # @test:test_a_subject_known_only_from_its_scores_still_appears
  Scenario: An agent evaluated offline is still in the report
    Given an agent with eval scores and no traffic on the trace
    When the report is produced
    Then it has a row of its own with its quality measured
    And the indicators that need a trace say there were no runs at all

  # @test:test_slo_command_refuses_to_join_an_eval_store_to_a_credential
  Scenario: A join that cannot be sound is refused rather than reported empty
    Given the report is grouped by the credential the gateway resolved
    When an eval store is offered as the source of scores
    Then the command refuses and says why
    And it does not quietly report quality as unmeasured instead

  # @test:test_a_key_keyed_subject_is_reported_and_never_emitted
  Scenario: The identity that is safe to gate on is not one the bus can carry
    Given the fleet is grouped by the credential rather than by the agent name
    When an objective is breached
    Then the report shows it against that credential
    And no event is sent, because the envelope's only subject is an agent id
    And nothing malformed is written in its place
