Feature: An event the operator asked for reaches the bus, or the command says it did not

  A regression that prints `regressed` and puts nothing on the bus is the
  quietest way for this plane to fail: the report looks complete and the
  consumer of the bus never hears. Found on 2026-11-04 by running `verdryx
  drift` against a real typed run with the event log on and no agent id typed.

  # @test:test_drift_command_regressed_event_carries_the_agent_the_run_was_stored_with
  Scenario: A regression reaches the bus under the agent that was evaluated
    Given an eval run stored under an agent id
    And a baseline above that run's mean
    When verdryx checks drift with the event log on and no agent id typed
    Then one quality_drift event is on the bus
    And it names the agent the run was evaluated as

  # @test:test_drift_command_event_names_the_newest_run_in_the_window_not_the_oldest
  Scenario: A drift event names the run that just ran, not the oldest in the window
    Given two eval runs of one model under two different agent ids
    And a drift window of both runs
    When verdryx checks drift with the event log on and no agent id typed
    Then the quality_drift event names the newer run and the newer agent

  # @test:test_drift_command_agent_id_flag_overrides_the_one_stored_with_the_run
  Scenario: An agent id typed on the drift command wins
    Given an eval run stored under an agent id
    When verdryx checks drift with a different agent id typed
    Then the quality_drift event names the typed one

  # @test:test_drift_command_regressed_with_no_agent_anywhere_says_the_event_was_not_written
  Scenario: A drift event nobody can attribute says it was not written
    Given an eval run stored with no agent id
    When verdryx checks drift with the event log on and no agent id typed
    Then the report says regressed
    And nothing is written to the event log
    And the command says on stderr that one event was not written and why

  # @test:test_eval_command_with_events_but_no_agent_id_says_nothing_was_written
  Scenario: An eval run with the event log on and no agent id says nothing was written
    Given an eval set of several cases
    When verdryx grades it with the event log on and no agent id
    Then nothing is written to the event log
    And the command says on stderr how many events were not written and why

  # @test:test_eval_command_with_events_and_an_agent_id_prints_no_event_warning
  Scenario: A run that did write its events does not warn
    Given an eval set of several cases
    When verdryx grades it with the event log on and an agent id
    Then the events are written
    And the command prints no warning about events

  # @test:test_a_burn_event_carries_the_count_behind_its_rate_so_an_unmeasured_rate_shows
  Scenario: A burn event says whether its rate was ever measured
    Given an agent whose error budget is exhausted and whose runs carry no timestamp
    When verdryx puts the slo_burn event on the bus
    Then the event carries a burn rate of zero and a count of zero runs behind it
    And the same event for runs that do carry timestamps carries a count above zero
