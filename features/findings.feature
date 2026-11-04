Feature: A quality failure is written down as a finding, without a model

  Drift and the error budget used to print a verdict and forget it. A finding
  is the record an operator comes back to: which agent, over what window, what
  kind of failure, which runs show it, and which model or version is the
  likeliest cause. Every part of it is computed from the stores and traces
  already on disk, so a finding is as useful with no model reachable as with
  one.

  # @test:test_a_regressed_drift_check_records_a_finding_with_agent_window_kind_and_evidence
  Scenario: A drift regression becomes a finding with everything an operator needs
    Given a baseline and a newer eval run of the same agent that scores far below it
    When verdryx checks drift against that baseline
    Then one open finding is stored for that agent
    And it names the kind drift, the window, and the run ids that show the drop
    And it names the lowest scoring cases of those runs

  # @test:test_an_on_track_drift_check_records_no_finding
  Scenario: A drift check that is on track writes nothing
    Given a baseline and a newer eval run that scores the same
    When verdryx checks drift against that baseline
    Then no finding is stored

  # @test:test_the_same_regression_seen_twice_is_one_finding_seen_twice
  Scenario: The same regression found again is the same finding
    Given a drift regression that is already an open finding
    When verdryx checks drift against the same baseline again
    Then there is still one finding
    And it counts two sightings and shows the newer run among its evidence

  # @test:test_the_finding_names_the_version_that_changed_since_the_baseline
  Scenario: The finding points at the version that changed
    Given a baseline run of version v1 and a regressed run of version v2 of the same agent
    When verdryx checks drift against that baseline
    Then the finding suspects version v2 and names the run it first appeared in

  # @test:test_a_finding_with_no_change_in_model_or_version_says_none_changed
  Scenario: With nothing changed the finding says so
    Given a baseline run and a regressed run that carry the same model and version
    When verdryx checks drift against that baseline
    Then the finding suspects no model or version
    And it says that model and version did not change across the runs examined

  # @test:test_find_suspect_over_a_single_run_says_there_is_nothing_to_compare
  Scenario: One run cannot show a change
    Given a single eval run
    When the suspect is looked for
    Then the answer says there is nothing to compare, which is not the same as no change

  # @test:test_a_regression_nobody_can_attribute_is_still_a_finding_without_an_agent
  Scenario: A regression with no agent id is kept, and says it has none
    Given a regressed run stored with no agent id
    When verdryx checks drift against its baseline
    Then a finding is stored with no agent
    And the listing shows that no agent id was recorded

  # @test:test_a_finding_is_recorded_with_no_model_and_no_network
  Scenario: Recording a finding needs no model and no network
    Given every network connection and every model adapter refuses to run
    When verdryx checks drift against a baseline the run has regressed from
    Then the finding is stored all the same

  # @test:test_an_established_slo_breach_records_a_finding_with_the_failing_runs
  Scenario: A breached error budget becomes a finding
    Given a trace in which half of one agent's runs escalate
    When verdryx computes the budget with a store given
    Then an open finding of kind slo_exhausted is stored for the indicator task_success
    And its evidence is the ids of the runs that failed

  # @test:test_a_slo_breach_the_interval_does_not_establish_records_no_finding
  Scenario: A breach the interval does not show is not a finding
    Given a trace whose point estimate is below the target but whose interval still covers it
    When verdryx computes the budget with a store given
    Then no finding is stored

  # @test:test_an_slo_breach_with_no_store_says_it_was_not_recorded
  Scenario: A breach with nowhere to record it says so
    Given a trace with an established breach
    When verdryx computes the budget with no store given
    Then the report says the breach was not recorded as a finding and how to record it

  # @test:test_an_slo_breach_of_a_key_is_reported_and_never_recorded_as_a_finding
  Scenario: A breach keyed on a credential is not an agent's finding
    Given a trace with an established breach grouped on key id
    When verdryx computes the budget with a store given
    Then no finding is stored
    And the report says the subject is a key and not an agent

  # @test:test_findings_list_and_show_print_what_was_recorded
  Scenario: The findings can be listed and read
    Given a stored finding
    When the operator lists findings and then shows that one by a prefix of its id
    Then the list names the finding with its status, agent and kind
    And the detail prints the window, the evidence, the suspect and the numbers
