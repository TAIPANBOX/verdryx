Feature: A fix to a finding is judged by the runs that come after it

  A finding that someone says is fixed is a claim. The next runs decide it:
  measured and clean closes the finding, measured and still failing brings it
  back with the attempt on its record, and anything short of a measurement
  leaves the claim open. Runs from before the fix are the ones it was written
  against, so they neither prove it nor undo it.

  # @test:test_registering_a_fix_marks_the_finding_as_waiting_for_its_run
  Scenario: A registered fix puts the finding in waiting
    Given an open drift finding
    When a fix is registered for it with a note
    Then the finding is marked as having a fix pending
    And the listing of open findings still shows it

  # @test:test_a_second_fix_is_refused_while_one_is_pending
  Scenario: Only one fix is on trial at a time
    Given a finding with a fix pending
    When a second fix is registered
    Then it is refused and the pending one is named

  # @test:test_a_fix_for_an_unknown_finding_or_with_an_empty_note_is_refused
  Scenario: A fix needs a finding that exists and a note that says something
    Given a store with one finding
    When a fix is registered for an id that matches nothing, or with a blank note
    Then each is refused with the reason

  # @test:test_a_run_after_the_fix_that_is_on_track_verifies_it_and_closes_the_finding
  Scenario: A clean run after the fix closes the finding
    Given a finding with a fix pending
    And an eval run that started after the fix and scores like the baseline
    When verdryx checks drift
    Then the fix is verified and the finding is closed with a closing time

  # @test:test_a_run_after_the_fix_that_still_regresses_fails_it_and_reopens_the_finding_with_the_attempt
  Scenario: A regressed run after the fix fails it and the finding comes back
    Given a finding with a fix pending
    And an eval run that started after the fix and is still far below the baseline
    When verdryx checks drift
    Then the fix is failed and the finding is open again
    And the finding still shows the failed attempt with its note and its numbers

  # @test:test_a_drift_check_with_no_run_since_the_fix_leaves_it_pending
  Scenario: With no run since the fix, nothing is decided
    Given a finding with a fix pending and no eval run after it
    When verdryx checks drift
    Then the fix stays pending and the output says no run has started since

  # @test:test_runs_from_before_the_fix_do_not_verify_it
  Scenario: Old runs in a wide window do not count for or against the fix
    Given a regressed run from before the fix and a clean one after it
    When verdryx checks drift over a window of both
    Then the window's own verdict is still regressed
    And the fix is verified by the run after it alone

  # @test:test_a_closed_finding_is_not_reopened_by_the_runs_it_was_closed_over
  Scenario: Closing a finding is not undone by the runs it was closed over
    Given a finding closed by a verified fix
    When verdryx checks drift again over a window that still holds the old regressed run
    Then no new finding is stored

  # @test:test_a_regression_after_a_verified_fix_is_a_new_finding
  Scenario: The failure coming back after a verified fix is a new finding
    Given a finding closed by a verified fix
    When a regressed eval run starts after the closure and drift is checked
    Then a second, open finding is stored and the closed one stays closed

  # @test:test_a_run_of_another_version_does_not_test_a_fix_that_names_one
  Scenario: A run of the old build cannot fail a fix that ships in a new one
    Given a fix that names version v2 and a regressed run after it of version v1
    When verdryx checks drift
    Then the fix stays pending and says the runs since it are of another version
    And a clean run of version v2 afterwards verifies it

  # @test:test_an_slo_fix_with_too_few_runs_since_stays_pending_and_says_not_measured
  Scenario: Too few runs after an error-budget fix is not a pass
    Given a finding on an error budget with a fix pending
    And only a handful of runs after the fix
    When verdryx computes the budget
    Then the fix stays pending and says the indicator was not measured since

  # @test:test_an_slo_fix_is_verified_by_runs_after_it_that_meet_the_objective
  Scenario: Enough clean runs after an error-budget fix verify it
    Given a finding on an error budget with a fix pending
    And enough runs after the fix that all succeed
    When verdryx computes the budget over the old and the new runs together
    Then the fix is verified and the finding is closed
    And the old runs, which still breach the pooled window, open no new finding

  # @test:test_an_slo_fix_that_still_breaches_is_failed
  Scenario: Runs after an error-budget fix that still breach fail it
    Given a finding on an error budget with a fix pending
    And enough runs after the fix, half of which fail
    When verdryx computes the budget
    Then the fix is failed and the finding is open again with the attempt kept

  # @test:test_an_slo_fix_whose_interval_still_covers_the_target_stays_pending
  Scenario: Runs after an error-budget fix that are below target but not provably so decide nothing
    Given a finding on an error budget with a fix pending
    And enough runs after the fix to measure, slightly below the target, with an interval that still covers it
    When verdryx computes the budget
    Then the fix stays pending and the output calls the result inconclusive
