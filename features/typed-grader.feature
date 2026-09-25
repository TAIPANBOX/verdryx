Feature: A typed verdict from typryx, and the human label back to it

  @decided 2026-09-25: verdryx is the first consumer of typryx. A grader
  asks typryx a typed question about each output instead of asking a
  priced judge for a number in prose, it is used only when the caller
  names a typryx address, and where an eval case already carries a human
  label that label goes back to typryx so its calibration can be scored.

  @decided 2026-09-25: an unanswered typed case is counted apart with its
  reason, gets no score and is never a zero, and the run is saved with
  that count shown beside the mean. An eval run used to fail whole on one
  unanswered case (measured through a local backend: 3 of 60 asks come
  back unanswered every time, so a 60-case run could never complete, and
  labels already posted for earlier cases stayed in typryx's ledger while
  the run itself was thrown away); a refusal or an unreachable typryx is a
  different fact -- an infrastructure failure, not a verdict -- and still
  fails the run.

  # @test:test_build_graders_no_typed_client_means_no_typed_grader
  Scenario: Nobody gets the typed grader without asking for it
    Given verdryx builds its graders with no typryx address
    Then there is no typed grader among them

  # @test:test_typed_grader_noul_value_is_the_probability_typryx_gave
  Scenario: A typed verdict is the probability typryx gave
    Given a typed eval case and a typryx that answers yes with 0.8
    When verdryx grades the case
    Then the score is 0.8

  # @test:test_typed_grader_sends_only_task_and_final_answer
  Scenario: Only the task and the answer leave verdryx
    Given a typed eval case
    When verdryx asks typryx about it
    Then the state it sends holds the task and the final answer and nothing else

  # @test:test_eval_command_typed_unanswered_case_is_counted_not_fatal_and_run_saved
  Scenario: An unanswered verdict is not a zero
    Given typryx answers unanswered for one case of a run
    When verdryx grades the run
    Then that case has no score and is counted as unanswered with its reason
    And the run is saved and its mean is taken over the answered cases only

  # @test:test_eval_command_typed_all_unanswered_run_has_no_mean_and_cannot_baseline
  Scenario: A run nobody could answer has no mean
    Given typryx answers unanswered for every case of a run
    When verdryx grades the run
    Then the run reports its mean as unmeasured rather than as zero
    And it cannot become a baseline

  # @test:test_eval_command_typed_refusal_still_fails_the_run_and_saves_nothing
  Scenario: A refusal is not an unanswered verdict
    Given typryx refuses a request
    When verdryx grades the run
    Then the run fails naming the status and code, and nothing is saved

  # @test:test_typed_grader_posts_noul_human_label_as_outcome
  Scenario: A human label reaches typryx calibration
    Given a typed eval case whose expected value is true
    When verdryx grades the case
    Then typryx receives that label as the truth for the same answer

  # @test:test_typed_grader_label_that_does_not_fit_noul_is_never_posted
  Scenario: A label that does not fit the question is never sent
    Given a typed eval case whose expected value is not a yes or a no
    When verdryx grades the case against a yes-or-no template
    Then nothing is posted to typryx as an outcome and the case fails

  # @test:test_typryx_error_names_status_and_code_not_the_key
  Scenario: The key never appears on a command line or in an error
    Given verdryx is given a typryx key file
    When typryx refuses a request
    Then the error names the status and typryx's code and not the key
