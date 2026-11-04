Feature: A person's grade of a case outweighs the model's, and says who gave it

  A model grader can be wrong, and the person who looked at the case knows it.
  A human grade replaces the model's value wherever a mean, a drift verdict or
  an error budget is computed, while the model's own value stays on record
  beside it. Whoever merely uploads results has no way to write one, and every
  grade carries its author, so a number that moved can be traced to a person.

  # @test:test_a_reviewer_grade_is_stored_with_its_author
  Scenario: A reviewer's grade is kept with who gave it and when
    Given a stored eval run and a registered reviewer
    When the reviewer grades one of its cases with a note
    Then the grade is stored with the reviewer's name, the value, the note and a time

  # @test:test_a_human_grade_outweighs_the_model_in_the_run_mean
  Scenario: The run's mean follows the person, not the model
    Given a run whose model scored every case 1.0
    When a reviewer grades two of its four cases 0.0
    Then the run reads back with a mean of 0.5
    And each graded score shows the reviewer and the model's original value

  # @test:test_a_human_grade_outweighs_the_model_in_drift
  Scenario: Drift follows the person
    Given a baseline and a newer run the model scored as on track
    When reviewers grade half of the newer run's cases 0.0 and drift is checked
    Then drift reads regressed and a finding is stored
    And the output and the finding say how many of the cases were graded by a person

  # @test:test_a_human_grade_outweighs_the_model_in_the_quality_floor
  Scenario: The error budget follows the person
    Given stored scores the model gave all 1.0 under one agent
    When reviewers grade half of them 0.0 and the budget is computed with those scores
    Then the quality floor is breached for that agent
    And the finding's evidence names the cases a person graded down

  # @test:test_a_principal_who_only_uploads_cannot_grade
  Scenario: Someone who only uploads results cannot grade
    Given a principal registered with the uploader role
    When it tries to grade a case with its own valid key
    Then the grade is refused, saying only a reviewer can grade, and nothing is written

  # @test:test_a_grade_with_no_credential_or_a_wrong_one_is_refused
  Scenario: A grade needs a credential that matches
    Given a registered reviewer
    When a grade is attempted with no key, with a wrong key, or under a name nobody registered
    Then each attempt is refused and nothing is written

  # @test:test_a_reviewer_cannot_sign_with_another_reviewers_name
  Scenario: A reviewer cannot grade under someone else's name
    Given two registered reviewers
    When one grades with their own key but the other's name
    Then the grade is refused and nothing is written

  # @test:test_a_later_human_grade_supersedes_the_earlier_one_and_both_stay_on_record
  Scenario: A second opinion replaces the first and the first is not erased
    Given a case already graded by one reviewer
    When another reviewer grades the same case differently
    Then the run reads back with the later value
    And both grades remain on record with their authors, oldest first

  # @test:test_a_grade_that_cannot_be_given_is_refused
  Scenario: A grade for something that does not exist, or is not a score, is refused
    Given a stored run with one answered case and one case nobody answered
    When a reviewer grades a case the run never had, the unanswered case, or gives a value outside 0 to 1
    Then each is refused and nothing is written

  # @test:test_principal_add_writes_the_key_once_and_the_store_keeps_only_its_hash
  Scenario: A registered principal's key is shown once and never stored
    Given an empty store
    When a principal is registered with a path for its key
    Then the key file exists, only its owner can read it, and the store holds a hash and not the key
    And registering the same name again, or writing to the same key path again, is refused

  # @test:test_uploading_results_needs_no_credential_and_writes_no_human_grade
  Scenario: Uploading results is unchanged and cannot carry a grade
    Given an eval set
    When verdryx grades it with the stub model and no credential at all
    Then the run is stored and no human grade exists

  # @test:test_drift_says_when_the_baseline_run_was_regraded_after_the_baseline_was_set
  Scenario: Drift says when the baseline's own run was regraded
    Given a baseline set from a run a reviewer then regrades
    When verdryx checks drift
    Then a warning says the snapshot and the run now disagree and that a new baseline is needed
