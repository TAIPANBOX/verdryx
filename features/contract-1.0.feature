Feature: The 1.0 event contract

  agent-passport reached SPEC 1.0 on 2026-09-12 and added a matching event
  schema: the same envelope verdryx already writes, with the version string
  changed and the agent id grammar widened to admit a claimed subject. SPEC
  6.4.1 requires a consumer to accept v0.1, v0.2 and v1.0 events from 1.0
  onward, and lets a producer keep emitting the version it emits today,
  moving to v1.0 in its own release.

  @decided 2026-09-12: verdryx keeps emitting v0.2 in this change and moves
  to v1.0 in a later, separate release. Nothing on the wire changes here.
  What these two scenarios prove instead: the line verdryx already writes
  fits the 1.0 contract once the version string is updated, and the only
  real difference between the two contracts is the widening SPEC 6.4.1
  allows.

  # @test:test_a_v1_0_stamped_event_this_emitter_writes_validates_under_the_vendored_v1_0_contract
  Scenario: An event verdryx writes today already fits the 1.0 contract
    Given an event verdryx writes today
    When its version string is set to v1.0
    Then it validates under the 1.0 contract byte for byte from agent-passport

  # @test:test_the_vendored_v1_0_contract_widens_only_the_subject
  Scenario: The only widening from 0.2 to 1.0 is the claimed subject
    Given the 1.0 contract
    When compared with v0.2
    Then the only widening is the claimed subject, which verdryx never writes
