Feature: A graded output cannot write into the judge's instructions

  @decided 2026-10-17: the LLM judge reads a candidate output as inert data.
  Whatever the output contains, it cannot close the block it sits in and
  address the judge from outside that block. Paraphrase of the 2026-10-17
  security review, finding F-027.

  # @test:test_judge_message_output_cannot_close_its_own_block
  Scenario: A candidate output that closes its own block stays inside it
    Given a candidate output that writes a closing tag and a new instruction after it
    When the judge message is built for that output
    Then the instruction sits inside the output block and nothing after the block except the score request

  # @test:test_judge_message_fence_is_fresh_on_every_call
  Scenario: A closing tag learned from one grading does not work on the next
    Given two judge messages built for the same candidate output
    Then each message closes its output block with a tag of its own and the two tags differ

  # @test:test_judge_message_fence_carries_at_least_128_bits
  Scenario: The closing tag is too long to guess
    Given any judge message
    Then the random token in its output tag is at least 128 bits long
