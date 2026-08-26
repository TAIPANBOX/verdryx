# Error budgets for agent fleets

Somebody asks whether the support agent can run overnight without a person
watching it. You have a quality score of 0.91, a drift report saying nothing
has moved, and a spend figure. None of those answers the question, and after a
while you notice that none of them can.

A score is a number with no denominator. Drift is a change with no threshold.
Both describe the agent; neither describes how much room you have left before
you should stop trusting it. That room is what an error budget is, and this
document is how we built one.

The implementation is `verdryx slo` in this repository. Everything below was
either measured against it or is a decision we made and can defend.

## A mean cannot become a budget

The first thing we tried was the obvious thing: take the mean of the quality
scores, set a floor, alert when it drops. It doesn't work, and the two reasons
took us a while to pull apart.

The small one is that a mean has no allowance in it. Ask "how much of our
tolerance for bad answers have we used" and there's nothing to answer with.
Nothing in a mean is denominated in bad answers.

The bigger one is that a mean hides the shape. An agent that scores 0.9 on
every single run and an agent that alternates between 1.0 and 0.8 have the
same mean and they are not equally trustworthy. The second one produces a
visibly wrong answer half the time. If what you care about is how often the
thing was unacceptable, count that. Which means picking a line and measuring
the ratio of runs on each side of it.

So every indicator here is a ratio: good runs over eligible runs. Not an
average of anything.

## The unit is a run

A task an agent was given is one success or one failure, whatever it cost in
model calls. Counting calls instead would let a chatty agent dilute its own
failures: fifteen calls where two went wrong looks like 87% if you count
calls, and looks like one failed task if you count tasks, and the person who
asked for the task agrees with the second number.

This also happens to be free for us. The gateway's trace already reduces to
one record per run under the last outcome tag the run carried, and the
reduction is the same one the cost report uses. Two figures about the same run
now come from one scan of it, which matters more than it sounds: we have been
bitten before by two readers of the same data disagreeing about what they
read.

## Four indicators

We landed on four. Each one is a question an operator actually asks.

`task_success`: did the run end with an outcome tag we consider good.

`quality_floor`: was the run's score at or above a threshold. Note that this
is a floor and a count, not a mean, for the reason above.

This one shipped dark and was measurable a week later, and the reason is
worth reading because it was not a missing feature. The trace has no score in
it: TokenFuse records what a call cost and how it was decided, never how good
the answer was. The scores were in our own eval store, and the two had no run
identity in common, so nothing could be joined. The store's `eval_runs` table
carried no subject either, which meant a score could not even be attributed to
an agent, and a whole indicator was reporting itself unmeasured because of a
missing column rather than because of missing data.

The fix was one column and one flag. `verdryx eval --agent-id` had existed
since the event log did, and was stamping the subject onto every event of a
run while dropping it on the way to the store. It now goes onto the run, and
`verdryx slo --scores-db` joins on it. The unit is one CASE and not one eval
run: a case is one thing a person asked the agent to do, and thresholding a
run's mean would hide exactly the distribution a floor exists to price.

The join is the agent and cannot be anything else. Group the report by
`key_id`, which is the sound key for anything enforced, and there is nothing
to join on, because the eval store records an agent and no credential. The
command refuses rather than reporting quality as unmeasured, since that would
look identical to passing no scores at all.

`containment`: did anything have to stop this run. Any refusal counts, whether
it came from the budget breaker or from the policy plane.

`cost_discipline`: did the run cost at most N times the fleet's median run.

The one that gets argued about is that an escalation to a human counts as a
failure of `task_success` by default. An escalation is often the right thing
for the customer. It is still a failure of the objective "this agent handles
this without us", and that objective is what the budget is denominated in. If
you disagree for your fleet, the flag is `--good-outcomes` and you can put
`escalated` in it. We think most people should not.

`cost_discipline` is measured against the fleet and never against the agent's
own history. An agent that has been expensive since the day it shipped would
otherwise be graded against its own habit and pass forever.

## The budget goes negative

Set the objective at 0.95 and you are saying 5% of runs may be bad. When 10%
have been, the budget left is not zero. It is minus one: you have spent twice
the allowance.

We don't clamp it. Clamped at zero, the agent that crossed the line this
morning and the agent that is twenty times over both read as "exhausted", and
the operator gets one number for two situations that want completely different
responses. The sign is the only place in the figure carrying how far past the
line you already are.

## The burn rate, and the bug we shipped for an afternoon

The budget tells you where you stand. It does not tell you how fast you are
moving, and how fast you are moving is the part there is still time to act on.

We wrote the burn rate first as: how many times faster than sustainable are
the failures arriving. Then we computed it over the same set of runs as the
budget.

That's wrong, and it's wrong in a way that leaves no mark. If both numbers
come off the same window, the rate is exactly `1 - remaining`. Which means a
rate at or above 1 implies a budget at or below zero, exhaustion wins every
comparison, and the two burn triggers become unreachable code. The tool would
have run for a year without either of them firing once and nobody would have
noticed, because a warning that can never fire looks exactly like a warning
with nothing to report.

A test caught it, and only because of how it was written: it asserted a slow
burn on a fixture that should have produced one, and got back nothing. A test
that had merely checked for the absence of an error would have sailed past.

The fix is that the two numbers answer two questions. The budget spans the
whole window. The rate looks at a recent slice of it. A fleet that was fine
for a month and broke this morning has budget left and a high rate, and that
combination is the entire point of measuring both.

There is a sibling failure guarded in the same place. A burn rate of zero with
no runs behind it means we had no clock and could not compute a rate at all,
which is the opposite fact from a fleet burning nothing, and the two are
identical in the float. The report carries the count beside the rate and says
which one you are looking at.

## Two bars: one to look, one to act

An observed ratio below target is not the same as a demonstrated breach. Fifty
six good runs out of sixty against an objective of 0.95 is a third over budget
by the point estimate, and the confidence interval around it runs from about
0.84 to 0.97, which comfortably contains the target. Fifty six out of sixty
just isn't enough to say a fleet is unreliable.

So there are two bars. The report shows the point estimate, the interval, the
budget and the trigger, all of it, because a person reading a report can see
the sample size and judge. The event bus is stricter: an event goes out only
when the upper end of the interval sits below the objective. A line on the bus
is read by a program that never sees the interval, and paging somebody for a
number that could be luck is how they learn to filter the sender, which costs
you the one that mattered.

We use the Wilson interval rather than the textbook normal one, because the
normal interval is wrong exactly where this work lives: near 1.0, on the small
samples a single agent produces, it will happily report an upper bound above
1.

A related decision: a slow burn is reported and never sent. Severity is fixed
per event type in our envelope, so one type is one paging band, and "the
budget will be gone by Friday" does not belong in the band reserved for "the
budget is gone". The slow burn figure is in the report and in the JSON output
where a dashboard can read it without waking anybody up.

## What we could not measure

There is no latency indicator, and not because we decided against one. The
gateway's call record carries a single settle timestamp and no duration field
at all, so there is no latency in the data to compute. We could have
subtracted two timestamps and called the difference latency. It would have
been the agent's think time plus whatever the operator's queue was doing,
reported as one number, and somebody would eventually have made a decision on
it.

The quality indicator has a gap too. Scores live in an evaluation store whose
run table carries no agent identity, so they cannot be joined to a fleet
subject without the caller supplying them. Until that changes, the indicator
reports itself as not measured, with a sentence saying where the missing input
would have to come from.

Every report also prints what it is blind to: which indicators had too few
eligible runs, and which subjects have burn windows too thin to ever fire at
the observed event rate. An operator seeing four calm indicators deserves to
know which of them are calm because nothing is wrong and which are calm
because nothing was measured.

## What we did not build, and why

The obvious next move is to let the budget gate the agent: budget exhausted,
drop from autonomous to approval required. We did not build it, and the
reasons are structural rather than a matter of effort. They're worth stating
because anyone copying this design will hit the same wall.

Our policy decision point is a pure function of the loaded policy set and the
request. That is enforced, not merely intended: a script fails the build if
the decision path imports the network, a database or a clock. So it can't look
a budget up. A gate would have to work by writing policy rather than by
reading state, which is a stranger design than it sounds and probably the
right one anyway.

There is no per-agent autonomy level anywhere in our stack to write into. No
mode, no tier, no state. The one approval switch we have is conditional on the
cost of the individual call, and it is skipped entirely when its threshold is
zero, so "hold everything" is not currently expressible.

And the identity we can soundly key a budget on is not the identity a policy
can target. The agent identifier on a call is a header the caller writes,
which is fine for attribution a cooperating fleet reports about itself and
unsound for anything enforced, because a caller can move off a budget by
sending a different one. The identifier resolved server side from the
presented credential is sound, and it is empty unless an operator has turned
client keys on. Policies target the first one. The budget should key on the
second. The join between them is a mapping that is off by default.

A gate built across that seam would look identical whether it was holding or
not, which is the failure this whole exercise exists to end. So the tool
measures, says what it measured, and leaves the decision to a person. We would
rather ship that than a gate whose holding we could not demonstrate.

The same seam shows up in a smaller way at the event bus. When you group by
credential, which is the sound choice, the subject of the measurement is a key
and not an agent, and our event envelope has exactly one subject field with a
fixed grammar for agent identifiers. So a credential keyed breach appears in
the report and never on the bus. Writing it there malformed for a consumer to
reject later would be worse than not writing it at all.

## If you want to copy this

Five things carried most of the value here, and none of them is specific to
our stack.

Count runs on each side of a line instead of averaging a score. Compute the
budget over the window and the rate over a recent slice, or one of them is the
other with its sign flipped. Put the sample size next to every ratio and hold
the alerting path to a higher bar than the report. Print what you could not
measure in the same output as what you could. And when the mechanism you want
to build cannot be built soundly on the ground you have, say so in the tool
instead of shipping something that looks like it works. That last one is the
one we would have got wrong a year ago, and it is the reason this document has
a section about what we did not build.

---

`@claude` unless marked otherwise. The design decisions here are ours and are
open to argument; the measurements about our own code were taken on 2026-08-26
against the implementation in this repository. The default that counts an
escalation as a failure is the one we expect to be argued with first, and
`--good-outcomes` exists because that argument is legitimate.
