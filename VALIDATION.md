# Live infrastructure validation

Verdryx computed cost-per-outcome from real Parquet traces produced by a live, Claude-backed multi-agent
run on disposable Hetzner infrastructure before any public launch - the first time it had run against
real outcome tags rather than a synthetic trace.

## Cost-per-outcome from a real support-agent run

Real outcome tags (`case_resolved` / `abandoned` / `escalated`) from a live support-agent workload
produced a genuine cost-quality split that a dollar total alone cannot show:

| Outcome | Share | Cost |
|---|---|---|
| Resolved | 60% | **$0.00042** per correctly resolved case |
| Abandoned | 20% | **$0.00025** spent for nothing |
| Escalated | 20% | (handed to a human) |

Quality drift across the run was flagged **stable** - no degradation in the resolved/abandoned/escalated
mix over the course of the campaign.

## What this proves

- Cost-per-outcome is a materially different (and more honest) number than cost-per-call: an abandoned
  attempt still cost real money for zero result, which a spend total alone hides.
- The `CostPerOutcomeReport`, `DriftReport`, and `ExactGrader` engines all ran end to end against a real
  Parquet trace from a live multi-agent campaign, not a synthetic fixture.
- Drift detection held correctly stable across the run, with no false-positive drift alerts.

## Method

Disposable Hetzner VPS boxes (deleted after each run), Verdryx reading Parquet traces produced by a real
Claude-backed gateway run; code delivered as a `git archive` tarball (no secrets, no `.git`, no token).
Nothing from these runs was ever exposed publicly, and no infrastructure or secret from the campaign
persists today.
