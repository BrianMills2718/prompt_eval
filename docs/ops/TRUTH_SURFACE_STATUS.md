Truth Surface Status
- Overall: fail
- Issues: 2
- Fail: 2
- Warn: 0
- Info: 0
- Findings:
  - [FAIL] consumed_reservation_missing_plan_file: Consumed reservation points to missing plan file: /home/brian/projects/prompt_eval_worktrees/plan-60-prompt-eval-coordination/docs/plans/14_authoritative-coordination-wave-1-rollout.md
  - [FAIL] consumed_reservation_missing_plan_file: Consumed reservation points to missing plan file: /home/brian/projects/prompt_eval_worktrees/plan-15-truth-surface-pilot/docs/plans/15_truth-surface-adoption-pilot.md
- Semantic Review: warn
- Semantic Findings: 3
- Semantic Warn: 3
- Semantic Info: 0
- Semantic Promotion Candidates: 3
- Semantic Overview: The semantic review found meaningful drift related to the description of current work. Specifically, the tracker file indicates that Plan 15 is complete and suggests using it as evidence for shared follow-on work, but the deterministic findings show that Plan 15 (and Plan 14) have missing plan files, indicating an inconsistency between the declared status and the physical availability of the plan documentation. The 'Current Default Next Step' section also suggests an action based on Plan 15, which is problematic given its reported missing status.
- Semantic Advisory Findings:
  - [WARN] misleading_summary: The tracker file indicates that Plan 15 is complete and suggests using it as a reference for future work, but the deterministic validator reports the plan file as missing.
    evidence: tracker_file, deterministic_issue_2
    promotion_candidate: yes
    promotion_rule_hint: A deterministic rule could check if any plan marked 'Complete' or referenced as 'Current Default Next Step' in the tracker file has a corresponding missing plan file according to consumed reservations.
  - [WARN] stale_prose: The 'Current Default Next Step' section points to a plan that the deterministic validator identifies as having a missing file.
    evidence: tracker_file, deterministic_issue_2
    promotion_candidate: yes
    promotion_rule_hint: A deterministic rule could flag any 'Current Default Next Step' reference to a plan ID that is reported as missing a file by a 'consumed_reservation_missing_plan_file' issue.
  - [WARN] misleading_summary: Plan 14 is listed as 'Complete' in the tracker file, but its associated plan file is reported as missing by the deterministic validator.
    evidence: tracker_file, deterministic_issue_1
    promotion_candidate: yes
    promotion_rule_hint: A deterministic rule could check if any plan marked 'Complete' in the tracker file corresponds to a 'consumed_reservation_missing_plan_file' deterministic issue.
