# === META-PROCESS TARGETS ===
# Added by meta-process install.sh

# Configuration
SCRIPTS_META := scripts/meta
PLANS_DIR := docs/plans
GITHUB_ACCOUNT ?= BrianMills2718
PR_AUTO_EXPECTED_REPO ?= $(notdir $(CURDIR))

# --- Session Start ---
.PHONY: status

status:  ## Show git status
	@git status --short --branch

# >>> META-PROCESS WORKTREE TARGETS >>>
WORKTREE_CREATE_SCRIPT := scripts/meta/worktree-coordination/create_worktree.py
WORKTREE_REMOVE_SCRIPT := scripts/meta/worktree-coordination/safe_worktree_remove.py
WORKTREE_CLAIMS_SCRIPT := scripts/meta/worktree-coordination/../check_coordination_claims.py
WORKTREE_SESSION_START_SCRIPT := scripts/meta/worktree-coordination/../session_start.py
WORKTREE_SESSION_HEARTBEAT_SCRIPT := scripts/meta/worktree-coordination/../session_heartbeat.py
WORKTREE_SESSION_STATUS_SCRIPT := scripts/meta/worktree-coordination/../session_status.py
WORKTREE_SESSION_FINISH_SCRIPT := scripts/meta/worktree-coordination/../session_finish.py
WORKTREE_SESSION_CLOSE_SCRIPT := scripts/meta/worktree-coordination/../session_close.py
WORKTREE_REVIEW_CLAIM_SCRIPT := scripts/meta/worktree-coordination/create_review_claim.py
WORKTREE_RAISE_CONCERN_SCRIPT := scripts/meta/worktree-coordination/raise_concern.py
WORKTREE_DIR ?= $(shell python "$(WORKTREE_CREATE_SCRIPT)" --repo-root . --print-default-worktree-dir)
WORKTREE_START_POINT ?= HEAD
WORKTREE_PROJECT ?= $(notdir $(CURDIR))
WORKTREE_AGENT ?= $(shell if [ -n "$$CODEX_THREAD_ID" ]; then printf codex; elif [ -n "$$CLAUDE_SESSION_ID" ] || [ -n "$$CLAUDE_CODE_SSE_PORT" ]; then printf claude-code; elif [ -n "$$OPENCLAW_SESSION_ID" ] || [ -n "$$OPENCLAW_RUN_ID" ]; then printf openclaw; fi)
SESSION_GOAL ?=
SESSION_PHASE ?=
SESSION_NEXT ?=
SESSION_DEPENDS ?=
SESSION_STOP_CONDITIONS ?=
SESSION_NOTE ?=
REVIEW_SCOPE ?=
REVIEW_NOTES ?=
RECIPIENT ?=

.PHONY: worktree worktree-list worktree-remove session-start session-heartbeat session-status session-finish session-close review-claim raise-concern

worktree:  ## Create claimed worktree (BRANCH=name TASK="..." [PLAN=N] [AGENT=name])
ifndef BRANCH
	$(error BRANCH is required. Usage: make worktree BRANCH=plan-42-feature TASK="Describe the task")
endif
ifndef TASK
	$(error TASK is required. Usage: make worktree BRANCH=plan-42-feature TASK="Describe the task")
endif
ifndef SESSION_GOAL
	$(error SESSION_GOAL is required. Name the broader objective, not the local branch)
endif
ifndef SESSION_PHASE
	$(error SESSION_PHASE is required. Describe the current execution phase)
endif
ifndef WORKTREE_AGENT
	$(error Unable to infer agent runtime. Set AGENT via WORKTREE_AGENT=codex|claude-code|openclaw)
endif
	@if [ ! -f "$(WORKTREE_CREATE_SCRIPT)" ]; then \
		echo "Missing worktree coordination module: $(WORKTREE_CREATE_SCRIPT)"; \
		echo "Install or sync the sanctioned worktree-coordination module before using make worktree."; \
		exit 1; \
	fi
	@if [ ! -f "$(WORKTREE_CLAIMS_SCRIPT)" ]; then \
		echo "Missing worktree coordination module: $(WORKTREE_CLAIMS_SCRIPT)"; \
		echo "Install or sync the sanctioned worktree-coordination module before using make worktree."; \
		exit 1; \
	fi
	@if [ ! -f "$(WORKTREE_SESSION_START_SCRIPT)" ]; then \
		echo "Missing session lifecycle module: $(WORKTREE_SESSION_START_SCRIPT)"; \
		echo "Install or sync the sanctioned session lifecycle module before using make worktree."; \
		exit 1; \
	fi
	@python "$(WORKTREE_CLAIMS_SCRIPT)" --claim \
		--agent "$(WORKTREE_AGENT)" \
		--project "$(WORKTREE_PROJECT)" \
		--scope "$(BRANCH)" \
		--intent "$(TASK)" \
		--claim-type program \
		--branch "$(BRANCH)" \
		--worktree-path "$(WORKTREE_DIR)/$(BRANCH)" \
		$(if $(PLAN),--plan "Plan #$(PLAN)",)
	@mkdir -p "$(WORKTREE_DIR)"
	@if ! python "$(WORKTREE_CREATE_SCRIPT)" --repo-root . --path "$(WORKTREE_DIR)/$(BRANCH)" --branch "$(BRANCH)" --start-point "$(WORKTREE_START_POINT)"; then \
		python "$(WORKTREE_CLAIMS_SCRIPT)" --release --agent "$(WORKTREE_AGENT)" --project "$(WORKTREE_PROJECT)" --scope "$(BRANCH)" >/dev/null 2>&1 || true; \
		exit 1; \
	fi
	@if ! python "$(WORKTREE_SESSION_START_SCRIPT)" \
		--agent "$(WORKTREE_AGENT)" \
		--project "$(WORKTREE_PROJECT)" \
		--scope "$(BRANCH)" \
		--intent "$(TASK)" \
		--repo-root "$(CURDIR)" \
		--worktree-path "$(WORKTREE_DIR)/$(BRANCH)" \
		--branch "$(BRANCH)" \
		--broader-goal "$(SESSION_GOAL)" \
		--current-phase "$(SESSION_PHASE)" \
		$(if $(PLAN),--plan "Plan #$(PLAN)",) \
		$(if $(SESSION_NEXT),--next-phase "$(SESSION_NEXT)",) \
		$(if $(SESSION_DEPENDS),--depends-on "$(SESSION_DEPENDS)",) \
		$(if $(SESSION_STOP_CONDITIONS),--stop-condition "$(SESSION_STOP_CONDITIONS)",) \
		$(if $(SESSION_NOTE),--notes "$(SESSION_NOTE)",); then \
		git worktree remove --force "$(WORKTREE_DIR)/$(BRANCH)" >/dev/null 2>&1 || true; \
		git branch -D "$(BRANCH)" >/dev/null 2>&1 || true; \
		python "$(WORKTREE_CLAIMS_SCRIPT)" --release --agent "$(WORKTREE_AGENT)" --project "$(WORKTREE_PROJECT)" --scope "$(BRANCH)" >/dev/null 2>&1 || true; \
		exit 1; \
	fi
	@echo ""
	@echo "Worktree created at $(WORKTREE_DIR)/$(BRANCH)"
	@echo "Claim created for branch $(BRANCH)"
	@echo "Session contract started for $(SESSION_GOAL)"

session-start:  ## Create or refresh the active session contract for BRANCH=name
ifndef BRANCH
	$(error BRANCH is required. Usage: make session-start BRANCH=plan-42-feature TASK="..." SESSION_GOAL="..." SESSION_PHASE="...")
endif
ifndef TASK
	$(error TASK is required. Usage: make session-start BRANCH=plan-42-feature TASK="...")
endif
ifndef SESSION_GOAL
	$(error SESSION_GOAL is required. Name the broader objective, not the local branch)
endif
ifndef SESSION_PHASE
	$(error SESSION_PHASE is required. Describe the current execution phase)
endif
ifndef WORKTREE_AGENT
	$(error Unable to infer agent runtime. Set AGENT via WORKTREE_AGENT=codex|claude-code|openclaw)
endif
	@python "$(WORKTREE_SESSION_START_SCRIPT)" \
		--agent "$(WORKTREE_AGENT)" \
		--project "$(WORKTREE_PROJECT)" \
		--scope "$(BRANCH)" \
		--intent "$(TASK)" \
		--repo-root "$(CURDIR)" \
		--worktree-path "$(WORKTREE_DIR)/$(BRANCH)" \
		--branch "$(BRANCH)" \
		--broader-goal "$(SESSION_GOAL)" \
		--current-phase "$(SESSION_PHASE)" \
		$(if $(PLAN),--plan "Plan #$(PLAN)",) \
		$(if $(SESSION_NEXT),--next-phase "$(SESSION_NEXT)",) \
		$(if $(SESSION_DEPENDS),--depends-on "$(SESSION_DEPENDS)",) \
		$(if $(SESSION_STOP_CONDITIONS),--stop-condition "$(SESSION_STOP_CONDITIONS)",) \
		$(if $(SESSION_NOTE),--notes "$(SESSION_NOTE)",)

session-heartbeat:  ## Refresh heartbeat and optional phase for BRANCH=name
ifndef BRANCH
	$(error BRANCH is required. Usage: make session-heartbeat BRANCH=plan-42-feature)
endif
ifndef WORKTREE_AGENT
	$(error Unable to infer agent runtime. Set AGENT via WORKTREE_AGENT=codex|claude-code|openclaw)
endif
	@python "$(WORKTREE_SESSION_HEARTBEAT_SCRIPT)" \
		--agent "$(WORKTREE_AGENT)" \
		--project "$(WORKTREE_PROJECT)" \
		--scope "$(BRANCH)" \
		--branch "$(BRANCH)" \
		$(if $(SESSION_PHASE),--current-phase "$(SESSION_PHASE)",)

session-status:  ## Show live session summaries for this repo
	@python "$(WORKTREE_SESSION_STATUS_SCRIPT)" --project "$(WORKTREE_PROJECT)"

session-finish:  ## Finish the session for BRANCH=name; blocks if the worktree is dirty
ifndef BRANCH
	$(error BRANCH is required. Usage: make session-finish BRANCH=plan-42-feature)
endif
ifndef WORKTREE_AGENT
	$(error Unable to infer agent runtime. Set AGENT via WORKTREE_AGENT=codex|claude-code|openclaw)
endif
	@python "$(WORKTREE_SESSION_FINISH_SCRIPT)" \
		--agent "$(WORKTREE_AGENT)" \
		--project "$(WORKTREE_PROJECT)" \
		--scope "$(BRANCH)" \
		--worktree-path "$(WORKTREE_DIR)/$(BRANCH)" \
		$(if $(SESSION_NOTE),--note "$(SESSION_NOTE)",)

session-close:  ## Close the claimed lane for BRANCH=name: cleanup worktree + branch + claim together
ifndef BRANCH
	$(error BRANCH is required. Usage: make session-close BRANCH=plan-42-feature)
endif
ifndef WORKTREE_AGENT
	$(error Unable to infer agent runtime. Set AGENT via WORKTREE_AGENT=codex|claude-code|openclaw)
endif
	@python "$(WORKTREE_SESSION_CLOSE_SCRIPT)" \
		--agent "$(WORKTREE_AGENT)" \
		--project "$(WORKTREE_PROJECT)" \
		--scope "$(BRANCH)" \
		--worktree-path "$(WORKTREE_DIR)/$(BRANCH)" \
		--branch "$(BRANCH)" \
		$(if $(SESSION_NOTE),--note "$(SESSION_NOTE)",)

worktree-list:  ## Show claimed worktree coordination status
	@if [ ! -f "$(WORKTREE_CLAIMS_SCRIPT)" ]; then \
		echo "Missing worktree coordination module: $(WORKTREE_CLAIMS_SCRIPT)"; \
		echo "Install or sync the sanctioned worktree-coordination module before using make worktree-list."; \
		exit 1; \
	fi
	@python "$(WORKTREE_CLAIMS_SCRIPT)" --list

worktree-remove:  ## Safely remove worktree for BRANCH=name
ifndef BRANCH
	$(error BRANCH is required. Usage: make worktree-remove BRANCH=plan-42-feature)
endif
	@if [ ! -f "$(WORKTREE_SESSION_CLOSE_SCRIPT)" ]; then \
		echo "Missing session lifecycle module: $(WORKTREE_SESSION_CLOSE_SCRIPT)"; \
		echo "Install or sync the sanctioned session lifecycle module before using make worktree-remove."; \
		exit 1; \
	fi
	@$(MAKE) session-close BRANCH="$(BRANCH)" $(if $(SESSION_NOTE),SESSION_NOTE="$(SESSION_NOTE)",)

review-claim:  ## Create a review claim for TARGET_BRANCH=name WRITE_PATHS="a|b" TASK="..."
ifndef TARGET_BRANCH
	$(error TARGET_BRANCH is required. Usage: make review-claim TARGET_BRANCH=plan-42-feature WRITE_PATHS="src/foo.py|tests/test_foo.py" TASK="Review concern")
endif
ifndef WRITE_PATHS
	$(error WRITE_PATHS is required. Provide one or more repo-relative paths separated by '|')
endif
ifndef TASK
	$(error TASK is required. Describe the review intent)
endif
ifndef WORKTREE_AGENT
	$(error Unable to infer agent runtime. Set AGENT via WORKTREE_AGENT=codex|claude-code|openclaw)
endif
	@python "$(WORKTREE_REVIEW_CLAIM_SCRIPT)" \
		--repo-root "$(CURDIR)" \
		--agent "$(WORKTREE_AGENT)" \
		--project "$(WORKTREE_PROJECT)" \
		--target-branch "$(TARGET_BRANCH)" \
		--intent "$(TASK)" \
		--write-path "$(WRITE_PATHS)" \
		$(if $(PLAN),--plan "Plan #$(PLAN)",) \
		$(if $(REVIEW_SCOPE),--scope "$(REVIEW_SCOPE)",) \
		$(if $(REVIEW_NOTES),--notes "$(REVIEW_NOTES)",)

raise-concern:  ## Route concern to TARGET_BRANCH via PR comment or local inbox
ifndef TARGET_BRANCH
	$(error TARGET_BRANCH is required. Usage: make raise-concern TARGET_BRANCH=plan-42-feature SUBJECT="..." MESSAGE="...")
endif
ifndef SUBJECT
	$(error SUBJECT is required. Usage: make raise-concern TARGET_BRANCH=plan-42-feature SUBJECT="..." MESSAGE="...")
endif
ifndef WORKTREE_AGENT
	$(error Unable to infer agent runtime. Set AGENT via WORKTREE_AGENT=codex|claude-code|openclaw)
endif
ifndef MESSAGE
ifndef MESSAGE_FILE
	$(error MESSAGE or MESSAGE_FILE is required. Provide inline content or a path to a concern file)
endif
endif
	@python "$(WORKTREE_RAISE_CONCERN_SCRIPT)" \
		--repo-root "$(CURDIR)" \
		--agent "$(WORKTREE_AGENT)" \
		--project "$(WORKTREE_PROJECT)" \
		--target-branch "$(TARGET_BRANCH)" \
		--subject "$(SUBJECT)" \
		$(if $(MESSAGE),--content "$(MESSAGE)",) \
		$(if $(MESSAGE_FILE),--content-file "$(MESSAGE_FILE)",) \
		$(if $(RECIPIENT),--recipient "$(RECIPIENT)",)
# <<< META-PROCESS WORKTREE TARGETS <<<

# --- During Implementation ---
.PHONY: test test-quick publish-check-extra check

CANONICAL_REPO_ROOT := $(shell common_dir="$$(git -C "$(CURDIR)" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"; if [ -n "$$common_dir" ]; then dirname "$$common_dir"; else printf '%s\n' "$(CURDIR)"; fi)
REPO_PYTHON := $(if $(wildcard $(CANONICAL_REPO_ROOT)/.venv/bin/python),$(CANONICAL_REPO_ROOT)/.venv/bin/python,python3)
LLM_CLIENT_ROOT := $(abspath $(CANONICAL_REPO_ROOT)/../llm_client)

test:  ## Run pytest
	$(REPO_PYTHON) -m pytest tests/ -v

test-quick:  ## Run pytest (no traceback)
	$(REPO_PYTHON) -m pytest tests/ -q --tb=no

check:  ## Run all checks (test, mypy, lint)
	@if [ ! -d "$(LLM_CLIENT_ROOT)/llm_client" ]; then \
		echo "Missing sibling llm_client checkout at $(LLM_CLIENT_ROOT)"; \
		echo "prompt_eval publish-check-extra requires llm_client source for mypy import resolution."; \
		exit 1; \
	fi
	@echo "Running tests..."
	@$(REPO_PYTHON) -m pytest tests/ -q --tb=short
	@echo ""
	@echo "Running mypy..."
	@MYPYPATH="$(LLM_CLIENT_ROOT)" $(REPO_PYTHON) -m mypy prompt_eval tests
	@echo ""
	@echo "Running ruff..."
	@$(REPO_PYTHON) -m ruff check prompt_eval tests
	@echo ""
	@echo "All checks passed!"

# --- PR Workflow ---
.PHONY: pr-ready pr merge finish pr-auto-check pr-auto

pr-ready:  ## Rebase on main and push
	@git fetch origin main
	@git rebase origin/main
	@git push -u origin HEAD

pr:  ## Create PR (opens browser)
	@gh pr create --fill --web

pr-auto-check:  ## Autonomous PR preflight (branch/clean tree/origin/account)
	@python $(SCRIPTS_META)/pr_auto.py --preflight-only --expected-origin-repo $(PR_AUTO_EXPECTED_REPO) --account $(GITHUB_ACCOUNT)

pr-auto:  ## Autonomous PR create + auto-merge request (non-interactive)
	@python $(SCRIPTS_META)/pr_auto.py --expected-origin-repo $(PR_AUTO_EXPECTED_REPO) --account $(GITHUB_ACCOUNT) --fill --auto-merge

merge:  ## Merge PR (PR=number required)
ifndef PR
	$(error PR is required. Usage: make merge PR=123)
endif
	@python $(SCRIPTS_META)/merge_pr.py $(PR)

finish:  ## Merge PR + cleanup branch (BRANCH=name PR=number required)
ifndef BRANCH
	$(error BRANCH is required. Usage: make finish BRANCH=plan-42-feature PR=123)
endif
ifndef PR
	$(error PR is required. Usage: make finish BRANCH=plan-42-feature PR=123)
endif
	@gh pr merge $(PR) --squash --delete-branch
	@git checkout main && git pull --ff-only
	@git branch -d $(BRANCH) 2>/dev/null || true

# --- Plans ---
.PHONY: plan-tests plan-complete

plan-tests:  ## Check plan's required tests (PLAN=N required)
ifndef PLAN
	$(error PLAN is required. Usage: make plan-tests PLAN=42)
endif
	@python $(SCRIPTS_META)/check_plan_tests.py --plan $(PLAN)

plan-complete:  ## Mark plan complete with verification (PLAN=N required)
ifndef PLAN
	$(error PLAN is required. Usage: make plan-complete PLAN=42)
endif
	@python $(SCRIPTS_META)/complete_plan.py --plan $(PLAN)

# --- Quality ---
.PHONY: dead-code dead-code-audit dead-code-validate

dead-code:  ## Run dead code detection
	@python $(SCRIPTS_META)/check_dead_code.py

dead-code-audit:  ## Refresh reviewed dead-code audit file
	@python $(SCRIPTS_META)/audit_dead_code.py --write

dead-code-validate:  ## Validate reviewed dead-code dispositions
	@python $(SCRIPTS_META)/validate_dead_code_audit.py

# >>> META-PROCESS PUBLISH TARGETS >>>
PUBLISH_PUSH_CHECK_SCRIPT := scripts/meta/check_push_safety.py
PUBLISH_DEAD_CODE_SCRIPT := scripts/meta/check_dead_code.py
PUBLISH_DEAD_CODE_VALIDATE_SCRIPT := scripts/meta/validate_dead_code_audit.py

.PHONY: publish-check

publish-check:  ## Run the governed publish gate (coordination, repo checks, reviewed dead-code)
	@if [ ! -f "$(PUBLISH_PUSH_CHECK_SCRIPT)" ]; then \
		echo "Missing push-safety validator: $(PUBLISH_PUSH_CHECK_SCRIPT)"; \
		echo "Install or sync the sanctioned governed-repo support before publishing."; \
		exit 1; \
	fi
	@if [ ! -f "$(PUBLISH_DEAD_CODE_SCRIPT)" ]; then \
		echo "Missing dead-code detector: $(PUBLISH_DEAD_CODE_SCRIPT)"; \
		echo "Install or sync the sanctioned governed-repo support before publishing."; \
		exit 1; \
	fi
	@if [ ! -f "$(PUBLISH_DEAD_CODE_VALIDATE_SCRIPT)" ]; then \
		echo "Missing dead-code audit validator: $(PUBLISH_DEAD_CODE_VALIDATE_SCRIPT)"; \
		echo "Install or sync the sanctioned governed-repo support before publishing."; \
		exit 1; \
	fi
	@python "$(PUBLISH_PUSH_CHECK_SCRIPT)"
	@python "$(PUBLISH_DEAD_CODE_SCRIPT)"
	@python "$(PUBLISH_DEAD_CODE_VALIDATE_SCRIPT)"
	@if $(MAKE) -n publish-check-extra >/dev/null 2>&1; then \
		$(MAKE) publish-check-extra; \
	fi

publish-check-extra:  ## Run repo-local publish blockers beyond the shared governed gate
	@$(MAKE) check
# <<< META-PROCESS PUBLISH TARGETS <<<

# --- Help ---
.PHONY: help-meta

help-meta:  ## Show meta-process targets
	@echo "Meta-Process Targets:"
	@echo ""
	@echo "  Session:"
	@echo "    status          Show git status"
	@echo ""
	@echo "  Development:"
	@echo "    test            Run tests"
	@echo "    check           Run all checks"
	@echo ""
	@echo "  PR Workflow:"
	@echo "    pr-ready        Rebase + push"
	@echo "    pr              Create PR"
	@echo "    pr-auto-check   Preflight autonomous PR flow"
	@echo "    pr-auto         Non-interactive PR + auto-merge request"
	@echo "    merge           Merge PR (PR=number)"
	@echo "    finish          Merge + cleanup (BRANCH=name PR=number)"
	@echo ""
	@echo "  Quality:"
	@echo "    dead-code       Run dead code detection"
	@echo "    dead-code-audit Refresh reviewed dead-code audit file"
	@echo "    dead-code-validate Validate reviewed dead-code dispositions"
	@echo ""
	@echo "  Plans:"
	@echo "    plan-tests      Check plan tests (PLAN=N)"
	@echo "    plan-complete   Complete plan (PLAN=N)"
