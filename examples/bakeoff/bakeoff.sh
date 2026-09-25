#!/usr/bin/env bash
# The judge bake-off, one command: builds typryx at a pinned commit, starts
# one typryx per requested typryx-backed judge (qwen, jev, stub), grades the
# same dataset through verdryx's own run_eval, and writes a report. See
# README.md in this directory for what each judge is and what it costs.
#
# Spends nothing without a separate, explicit go-ahead:
#   - qwen and stub are always free (local Ollama / typryx's own stub backend).
#   - jev needs a Keychain key (service "typesafe-jev-api-key"); absent, it
#     is skipped and everything else still runs.
#   - claude never runs unless --judges names it AND BAKEOFF_CONFIRM_SPEND=yes
#     is set; without that, this script prints the estimate and exits before
#     doing anything else at all (no typryx build, no server, nothing).
#
# Usage:
#   ./bakeoff.sh [--judges qwen,jev,claude,stub] [--n 1000] [--seed 1]
#                [--cases FILE] [--out DIR] [--dry-run]
#
# Bash 3.2 compatible on purpose (macOS's system /bin/bash): no associative
# arrays, no `declare -A`, only plain indexed arrays and "$@" forwarding.

set -euo pipefail

# ------------------------------------------------------------------
# Pinned typryx commit. A variable, not a floating branch: built from a
# `git worktree` checkout of exactly this commit, so the build never
# depends on whatever is uncommitted in the ~/Development/typryx working
# tree at run time.
# ------------------------------------------------------------------
TYPRYX_COMMIT="2907caf"
TYPRYX_REPO="${TYPRYX_REPO:-$HOME/Development/typryx}"

QWEN_PORT=14321
JEV_PORT=14322
STUB_PORT=14323
FAKE_JEV_PORT=14330

CLAUDE_MODEL="${CLAUDE_MODEL:-claude-haiku-4-5-20251001}"

# ------------------------------------------------------------------
# Args
# ------------------------------------------------------------------
JUDGES="qwen,jev"
N=1000
SEED=1
CASES_FILE=""
OUT=""
DRY_RUN=0

while [ $# -gt 0 ]; do
	case "$1" in
	--judges)
		JUDGES="$2"
		shift 2
		;;
	--n)
		N="$2"
		shift 2
		;;
	--seed)
		SEED="$2"
		shift 2
		;;
	--cases)
		CASES_FILE="$2"
		shift 2
		;;
	--out)
		OUT="$2"
		shift 2
		;;
	--dry-run)
		DRY_RUN=1
		shift
		;;
	*)
		echo "bakeoff.sh: unknown argument $1" >&2
		exit 2
		;;
	esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# The Python that runs verdryx: $PYTHON if set, else this checkout's own
# .venv, else python3. Checked here, before anything is built or started,
# because a python3 without verdryx's one runtime dependency would otherwise
# fail only after typryx was built and servers were already up.
if [ -z "${PYTHON:-}" ]; then
	if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
		PYTHON="$REPO_ROOT/.venv/bin/python"
	else
		PYTHON="python3"
	fi
fi
if ! "$PYTHON" -c 'import rfc8785, verdryx' >/dev/null 2>&1; then
	echo "bakeoff.sh: $PYTHON cannot import verdryx. Create the venv first:" >&2
	echo "  python3 -m venv .venv && .venv/bin/pip install -e '.[anthropic]'   (the anthropic extra is needed only for the claude judge)" >&2
	echo "or point PYTHON at an interpreter that has it." >&2
	exit 1
fi

if [ -z "$OUT" ]; then
	OUT="$REPO_ROOT/examples/bakeoff/results/$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$OUT"

# ------------------------------------------------------------------
# Which judges were asked for, as a plain list (comma-split, bash 3.2 safe).
# ------------------------------------------------------------------
WANT_QWEN=0
WANT_JEV=0
WANT_CLAUDE=0
WANT_STUB=0
OLDIFS="$IFS"
IFS=,
for j in $JUDGES; do
	case "$j" in
	qwen) WANT_QWEN=1 ;;
	jev) WANT_JEV=1 ;;
	claude) WANT_CLAUDE=1 ;;
	stub) WANT_STUB=1 ;;
	*)
		echo "bakeoff.sh: unknown judge '$j' (know: qwen, jev, claude, stub)" >&2
		exit 2
		;;
	esac
done
IFS="$OLDIFS"

# ------------------------------------------------------------------
# claude gate, first, before anything else runs or spends a second of
# build/server time: prints the estimate and exits(1) if not confirmed.
# Pure Python, no network either way (see run.py's require_claude_confirmation).
# ------------------------------------------------------------------
if [ "$WANT_CLAUDE" -eq 1 ]; then
	if ! "$PYTHON" -m examples.bakeoff.run claude-estimate --n "$N" --model "$CLAUDE_MODEL"; then
		echo "bakeoff.sh: claude was requested without BAKEOFF_CONFIRM_SPEND=yes. Nothing else ran." >&2
		exit 1
	fi
fi

# ------------------------------------------------------------------
# Cleanup: kill only PIDs this script started, remove only tmpdirs/worktree
# this script created. Runs on every exit path.
# ------------------------------------------------------------------
PIDS=()
TMPDIRS=()
WORKTREE_DIR=""

cleanup() {
	for pid in "${PIDS[@]:-}"; do
		[ -n "$pid" ] && kill "$pid" >/dev/null 2>&1 || true
	done
	for pid in "${PIDS[@]:-}"; do
		[ -n "$pid" ] && wait "$pid" 2>/dev/null || true
	done
	if [ -n "$WORKTREE_DIR" ] && [ -d "$WORKTREE_DIR" ]; then
		git -C "$TYPRYX_REPO" worktree remove --force "$WORKTREE_DIR" >/dev/null 2>&1 || true
	fi
	for d in "${TMPDIRS[@]:-}"; do
		[ -n "$d" ] && rm -rf "$d"
	done
}
trap cleanup EXIT INT TERM

gen_key() {
	if command -v openssl >/dev/null 2>&1; then
		openssl rand -hex 8
	else
		# Fallback with no openssl: still local-only, throwaway, never a
		# real secret -- this authenticates a loopback process this script
		# itself started and stops on exit.
		echo "k${RANDOM}${RANDOM}${RANDOM}"
	fi
}

# ------------------------------------------------------------------
# Dataset: generate, or use an operator-supplied --cases file.
# ------------------------------------------------------------------
if [ -n "$CASES_FILE" ]; then
	CASES_PATH="$CASES_FILE"
	N_EFFECTIVE="$(grep -c . "$CASES_PATH")"
	echo "dataset: external file $CASES_PATH ($N_EFFECTIVE cases)"
	RUN_DATASET_FLAGS=(--external-cases "$CASES_PATH")
else
	CASES_PATH="$OUT/cases.jsonl"
	"$PYTHON" -m examples.bakeoff.dataset --n "$N" --seed "$SEED" --out "$CASES_PATH"
	N_EFFECTIVE="$N"
	RUN_DATASET_FLAGS=(--n "$N" --seed "$SEED")
fi

# ------------------------------------------------------------------
# TYPRYX_MAX_CALLS_PER_HOUR: the run's own N plus 5%, rounded up. This IS
# the spend cap for jev (an hourly cap on the paid backend), and it is
# printed so an operator can see it before anything runs against it.
# ------------------------------------------------------------------
CAP=$((N_EFFECTIVE + (N_EFFECTIVE * 5 + 99) / 100))
echo "TYPRYX_MAX_CALLS_PER_HOUR=$CAP (n=$N_EFFECTIVE plus 5%, rounded up)"

# ------------------------------------------------------------------
# Build typryx at the pinned commit, from a clean `git worktree` export
# (never the possibly-dirty ~/Development/typryx working tree).
# ------------------------------------------------------------------
JUDGES_JSON_ENTRIES=()

if [ "$WANT_QWEN" -eq 1 ] || [ "$WANT_JEV" -eq 1 ] || [ "$WANT_STUB" -eq 1 ]; then
	if ! command -v go >/dev/null 2>&1; then
		echo "bakeoff.sh: 'go' is not on PATH; cannot build typryx. Install Go to run qwen/jev/stub." >&2
		exit 1
	fi
	BUILD_DIR="$(mktemp -d)"
	TMPDIRS+=("$BUILD_DIR")
	WORKTREE_DIR="$BUILD_DIR/typryx-src"
	git -C "$TYPRYX_REPO" worktree add --detach "$WORKTREE_DIR" "$TYPRYX_COMMIT" >/dev/null
	mkdir -p "$OUT/bin"
	(cd "$WORKTREE_DIR" && go build -o "$OUT/bin/typryx" ./cmd/typryx)
	TYPRYX_BIN="$OUT/bin/typryx"
	TEMPLATES_DIR="$WORKTREE_DIR/examples/templates"
	echo "built typryx $TYPRYX_COMMIT -> $TYPRYX_BIN"
fi

start_typryx() {
	# $1 name, $2 port, $3 client_key, $4 judge_dir; remaining args are
	# extra TYPRYX_* KEY=VALUE env pairs for this judge's backend.
	local name="$1" port="$2" client_key="$3" judge_dir="$4"
	shift 4
	mkdir -p "$judge_dir/ledger"
	env \
		"TYPRYX_ADDR=127.0.0.1:$port" \
		"TYPRYX_KEYS=$client_key=agent://bakeoff.local/$name" \
		"TYPRYX_TEMPLATES=$TEMPLATES_DIR" \
		"TYPRYX_LEDGER_DIR=$judge_dir/ledger" \
		"TYPRYX_EVENTS=$judge_dir/events.ndjson" \
		"TYPRYX_TIMEOUT_MS=30000" \
		"TYPRYX_MAX_CALLS_PER_HOUR=$CAP" \
		"$@" \
		"$TYPRYX_BIN" >"$judge_dir/typryx.log" 2>&1 &
	local pid=$!
	PIDS+=("$pid")
	local tries=0
	while [ "$tries" -lt 40 ]; do
		if curl -fsS "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
			echo "judge $name: typryx up on port $port"
			return 0
		fi
		tries=$((tries + 1))
		sleep 0.5
	done
	echo "judge $name: typryx did not become healthy on port $port; skipping. See $judge_dir/typryx.log" >&2
	kill "$pid" >/dev/null 2>&1 || true
	return 1
}

add_typed_judge_json() {
	local name="$1" port="$2" client_key="$3" unpriced="$4"
	JUDGES_JSON_ENTRIES+=("{\"name\":\"$name\",\"kind\":\"typed\",\"typed_url\":\"http://127.0.0.1:$port\",\"typed_key\":\"$client_key\",\"typed_template\":\"eval.outcome_met\",\"cost_unpriced\":$unpriced}")
}

if [ "$WANT_QWEN" -eq 1 ]; then
	QWEN_KEY="$(gen_key)"
	if [ "$DRY_RUN" -eq 1 ]; then
		if start_typryx qwen "$QWEN_PORT" "$QWEN_KEY" "$OUT/qwen" "TYPRYX_BACKEND=stub"; then
			add_typed_judge_json qwen "$QWEN_PORT" "$QWEN_KEY" false
		fi
	else
		OLLAMA_TAGS="$(curl -fsS http://127.0.0.1:11434/api/tags 2>/dev/null || true)"
		if [ -z "$OLLAMA_TAGS" ] || ! printf '%s' "$OLLAMA_TAGS" | grep -q 'qwen2.5:7b'; then
			echo "judge qwen: Ollama not reachable at 127.0.0.1:11434, or qwen2.5:7b not pulled (ollama pull qwen2.5:7b). Skipping." >&2
		elif start_typryx qwen "$QWEN_PORT" "$QWEN_KEY" "$OUT/qwen" \
			"TYPRYX_BACKEND=openai-logprobs" \
			"TYPRYX_OPENAI_URL=http://127.0.0.1:11434/v1" \
			"TYPRYX_OPENAI_MODEL=qwen2.5:7b"; then
			add_typed_judge_json qwen "$QWEN_PORT" "$QWEN_KEY" false
		fi
	fi
fi

if [ "$WANT_STUB" -eq 1 ]; then
	STUB_KEY="$(gen_key)"
	if start_typryx stub "$STUB_PORT" "$STUB_KEY" "$OUT/stub" "TYPRYX_BACKEND=stub"; then
		add_typed_judge_json stub "$STUB_PORT" "$STUB_KEY" false
	fi
fi

if [ "$WANT_JEV" -eq 1 ]; then
	JEV_CLIENT_KEY="$(gen_key)"
	JEV_UNPRICED=true
	if [ -n "${TYPRYX_JEV_PRICE_PER_MTOK_INPUT:-}" ] || [ -n "${TYPRYX_JEV_PRICE_PER_MTOK_OUTPUT:-}" ]; then
		JEV_UNPRICED=false
	fi
	JEV_EXTRA_ENV=()
	JEV_OK=1
	if [ "$DRY_RUN" -eq 1 ]; then
		JEV_BACKEND_KEY="$(gen_key)"
		KEYDIR="$(mktemp -d)"
		chmod 700 "$KEYDIR"
		TMPDIRS+=("$KEYDIR")
		JEV_KEYFILE="$KEYDIR/jev.key"
		printf '%s' "$JEV_BACKEND_KEY" >"$JEV_KEYFILE"
		chmod 600 "$JEV_KEYFILE"
		"$PYTHON" -m examples.bakeoff.fake_jev --port "$FAKE_JEV_PORT" --key "$JEV_BACKEND_KEY" \
			>"$OUT/fake_jev.log" 2>&1 &
		FAKE_JEV_PID=$!
		PIDS+=("$FAKE_JEV_PID")
		tries=0
		FAKE_UP=0
		while [ "$tries" -lt 20 ]; do
			if grep -q FAKE_JEV_LISTENING "$OUT/fake_jev.log" 2>/dev/null; then
				FAKE_UP=1
				break
			fi
			tries=$((tries + 1))
			sleep 0.25
		done
		if [ "$FAKE_UP" -ne 1 ]; then
			echo "judge jev: fake_jev.py did not start; skipping. See $OUT/fake_jev.log" >&2
			JEV_OK=0
		else
			JEV_EXTRA_ENV=("TYPRYX_BACKEND=jev" "TYPRYX_JEV_KEY_FILE=$JEV_KEYFILE" "TYPRYX_JEV_URL=http://127.0.0.1:$FAKE_JEV_PORT/")
		fi
	else
		if ! JEV_BACKEND_KEY="$(security find-generic-password -s typesafe-jev-api-key -w 2>/dev/null)"; then
			echo 'judge jev: no key in Keychain (service "typesafe-jev-api-key"). Add one with:' >&2
			echo '  security add-generic-password -s typesafe-jev-api-key -a "$USER" -w' >&2
			echo "Skipping jev; the rest of the run still continues." >&2
			JEV_OK=0
		else
			KEYDIR="$(mktemp -d)"
			chmod 700 "$KEYDIR"
			TMPDIRS+=("$KEYDIR")
			JEV_KEYFILE="$KEYDIR/jev.key"
			printf '%s' "$JEV_BACKEND_KEY" >"$JEV_KEYFILE"
			chmod 600 "$JEV_KEYFILE"
			unset JEV_BACKEND_KEY
			JEV_EXTRA_ENV=("TYPRYX_BACKEND=jev" "TYPRYX_JEV_KEY_FILE=$JEV_KEYFILE")
		fi
	fi
	if [ "$JEV_OK" -eq 1 ]; then
		if [ "$JEV_UNPRICED" = true ]; then
			echo "judge jev: TYPRYX_JEV_PRICE_PER_MTOK_INPUT/_OUTPUT not set; jev's cost will be reported as unpriced, not \$0."
		fi
		if start_typryx jev "$JEV_PORT" "$JEV_CLIENT_KEY" "$OUT/jev" "${JEV_EXTRA_ENV[@]}"; then
			add_typed_judge_json jev "$JEV_PORT" "$JEV_CLIENT_KEY" "$JEV_UNPRICED"
		fi
	fi
fi

if [ "$WANT_CLAUDE" -eq 1 ]; then
	CLAUDE_BASE_URL_JSON="null"
	if [ -n "${ANTHROPIC_BASE_URL:-}" ]; then
		CLAUDE_BASE_URL_JSON="\"$ANTHROPIC_BASE_URL\""
	fi
	JUDGES_JSON_ENTRIES+=("{\"name\":\"claude\",\"kind\":\"llm_judge\",\"model\":\"$CLAUDE_MODEL\",\"base_url\":$CLAUDE_BASE_URL_JSON}")
fi

if [ "${#JUDGES_JSON_ENTRIES[@]}" -eq 0 ]; then
	echo "bakeoff.sh: no judge could be started (see the messages above). Nothing to grade." >&2
	exit 1
fi

JUDGES_CONFIG="$OUT/judges.json"
{
	printf '['
	first=1
	for entry in "${JUDGES_JSON_ENTRIES[@]}"; do
		[ "$first" -eq 1 ] || printf ','
		printf '%s' "$entry"
		first=0
	done
	printf ']\n'
} >"$JUDGES_CONFIG"

VERDRYX_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

"$PYTHON" -m examples.bakeoff.run run \
	--judges-config "$JUDGES_CONFIG" \
	--cases "$CASES_PATH" \
	--out "$OUT" \
	"${RUN_DATASET_FLAGS[@]}" \
	--verdryx-commit "$VERDRYX_COMMIT" \
	--typryx-commit "$TYPRYX_COMMIT"

echo "bake-off done: $OUT/report.md, $OUT/report.json"
