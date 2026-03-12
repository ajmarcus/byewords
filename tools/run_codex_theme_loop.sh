#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

CODEX_BIN=${CODEX_BIN:-codex}
MAX_ITERS=${CODEX_THEME_LOOP_MAX_ITERS:-100}
SLEEP_SECONDS=${CODEX_THEME_LOOP_SLEEP_SECONDS:-0}
SANDBOX_MODE=${CODEX_THEME_LOOP_SANDBOX:-workspace-write}

read -r -d '' PROMPT <<'EOF' || true
read docs/theme.md and continue implementation in
small testable chunks that represent a
meaningful milestone you can measure and
confirm. update docs/theme.md when you are
done with the chunk. finally commit and push
your changes. once the full docs/theme.md
plan is done, just return <DONE>.
EOF

if ! [[ "$MAX_ITERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "CODEX_THEME_LOOP_MAX_ITERS must be a positive integer" >&2
    exit 1
fi

if ! [[ "$SLEEP_SECONDS" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    echo "CODEX_THEME_LOOP_SLEEP_SECONDS must be a non-negative number" >&2
    exit 1
fi

case "$SANDBOX_MODE" in
    read-only | workspace-write | danger-full-access) ;;
    *)
        echo "CODEX_THEME_LOOP_SANDBOX must be one of: read-only, workspace-write, danger-full-access" >&2
        exit 1
        ;;
esac

if ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
    echo "Could not find Codex executable: $CODEX_BIN" >&2
    exit 1
fi

if [[ ! -f "$REPO_ROOT/docs/theme.md" ]]; then
    echo "Could not find theme plan: $REPO_ROOT/docs/theme.md" >&2
    exit 1
fi

OUTPUT_DIR=$(mktemp -d)
OUTPUT_FILE=$OUTPUT_DIR/last_message.txt
CODEX_STDOUT_FILE=$OUTPUT_DIR/codex_stdout.txt
cleanup() {
    rm -rf "$OUTPUT_DIR"
}
trap cleanup EXIT

is_done_message() {
    local message=${1//$'\r'/}
    [[ "$message" =~ ^[[:space:]]*\<DONE\>[[:space:]]*$ ]]
}

run_codex_once() {
    rm -f "$OUTPUT_FILE"
    : >"$CODEX_STDOUT_FILE"
    local -a cmd=(
        "$CODEX_BIN"
        exec
        --sandbox
        "$SANDBOX_MODE"
        --color
        never
        -C
        "$REPO_ROOT"
        -o
        "$OUTPUT_FILE"
        "$PROMPT"
    )
    if "${cmd[@]}" >"$CODEX_STDOUT_FILE"; then
        return 0
    fi

    if [[ -s "$CODEX_STDOUT_FILE" ]]; then
        cat "$CODEX_STDOUT_FILE" >&2
    fi
    return 1
}

for ((iteration = 1; iteration <= MAX_ITERS; iteration++)); do
    echo "codex theme loop iteration $iteration/$MAX_ITERS" >&2
    if ! run_codex_once; then
        echo "Codex command failed on iteration $iteration" >&2
        exit 1
    fi

    if [[ ! -s "$OUTPUT_FILE" ]]; then
        echo "Codex did not produce a non-empty output file" >&2
        exit 1
    fi

    last_message=$(tr -d '\r' <"$OUTPUT_FILE")
    printf '%s\n' "$last_message"

    if is_done_message "$last_message"; then
        exit 0
    fi

    if (( iteration < MAX_ITERS )) && [[ "$SLEEP_SECONDS" != "0" ]]; then
        sleep "$SLEEP_SECONDS"
    fi
done

echo "Codex theme loop reached $MAX_ITERS iterations without receiving <DONE>" >&2
exit 1
