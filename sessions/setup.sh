#!/usr/bin/env bash
# Usage: ./setup.sh <session_name>
# Creates (or resets) a session folder from the agent/ template.
#
# Example:
#   ./setup.sh engineer_1
#   ./setup.sh engineer_2

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <session_name>"
    exit 1
fi

SESSION_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$SCRIPT_DIR/../agent"
SESSION_DIR="$SCRIPT_DIR/$SESSION_NAME"

if [ ! -d "$AGENT_DIR" ]; then
    echo "Error: agent/ directory not found at $AGENT_DIR"
    exit 1
fi

if [ -d "$SESSION_DIR" ]; then
    echo "Session '$SESSION_NAME' already exists at $SESSION_DIR"
    read -p "Reset it? This will delete all contents. [y/N] " confirm
    if [[ "$confirm" != [yY] ]]; then
        echo "Aborted."
        exit 0
    fi
    rm -rf "$SESSION_DIR"
fi

mkdir -p "$SESSION_DIR"

# Copy agent template files
cp "$AGENT_DIR/CLAUDE.md" "$SESSION_DIR/"
cp "$AGENT_DIR/.claudeignore" "$SESSION_DIR/"
cp "$AGENT_DIR/.mcp.json" "$SESSION_DIR/"
cp -r "$AGENT_DIR/previous_run" "$SESSION_DIR/"
mkdir -p "$SESSION_DIR/ref_documents"
cp "$AGENT_DIR/ref_documents/task_description.pdf" "$SESSION_DIR/ref_documents/"

# Copy YAML input files to session root
cp "$AGENT_DIR/inputs/"*.yaml "$SESSION_DIR/"

echo "Session '$SESSION_NAME' ready at:"
echo "  $SESSION_DIR"

# Open in VS Code
code "$SESSION_DIR"
