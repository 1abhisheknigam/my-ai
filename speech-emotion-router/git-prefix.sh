#!/usr/bin/env bash

# ==============================================================================
# Git Workflow Utility: Automated Commit Prefixing
# ------------------------------------------------------------------------------
# Standardizes commit messages across development milestones via local Git hooks.
#
# Usage:
#   ./git-prefix.sh set "<prefix>"  # Install hook, set prefix string, and enable
#   ./git-prefix.sh on              # Resume prefixing using existing stored prefix
#   ./git-prefix.sh off             # Temporarily pause prefixing (retains prefix)
#   ./git-prefix.sh status          # Display current activation state and string
#
# Examples:
#   ./git-prefix.sh set "[SER] 01_eda: "
#   ./git-prefix.sh off
#   ./git-prefix.sh on
# ==============================================================================

HOOK_FILE=".git/hooks/prepare-commit-msg"

# Ensure we are inside a Git repository
if [ ! -d ".git" ]; then
    echo "Error: Must be run from the root of a Git repository."
    exit 1
fi

# Ensure hook file exists and is configured
ensure_hook_installed() {
    if [ ! -f "$HOOK_FILE" ]; then
        cat << 'EOF' > "$HOOK_FILE"
#!/bin/sh
COMMIT_MSG_FILE=$1
COMMIT_SOURCE=$2

ENABLED=$(git config --bool --get custom.prefix.enabled 2>/dev/null)
if [ "$ENABLED" != "true" ]; then
    exit 0
fi

PREFIX=$(git config --get custom.prefix.value 2>/dev/null)
if [ -z "$PREFIX" ]; then
    exit 0
fi

if [ -z "$COMMIT_SOURCE" ] \vert{}\vert{} [ "$COMMIT_SOURCE" = "message" ]; then
    if ! grep -qF "$PREFIX" "$COMMIT_MSG_FILE"; then
        sed -i.bak -e "1s|^|$PREFIX\vert{}" "$COMMIT_MSG_FILE"
        rm -f "${COMMIT_MSG_FILE}.bak"
    fi
fi
EOF
        chmod +x "$HOOK_FILE"
    fi
}

case "$1" in
    set)
        if [ -z "$2" ]; then
            echo "Error: Missing prefix text."
            echo "Usage: ./git-prefix.sh set \"[SER] 01_eda: \""
            exit 1
        fi
        ensure_hook_installed
        git config custom.prefix.value "$2"
        git config custom.prefix.enabled true
        echo "Prefix set and enabled: \"$2\""
        ;;
    on|enable)
        ensure_hook_installed
        CURRENT_PREFIX=$(git config --get custom.prefix.value 2>/dev/null)
        if [ -z "$CURRENT_PREFIX" ]; then
            echo "Error: No prefix currently set. Run './git-prefix.sh set \"<prefix>\"' first."
            exit 1