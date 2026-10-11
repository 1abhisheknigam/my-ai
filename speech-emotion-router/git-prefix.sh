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

# Locate the actual Git directory regardless of current working directory or submodules
GIT_DIR=$(git rev-parse --git-dir 2>/dev/null)

if [ -z "$GIT_DIR" ]; then
    echo "Error: Must be run inside a Git repository."
    exit 1
fi

HOOK_FILE="$GIT_DIR/hooks/prepare-commit-msg"

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

if [ -z "$COMMIT_SOURCE" ] || [ "$COMMIT_SOURCE" = "message" ]; then
    if ! grep -qF "$PREFIX" "$COMMIT_MSG_FILE"; then
        sed -i.bak -e "1s|^|$PREFIX|" "$COMMIT_MSG_FILE"
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
        fi
        git config custom.prefix.enabled true
        echo "Prefix enabled: \"$CURRENT_PREFIX\""
        ;;
    off|disable)
        git config custom.prefix.enabled false
        echo "Prefix disabled."
        ;;
    status)
        ENABLED=$(git config --bool --get custom.prefix.enabled 2>/dev/null)
        PREFIX=$(git config --get custom.prefix.value 2>/dev/null)
        echo "Enabled: ${ENABLED:-false}"
        echo "Prefix:  ${PREFIX:-(none)}"
        ;;
    *)
        echo "Usage: ./git-prefix.sh {set <prefix>|on|off|status}"
        echo "Examples:"
        echo "  ./git-prefix.sh set \"[SER] 01_eda: \""
        echo "  ./git-prefix.sh off"
        echo "  ./git-prefix.sh on"
        echo "  ./git-prefix.sh status"
        exit 1
        ;;
esac