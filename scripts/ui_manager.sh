#!/bin/bash
# =============================================================================
# CellViT-Optimus UI Manager
# =============================================================================
# Usage:
#   ./scripts/ui_manager.sh start [cockpit|pathologist|all]
#   ./scripts/ui_manager.sh stop [cockpit|pathologist|all]
#   ./scripts/ui_manager.sh restart [cockpit|pathologist|all]
#   ./scripts/ui_manager.sh status
#   ./scripts/ui_manager.sh logs [cockpit|pathologist]
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$PROJECT_ROOT/.pids"

# Ports
COCKPIT_PORT=7860
PATHOLOGIST_PORT=7861

# Default organ
DEFAULT_ORGAN="Lung"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directories
mkdir -p "$LOG_DIR" "$PID_DIR"

# =============================================================================
# Helper functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

get_pid() {
    local name=$1
    local pid_file="$PID_DIR/${name}.pid"
    if [[ -f "$pid_file" ]]; then
        cat "$pid_file"
    fi
}

is_running() {
    local name=$1
    local pid=$(get_pid "$name")
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

# =============================================================================
# Start functions
# =============================================================================

start_cockpit() {
    local organ="${1:-$DEFAULT_ORGAN}"

    if is_running "cockpit"; then
        log_warn "R&D Cockpit already running (PID: $(get_pid cockpit))"
        return 1
    fi

    log_info "Starting R&D Cockpit on port $COCKPIT_PORT (organ: $organ)..."

    cd "$PROJECT_ROOT"
    nohup python -m src.ui.app \
        --preload \
        --organ "$organ" \
        --port "$COCKPIT_PORT" \
        > "$LOG_DIR/cockpit.log" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_DIR/cockpit.pid"

    # Wait a bit and check if still running
    sleep 2
    if is_running "cockpit"; then
        log_success "R&D Cockpit started (PID: $pid)"
        log_info "  URL: http://localhost:$COCKPIT_PORT"
        log_info "  Logs: $LOG_DIR/cockpit.log"
    else
        log_error "R&D Cockpit failed to start. Check logs: $LOG_DIR/cockpit.log"
        return 1
    fi
}

start_pathologist() {
    local organ="${1:-$DEFAULT_ORGAN}"

    if is_running "pathologist"; then
        log_warn "Pathologist UI already running (PID: $(get_pid pathologist))"
        return 1
    fi

    log_info "Starting Pathologist UI on port $PATHOLOGIST_PORT (organ: $organ)..."

    cd "$PROJECT_ROOT"
    nohup python -m src.ui.app_pathologist \
        --preload \
        --organ "$organ" \
        --port "$PATHOLOGIST_PORT" \
        > "$LOG_DIR/pathologist.log" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_DIR/pathologist.pid"

    # Wait a bit and check if still running
    sleep 2
    if is_running "pathologist"; then
        log_success "Pathologist UI started (PID: $pid)"
        log_info "  URL: http://localhost:$PATHOLOGIST_PORT"
        log_info "  Logs: $LOG_DIR/pathologist.log"
    else
        log_error "Pathologist UI failed to start. Check logs: $LOG_DIR/pathologist.log"
        return 1
    fi
}

# =============================================================================
# Stop functions
# =============================================================================

stop_cockpit() {
    if ! is_running "cockpit"; then
        log_warn "R&D Cockpit is not running"
        rm -f "$PID_DIR/cockpit.pid"
        return 0
    fi

    local pid=$(get_pid "cockpit")
    log_info "Stopping R&D Cockpit (PID: $pid)..."

    kill "$pid" 2>/dev/null || true

    # Wait for graceful shutdown
    local count=0
    while is_running "cockpit" && [[ $count -lt 10 ]]; do
        sleep 1
        count=$((count + 1))
    done

    # Force kill if still running
    if is_running "cockpit"; then
        log_warn "Force killing R&D Cockpit..."
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PID_DIR/cockpit.pid"
    log_success "R&D Cockpit stopped"
}

stop_pathologist() {
    if ! is_running "pathologist"; then
        log_warn "Pathologist UI is not running"
        rm -f "$PID_DIR/pathologist.pid"
        return 0
    fi

    local pid=$(get_pid "pathologist")
    log_info "Stopping Pathologist UI (PID: $pid)..."

    kill "$pid" 2>/dev/null || true

    # Wait for graceful shutdown
    local count=0
    while is_running "pathologist" && [[ $count -lt 10 ]]; do
        sleep 1
        count=$((count + 1))
    done

    # Force kill if still running
    if is_running "pathologist"; then
        log_warn "Force killing Pathologist UI..."
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PID_DIR/pathologist.pid"
    log_success "Pathologist UI stopped"
}

# =============================================================================
# Status function
# =============================================================================

show_status() {
    echo ""
    echo "=== CellViT-Optimus UI Status ==="
    echo ""

    # Cockpit
    if is_running "cockpit"; then
        local pid=$(get_pid "cockpit")
        echo -e "R&D Cockpit:     ${GREEN}RUNNING${NC} (PID: $pid)"
        echo "  URL: http://localhost:$COCKPIT_PORT"
    else
        echo -e "R&D Cockpit:     ${RED}STOPPED${NC}"
    fi

    # Pathologist
    if is_running "pathologist"; then
        local pid=$(get_pid "pathologist")
        echo -e "Pathologist UI:  ${GREEN}RUNNING${NC} (PID: $pid)"
        echo "  URL: http://localhost:$PATHOLOGIST_PORT"
    else
        echo -e "Pathologist UI:  ${RED}STOPPED${NC}"
    fi

    echo ""
}

# =============================================================================
# Logs function
# =============================================================================

show_logs() {
    local name=$1
    local log_file="$LOG_DIR/${name}.log"

    if [[ ! -f "$log_file" ]]; then
        log_error "Log file not found: $log_file"
        return 1
    fi

    log_info "Showing last 50 lines of $log_file (Ctrl+C to exit)..."
    tail -f -n 50 "$log_file"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 <command> [target] [options]"
    echo ""
    echo "Commands:"
    echo "  start [cockpit|pathologist|all]   Start UI(s)"
    echo "  stop [cockpit|pathologist|all]    Stop UI(s)"
    echo "  restart [cockpit|pathologist|all] Restart UI(s)"
    echo "  status                            Show status"
    echo "  logs [cockpit|pathologist]        Show logs (tail -f)"
    echo ""
    echo "Options:"
    echo "  --organ ORGAN   Set initial organ (default: Lung)"
    echo ""
    echo "Examples:"
    echo "  $0 start all                    # Start both UIs"
    echo "  $0 start cockpit --organ Breast # Start cockpit with Breast model"
    echo "  $0 stop all                     # Stop all UIs"
    echo "  $0 restart pathologist          # Restart pathologist UI"
    echo "  $0 status                       # Show status"
    echo "  $0 logs cockpit                 # Follow cockpit logs"
}

# Parse arguments
COMMAND="${1:-}"
TARGET="${2:-all}"
ORGAN="$DEFAULT_ORGAN"

# Parse options
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --organ)
            ORGAN="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

case "$COMMAND" in
    start)
        case "$TARGET" in
            cockpit)
                start_cockpit "$ORGAN"
                ;;
            pathologist)
                start_pathologist "$ORGAN"
                ;;
            all)
                start_cockpit "$ORGAN"
                start_pathologist "$ORGAN"
                ;;
            *)
                log_error "Unknown target: $TARGET"
                usage
                exit 1
                ;;
        esac
        ;;

    stop)
        case "$TARGET" in
            cockpit)
                stop_cockpit
                ;;
            pathologist)
                stop_pathologist
                ;;
            all)
                stop_cockpit
                stop_pathologist
                ;;
            *)
                log_error "Unknown target: $TARGET"
                usage
                exit 1
                ;;
        esac
        ;;

    restart)
        case "$TARGET" in
            cockpit)
                stop_cockpit
                sleep 1
                start_cockpit "$ORGAN"
                ;;
            pathologist)
                stop_pathologist
                sleep 1
                start_pathologist "$ORGAN"
                ;;
            all)
                stop_cockpit
                stop_pathologist
                sleep 1
                start_cockpit "$ORGAN"
                start_pathologist "$ORGAN"
                ;;
            *)
                log_error "Unknown target: $TARGET"
                usage
                exit 1
                ;;
        esac
        ;;

    status)
        show_status
        ;;

    logs)
        if [[ "$TARGET" == "all" ]]; then
            log_error "Please specify cockpit or pathologist"
            exit 1
        fi
        show_logs "$TARGET"
        ;;

    *)
        usage
        exit 1
        ;;
esac
