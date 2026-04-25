#!/usr/bin/env bash
# bootstrap.sh — set up the content-creator development environment
set -euo pipefail

deactivate 2>/dev/null || true   # in case we're already in a venv
conda deactivate 2>/dev/null || true  # in case we're in a conda env


# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
ENVRC_EXAMPLE="${SCRIPT_DIR}/examples/envrc.example"
ENVRC="${SCRIPT_DIR}/.envrc"
TRUFFLEHOG_EXCLUDE="${SCRIPT_DIR}/.trufflehog-exclude.txt"
REQUIRED_DIRS=("output" "work")
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

# ── colour helpers ─────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  RED='\033[0;31m' YELLOW='\033[1;33m' GREEN='\033[0;32m'
  CYAN='\033[0;36m' BOLD='\033[1m' DIM='\033[2m' RESET='\033[0m'
else
  RED='' YELLOW='' GREEN='' CYAN='' BOLD='' DIM='' RESET=''
fi

info()    { printf "${CYAN}ℹ️  %s${RESET}\n"  "$*"; }
success() { printf "${GREEN}✅  %s${RESET}\n" "$*"; }
warn()    { printf "${YELLOW}⚠️  %s${RESET}\n" "$*"; }
error()   { printf "${RED}❌  %s${RESET}\n"  "$*" >&2; }
header()  { printf "\n${BOLD}━━  %s  ━━${RESET}\n" "$*"; }
dim()     { printf "${DIM}    %s${RESET}\n"  "$*"; }

die() { error "$*"; exit 1; }

# ── usage ──────────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Bootstrap the content-creator development environment.

Options:
  -e, --skip-envrc        Skip .envrc setup
  -v, --skip-venv         Skip Python venv creation and package install
  -d, --skip-dirs         Skip required directory creation
  -t, --skip-tests        Skip running pytest
  -s, --skip-trufflehog   Skip TruffleHog secret scan
  -h, --help              Show this help and exit

Examples:
  ./bootstrap.sh                  # full setup
  ./bootstrap.sh --skip-tests     # skip pytest
  ./bootstrap.sh -e -s            # skip envrc + trufflehog
EOF
}

# ── option parsing ─────────────────────────────────────────────────────────────
SKIP_ENVRC=false
SKIP_VENV=false
SKIP_DIRS=false
SKIP_TESTS=false
SKIP_TRUFFLEHOG=false

# GNU getopt returns exit code 4 when enhanced (long-option) mode is available.
_gnu_getopt=''
getopt --test > /dev/null 2>&1 && true   # avoid set -e triggering
if [[ $? -eq 4 ]]; then
  _gnu_getopt="$(command -v getopt)"
elif command -v /opt/homebrew/opt/gnu-getopt/bin/getopt &>/dev/null; then
  _gnu_getopt="/opt/homebrew/opt/gnu-getopt/bin/getopt"
fi

if [[ -n "$_gnu_getopt" ]]; then
  _parsed=$("$_gnu_getopt" \
    --options     evdtsh \
    --longoptions skip-envrc,skip-venv,skip-dirs,skip-tests,skip-trufflehog,help \
    --name        "$(basename "$0")" \
    -- "$@")
  eval set -- "$_parsed"
  while true; do
    case "$1" in
      -e|--skip-envrc)      SKIP_ENVRC=true;      shift ;;
      -v|--skip-venv)       SKIP_VENV=true;       shift ;;
      -d|--skip-dirs)       SKIP_DIRS=true;       shift ;;
      -t|--skip-tests)      SKIP_TESTS=true;      shift ;;
      -s|--skip-trufflehog) SKIP_TRUFFLEHOG=true; shift ;;
      -h|--help)            usage; exit 0 ;;
      --)                   shift; break ;;
      *)                    error "Unknown option: $1"; usage; exit 1 ;;
    esac
  done
else
  # Fall back to POSIX getopts (short flags only) on systems without GNU getopt
  warn "GNU getopt not found; long options unavailable. Install via: brew install gnu-getopt"
  while getopts ":evdtsh" _opt; do
    case "$_opt" in
      e) SKIP_ENVRC=true ;;
      v) SKIP_VENV=true ;;
      d) SKIP_DIRS=true ;;
      t) SKIP_TESTS=true ;;
      s) SKIP_TRUFFLEHOG=true ;;
      h) usage; exit 0 ;;
      \?) error "Unknown option: -${OPTARG}"; usage; exit 1 ;;
    esac
  done
  shift $((OPTIND - 1))
fi

# ── helpers ────────────────────────────────────────────────────────────────────
_require_cmd() {
  command -v "$1" &>/dev/null || die "Required command not found: $1"
}

_python_version_ok() {
  local py="$1"
  local major minor
  major=$("$py" -c 'import sys; print(sys.version_info.major)')
  minor=$("$py" -c 'import sys; print(sys.version_info.minor)')
  [[ "$major" -ge "$PYTHON_MIN_MAJOR" && "$minor" -ge "$PYTHON_MIN_MINOR" ]]
}

_find_python() {
  for candidate in python3 python3.12 python3.11 python3.10 python; do
    if command -v "$candidate" &>/dev/null && _python_version_ok "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  die "Python >= ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR} not found on PATH."
}

# ── 1. .envrc setup ────────────────────────────────────────────────────────────
setup_envrc() {
  header "Environment (.envrc)"

  if [[ -f "$ENVRC" ]]; then
    info ".envrc already exists — skipping copy"
  else
    cp "$ENVRC_EXAMPLE" "$ENVRC"
    success "Copied examples/envrc.example → .envrc"
  fi

  printf "\n${BOLD}Required API key:${RESET}\n"
  dim "HF_TOKEN — your Hugging Face access token"
  dim "  → https://huggingface.co/settings/tokens"
  dim "  → Needs 'Read' scope (or 'Write' if you push models)"

  printf "\n${BOLD}Gated models — you must accept terms before first use:${RESET}\n"
  dim "pyannote/speaker-diarization-3.1  (required for --preserve-speaker)"
  dim "  → https://huggingface.co/pyannote/speaker-diarization-3.1"
  dim "meta-llama/Llama-3.1-8B-Instruct  (default LLM)"
  dim "  → https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"

  printf "\n${BOLD}Optional env overrides (defaults shown):${RESET}\n"
  dim "HF_LLM_MODEL      meta-llama/Llama-3.1-8B-Instruct"
  dim "HF_STT_MODEL      openai/whisper-large-v3"
  dim "HF_TTS_MODEL      hexgrad/Kokoro-82M"
  dim "HF_IMAGE_MODEL    stabilityai/stable-diffusion-xl-base-1.0"
  dim "HF_DIARIZATION_MODEL  pyannote/speaker-diarization-3.1"
  dim "VIDEO_GENERATOR_WORK_DIR  ./output"
  dim "HF_TRANSCRIBE_WORKERS     1"
  dim "HF_IMAGE_WORKERS          1"

  if grep -q 'your_hugging_face_token' "$ENVRC" 2>/dev/null; then
    warn "HF_TOKEN is still the placeholder value — edit .envrc before running the CLI"
  fi
}

# ── 2. Python venv ─────────────────────────────────────────────────────────────
setup_venv() {
  header "Python virtual environment"

  _require_cmd pip3 || true   # informational — python -m pip used below

  local py
  py=$(_find_python)
  info "Using Python: $("$py" --version) ($(command -v "$py"))"

  if [[ -d "$VENV_DIR" ]]; then
    info ".venv already exists — skipping creation"
  else
    "$py" -m venv "$VENV_DIR"
    success "Created .venv"
  fi

  local pip="${VENV_DIR}/bin/pip"
  info "Upgrading pip…"
  "$pip" install --quiet --upgrade pip

  info "Installing package with [test] extras…"
  "$pip" install --quiet -e "${SCRIPT_DIR}[test]"
  success "Installed content-creator[test]"

  info "Installing optional [diarization] extras…"
  if "$pip" install --quiet -e "${SCRIPT_DIR}[diarization]" 2>/dev/null; then
    success "Installed content-creator[diarization]"
  else
    warn "diarization extras failed (may require additional system libraries) — skipping"
  fi
}

# ── 3. Required directories ────────────────────────────────────────────────────
setup_dirs() {
  header "Required directories"
  for dir in "${REQUIRED_DIRS[@]}"; do
    local full_path="${SCRIPT_DIR}/${dir}"
    if [[ -d "$full_path" ]]; then
      dim "${dir}/  already exists"
    else
      mkdir -p "$full_path"
      success "Created ${dir}/"
    fi
  done
}

# ── 4. Tests ───────────────────────────────────────────────────────────────────
run_tests() {
  header "pytest"
  local pytest="${VENV_DIR}/bin/pytest"
  if [[ ! -x "$pytest" ]]; then
    die "pytest not found in venv — run without --skip-venv first"
  fi
  info "Running test suite…"
  "$pytest" -q --tb=short "${SCRIPT_DIR}/tests"
  success "All tests passed"
}

# ── 5. TruffleHog ──────────────────────────────────────────────────────────────
run_trufflehog() {
  header "TruffleHog secret scan"
  if ! command -v trufflehog &>/dev/null; then
    warn "trufflehog not found on PATH — skipping scan"
    warn "Install: brew install trufflesecurity/trufflehog/trufflehog"
    return 0
  fi

  local exclude_args=()
  if [[ -f "$TRUFFLEHOG_EXCLUDE" ]]; then
    exclude_args=(--exclude-paths="$TRUFFLEHOG_EXCLUDE")
  fi

  info "Scanning filesystem (local mode, no update)…"
  trufflehog filesystem "${SCRIPT_DIR}" \
    "${exclude_args[@]}" \
    --no-update \
    --fail
  success "No secrets detected"
}

# ── main ───────────────────────────────────────────────────────────────────────
main() {
  printf "${BOLD}content-creator bootstrap${RESET}  ${DIM}(${SCRIPT_DIR})${RESET}\n"

  [[ "$SKIP_ENVRC"      == false ]] && setup_envrc
  [[ "$SKIP_VENV"       == false ]] && setup_venv
  [[ "$SKIP_DIRS"       == false ]] && setup_dirs
  [[ "$SKIP_TESTS"      == false ]] && run_tests
  [[ "$SKIP_TRUFFLEHOG" == false ]] && run_trufflehog

  printf "\n${GREEN}${BOLD}Bootstrap complete.${RESET}\n"
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    printf "${DIM}Activate the venv:  source .venv/bin/activate${RESET}\n"
  fi
}

main "$@"
