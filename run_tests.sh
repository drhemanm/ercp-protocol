#!/bin/bash
#
# Test runner script for ERCP Protocol
#
# Usage:
#   ./run_tests.sh [options]
#
# Options:
#   --unit          Run only unit tests
#   --integration   Run only integration tests
#   --golden        Run only golden tests
#   --coverage      Run with coverage reporting (default)
#   --no-coverage   Run without coverage
#   --verbose       Verbose output
#   --failfast      Stop on first failure
#   --markers       Run specific pytest markers (e.g., --markers asyncio)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
RUN_COVERAGE=true
VERBOSE=""
FAILFAST=""
TEST_PATH="tests/"
MARKERS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_PATH="tests/unit/"
            shift
            ;;
        --integration)
            TEST_PATH="tests/integration/"
            shift
            ;;
        --golden)
            TEST_PATH="tests/golden/"
            shift
            ;;
        --coverage)
            RUN_COVERAGE=true
            shift
            ;;
        --no-coverage)
            RUN_COVERAGE=false
            shift
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --failfast|-x)
            FAILFAST="-x"
            shift
            ;;
        --markers|-m)
            MARKERS="-m $2"
            shift 2
            ;;
        --help|-h)
            echo "ERCP Protocol Test Runner"
            echo ""
            echo "Usage: ./run_tests.sh [options]"
            echo ""
            echo "Options:"
            echo "  --unit          Run only unit tests"
            echo "  --integration   Run only integration tests"
            echo "  --golden        Run only golden tests"
            echo "  --coverage      Run with coverage reporting (default)"
            echo "  --no-coverage   Run without coverage"
            echo "  --verbose, -v   Verbose output"
            echo "  --failfast, -x  Stop on first failure"
            echo "  --markers, -m   Run specific pytest markers"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                    # Run all tests with coverage"
            echo "  ./run_tests.sh --unit             # Run only unit tests"
            echo "  ./run_tests.sh --golden -v        # Run golden tests verbosely"
            echo "  ./run_tests.sh --no-coverage -x   # Run without coverage, stop on first fail"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ERCP Protocol Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}Warning: No virtual environment detected${NC}"
    echo -e "${YELLOW}Consider activating your venv first${NC}"
    echo ""
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install it with: pip install pytest pytest-asyncio pytest-cov"
    exit 1
fi

# Set up environment variables for testing
export TESTING=true
export DATABASE_URL="${DATABASE_URL:-sqlite+aiosqlite:///:memory:}"
export JWT_SECRET_KEY="${JWT_SECRET_KEY:-test-secret-key-for-testing-only}"
export API_KEYS="${API_KEYS:-test-api-key-1,test-api-key-2}"

echo -e "${GREEN}Test Configuration:${NC}"
echo "  Test Path: $TEST_PATH"
echo "  Coverage: $RUN_COVERAGE"
echo "  Verbose: ${VERBOSE:-false}"
echo "  Fail Fast: ${FAILFAST:-false}"
echo "  Database: $DATABASE_URL"
echo ""

# Build pytest command
PYTEST_CMD="pytest $TEST_PATH"

# Add coverage if enabled
if [ "$RUN_COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=server --cov-report=term-missing --cov-report=html"
fi

# Add verbose flag
if [ -n "$VERBOSE" ]; then
    PYTEST_CMD="$PYTEST_CMD $VERBOSE"
fi

# Add failfast flag
if [ -n "$FAILFAST" ]; then
    PYTEST_CMD="$PYTEST_CMD $FAILFAST"
fi

# Add markers if specified
if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

# Add additional pytest options
PYTEST_CMD="$PYTEST_CMD --tb=short --strict-markers"

# Run the tests
echo -e "${GREEN}Running tests...${NC}"
echo -e "${BLUE}Command: $PYTEST_CMD${NC}"
echo ""

if $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  All tests passed! ✓${NC}"
    echo -e "${GREEN}========================================${NC}"

    if [ "$RUN_COVERAGE" = true ]; then
        echo ""
        echo -e "${BLUE}Coverage report saved to: htmlcov/index.html${NC}"
        echo -e "${BLUE}Open it with: open htmlcov/index.html (macOS) or xdg-open htmlcov/index.html (Linux)${NC}"
    fi

    exit 0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Tests failed! ✗${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
