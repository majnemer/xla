#!/bin/sh

set -u

cd "$(dirname "$0")/.."

CXX=${CXX:-c++}
CXXFLAGS=${CXXFLAGS:-}
COMMON_FLAGS="-std=c++17 -Wall -Wextra -I."

pass() {
    file=$1
    echo "PASS-COMPILE $file"
    # shellcheck disable=SC2086
    "$CXX" $CXXFLAGS $COMMON_FLAGS -fsyntax-only "$file"
}

fail() {
    file=$1
    expected=$2
    log=$(mktemp "${TMPDIR:-/tmp}/objc_bridge_compile.XXXXXX") || return 1

    echo "FAIL-COMPILE $file"
    # shellcheck disable=SC2086
    if "$CXX" $CXXFLAGS $COMMON_FLAGS -fsyntax-only "$file" >"$log" 2>&1; then
        echo "compile_tests: expected failure, but compilation succeeded: $file" >&2
        cat "$log" >&2
        rm -f "$log"
        return 1
    fi

    if ! grep -F "$expected" "$log" >/dev/null; then
        echo "compile_tests: expected diagnostic not found for $file" >&2
        echo "expected substring: $expected" >&2
        cat "$log" >&2
        rm -f "$log"
        return 1
    fi

    rm -f "$log"
    return 0
}

rc=0

pass tests/compile_pass.cpp || rc=1

fail tests/fail_marker_incomplete.cpp "objc::retained<T>" || rc=1
fail tests/fail_marker_non_handle.cpp "objc::retained<T>" || rc=1
fail tests/fail_marker_block.cpp "objc::retained<T>" || rc=1
fail tests/fail_method_facade_return.cpp "objc::method return type" || rc=1
fail tests/fail_method_object_return.cpp "objc::method return type" || rc=1
fail tests/fail_method_ownership_non_handle_return.cpp \
    "objc::retained<T>" || rc=1
fail tests/fail_method_object_param.cpp "objc::method parameters" || rc=1
fail tests/fail_method_block_non_block_arg.cpp "objc::method block parameters" || rc=1
fail tests/fail_method_block_signature_mismatch.cpp \
    "objc::method block parameters" || rc=1
fail tests/fail_method_out_return.cpp \
    "objc::method return type must not be objc::out" || rc=1
fail tests/fail_method_out_bad_param.cpp "objc::method parameters" || rc=1
fail tests/fail_method_out_non_handle_param.cpp "objc::retained<T>" || rc=1
fail tests/fail_send_ownership_marker.cpp "objc::send<retained<T>" || rc=1
fail tests/fail_send_out_marker.cpp "objc::send<objc::out<...>>" || rc=1
fail tests/fail_block_facade_return.cpp "objc::block return type" || rc=1
fail tests/fail_block_object_return.cpp "objc::block return type" || rc=1
fail tests/fail_block_ownership_non_handle_return.cpp \
    "objc::retained<T>" || rc=1
fail tests/fail_block_object_param.cpp "objc::block parameters" || rc=1
fail tests/fail_block_out_return.cpp \
    "objc::block return type must not be objc::out" || rc=1
fail tests/fail_block_out_param.cpp "objc::block parameters" || rc=1
fail tests/fail_block_not_retained_owner_return.cpp \
    "objc::block<not_retained<T>()> callbacks must return a borrowed facade" || rc=1

exit "$rc"
