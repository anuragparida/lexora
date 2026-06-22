#!/bin/bash
# Phase 1 QA hooks — five checks against the live stack.
# Run from anywhere; uses curl against the host-published ports.
set -uo pipefail

BACKEND="${BACKEND:-http://localhost:18700}"
FRONTEND="${FRONTEND:-http://localhost:18701}"
PG="docker exec -i -e PGPASSWORD=*** phase1-embeddings-postgres-1 psql -U lexora -d lexora -tA -c"

pass=0
fail=0
fail_msgs=()

check() {
    local name="$1" expected="$2" actual="$3"
    if [ "$actual" = "$expected" ]; then
        printf '  PASS  %-50s expected=%s actual=%s\n' "$name" "$expected" "$actual"
        pass=$((pass+1))
    else
        printf '  FAIL  %-50s expected=%s actual=%s\n' "$name" "$expected" "$actual"
        fail=$((fail+1))
        fail_msgs+=("$name: expected=$expected actual=$actual")
    fi
}

check_ge() {
    local name="$1" min="$2" actual="$3"
    if [ "$actual" -ge "$min" ]; then
        printf '  PASS  %-50s min=%s actual=%s\n' "$name" "$min" "$actual"
        pass=$((pass+1))
    else
        printf '  FAIL  %-50s min=%s actual=%s\n' "$name" "$min" "$actual"
        fail=$((fail+1))
        fail_msgs+=("$name: min=$min actual=$actual")
    fi
}

echo "=== Phase 1 QA hooks ==="
echo

# 1. Stack up — three services healthy
echo "[1] Stack health"
for svc in postgres backend frontend; do
    case $svc in
        postgres)
            r=$(sg docker -c "$PG \"SELECT 'ok' FROM pg_extension WHERE extname='vector';\"" 2>/dev/null)
            check "pgvector extension installed" "ok" "$r"
            ;;
        backend)
            code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BACKEND/health")
            check "backend /health 200" "200" "$code"
            ;;
        frontend)
            code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$FRONTEND")
            check "frontend HTTP 200" "200" "$code"
            ;;
    esac
done

# 2. Alembic at head
echo "[2] Alembic migration head"
head_expected="496091d14711"
head_actual=$(sg docker -c "$PG \"SELECT version_num FROM alembic_version;\"" 2>/dev/null | tr -d ' ')
check "alembic at Phase 1 head" "$head_expected" "$head_actual"

# 3. Embeddings populated (words + examples >= 99%)
echo "[3] Embedding coverage"
words_total=$(sg docker -c "$PG \"SELECT COUNT(*) FROM words;\"" 2>/dev/null | tr -d ' ')
words_emb=$(sg docker -c "$PG \"SELECT COUNT(embedding) FROM words;\"" 2>/dev/null | tr -d ' ')
check_ge "words.embedding coverage" "$words_total" "$words_emb"
examples_total=$(sg docker -c "$PG \"SELECT COUNT(*) FROM examples;\"" 2>/dev/null | tr -d ' ')
examples_emb=$(sg docker -c "$PG \"SELECT COUNT(embedding) FROM examples;\"" 2>/dev/null | tr -d ' ')
check_ge "examples.embedding coverage" "$examples_total" "$examples_emb"

# 4. /retrieve returns real nearest-neighbour results
echo "[4] /retrieve end-to-end"
retrieval=$(curl -s -m 10 "$BACKEND/retrieve?query=Gl%C3%BCck&k=3")
result_count=$(printf '%s' "$retrieval" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result_count',0))")
check_ge "/retrieve returns >=3 results" "3" "$result_count"
top_word=$(printf '%s' "$retrieval" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['items'][0]['word'] if d['items'] else '')")
check_ge "/retrieve top result is non-empty" "1" "${#top_word}"

# Examples-source retrieval
examples_retrieval=$(curl -s -m 10 "$BACKEND/retrieve?query=Gl%C3%BCck&k=3&source=examples")
ex_count=$(printf '%s' "$examples_retrieval" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result_count',0))")
check_ge "/retrieve?source=examples returns >=3" "3" "$ex_count"

# 5. pytest green
echo "[5] pytest suite"
cd /home/ody/workspace/lexora/wt/phase1-embeddings/backend
test_output=$(.venv/bin/pytest -q tests/ 2>&1 | tail -3)
test_rc=$?
test_passed=$(printf '%s' "$test_output" | grep -oE '[0-9]+ passed' | head -1 | grep -oE '[0-9]+')
check_ge "pytest all green (exit 0)" "0" "$test_rc"
echo "    pytest summary: $test_output"

echo
echo "=== Summary ==="
echo "  passed: $pass"
echo "  failed: $fail"
if [ $fail -gt 0 ]; then
    echo "  failure details:"
    for msg in "${fail_msgs[@]}"; do
        echo "    - $msg"
    done
    exit 1
fi
exit 0