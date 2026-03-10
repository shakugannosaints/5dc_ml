import sys, re
sys.path.insert(0, 'build_py_ml')
import engine

# Simulate the exact PGN that WASM exports
pgn_raw = '[Board "Very Small - Open"]\n[Mode "5D"]\n[nbrk/3p*/P*3/KRBN:0:1:w]\n\n1. (0T1)Rb1xb4\n'

print("=== RAW PGN (with FEN) ===")
print(repr(pgn_raw))
try:
    s = engine.create_state_from_pgn(pgn_raw)
    print("Result:", s)
except Exception as e:
    print("ERROR:", e)

# Apply the same stripping logic as ai_server.py
print("\n=== CLEANED PGN (FEN stripped) ===")
cleaned_lines = []
for line in pgn_raw.splitlines():
    m = re.match(r'^\[(\w+)\s+"(.*)"\]\s*$', line)
    if m:
        tag_name, tag_value = m.group(1), m.group(2)
        if '/' in tag_value and tag_name not in ('Board', 'Mode', 'Site', 'Event', 'Date', 'White', 'Black', 'Result'):
            print(f"  Stripping FEN header: {line}")
            continue
    if re.match(r'^\[[\w\d\*\./ :]+\]\s*$', line) and '/' in line:
        print(f"  Stripping bare FEN line: {line}")
        continue
    cleaned_lines.append(line)
pgn_clean = '\n'.join(cleaned_lines)
print(repr(pgn_clean))

try:
    s2 = engine.create_state_from_pgn(pgn_clean)
    print("Result:", s2)
    if s2 is not None:
        print("Match status:", s2.get_match_status())
        # Get legal actions to verify state is usable
        actions = engine.get_legal_actions(s2)
        print(f"Legal actions: {len(actions)}")
except Exception as e:
    print("ERROR:", e)
