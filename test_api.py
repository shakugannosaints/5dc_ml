import requests, time

# Wait for server to be up
for i in range(10):
    try:
        r = requests.get('http://127.0.0.1:8080/api/status', timeout=2)
        print(f"Server is up: {r.json()}")
        break
    except:
        print(f"Waiting for server... ({i+1})")
        time.sleep(1)
else:
    print("Server not reachable!")
    exit(1)

# Exact PGN the WASM engine exports - with ACTUAL newlines (not literal \n)
pgn = '[Board "Very Small - Open"]\n[Mode "5D"]\n[nbrk/3p*/P*3/KRBN:0:1:w]\n\n1. (0T1)Rb1xb4\n'

print("\nTest 1: PGN with literal \\n characters (as JSON string)")
print("PGN:", repr(pgn))
r = requests.post('http://127.0.0.1:8080/api/move', json={'pgn': pgn})
print(f"Status: {r.status_code}")
print(f"Response: {r.json()}")

# Now test with actual newlines
pgn2 = """[Board "Very Small - Open"]
[Mode "5D"]
[nbrk/3p*/P*3/KRBN:0:1:w]

1. (0T1)Rb1xb4
"""
print("\nTest 2: PGN with real newlines")
print("PGN:", repr(pgn2))
r2 = requests.post('http://127.0.0.1:8080/api/move', json={'pgn': pgn2})
print(f"Status: {r2.status_code}")
print(f"Response: {r2.json()}")
