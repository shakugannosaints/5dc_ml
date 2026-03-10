"""Quick API test."""
import urllib.request
import json

# Test status
r = urllib.request.urlopen('http://127.0.0.1:8080/api/status')
print("STATUS:", json.loads(r.read()))

# Test move  
pgn = '[Board "Standard"]\n1. e3'
body = json.dumps({'pgn': pgn, 'temperature': 0.5}).encode()
req = urllib.request.Request(
    'http://127.0.0.1:8080/api/move',
    data=body,
    headers={'Content-Type': 'application/json'}
)
r = urllib.request.urlopen(req)
data = json.loads(r.read())
print("MOVE:", data)
print(f"  Success: {data['success']}")
print(f"  Moves: {data['moves']}")
print(f"  Value: {data['value']}")
