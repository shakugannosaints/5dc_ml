"""Quick test of the AI server components."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.ai_server import AIPlayer

print("=== Testing AIPlayer ===")
player = AIPlayer(checkpoint_dir='ml/checkpoints', device='cpu')
print("Model info:", player.model_info)

# Test with a standard opening
pgn = '[Board "Standard"]\n1. e3'
print(f"\nTesting with PGN: {pgn!r}")
result = player.get_ai_move(pgn, temperature=0.5)
print("Result:", result)

# Test with a fresh game (white to move)
pgn2 = '[Board "Standard"]'
print(f"\nTesting with PGN: {pgn2!r}")
result2 = player.get_ai_move(pgn2, temperature=0.5)
print("Result:", result2)

print("\n=== All tests passed! ===")
