"""Test transfer learning: small → standard model migration."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build_py_ml'))

from ml.config import SMALL_CONFIG, STANDARD_CONFIG
from ml.models.agent import Agent

# 1. Create a small agent and save it
print('=== Step 1: Create small agent (4x4) ===')
small_agent = Agent(SMALL_CONFIG).to('cuda')
small_params = sum(p.numel() for p in small_agent.parameters())
print(f'  Small agent params: {small_params:,}')
pe = small_agent.policy_head.move_embedder.pos_embed
print(f'  pos_embed shape: {pe.weight.shape}  (4x4 = 16 positions)')

# Save a fake checkpoint
ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         'ml', 'checkpoints', 'test_transfer.pt')
torch.save({
    'epoch': 100,
    'model_state_dict': small_agent.state_dict(),
    'optimizer_state_dict': {},
    'scheduler_state_dict': {},
    'config': SMALL_CONFIG,
    'total_games': 400,
    'total_samples': 10000,
}, ckpt_path)
print(f'  Saved: {ckpt_path}')

# 2. Create a standard agent and load with transfer
print()
print('=== Step 2: Create standard agent (8x8) and transfer ===')
std_agent = Agent(STANDARD_CONFIG).to('cuda')
std_params = sum(p.numel() for p in std_agent.parameters())
print(f'  Standard agent params: {std_params:,}')
pe2 = std_agent.policy_head.move_embedder.pos_embed
print(f'  pos_embed shape: {pe2.weight.shape}  (8x8 = 64 positions)')

# Load with transfer
print()
ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
adapted, skipped, missing = std_agent.load_state_dict_transfer(ckpt['model_state_dict'])
print(f'  Adapted: {len(adapted)} keys')
print(f'  Missing (random init): {len(missing)} keys')

# 3. Verify the pos_embed was correctly expanded
print()
print('=== Step 3: Verify pos_embed expansion ===')
small_pe = small_agent.policy_head.move_embedder.pos_embed.weight.data
std_pe = std_agent.policy_head.move_embedder.pos_embed.weight.data
# Position (0,0) should match: idx 0 in both
diff_00 = (small_pe[0] - std_pe[0]).abs().sum().item()
# Position (1,1) should match: idx 1*4+1=5 in small, 1*8+1=9 in standard
diff_11 = (small_pe[5] - std_pe[9]).abs().sum().item()
# Position (3,3) should match: idx 3*4+3=15 in small, 3*8+3=27 in standard
diff_33 = (small_pe[15] - std_pe[27]).abs().sum().item()
print(f'  pos (0,0) diff: {diff_00:.6f}')
print(f'  pos (1,1) diff: {diff_11:.6f}')
print(f'  pos (3,3) diff: {diff_33:.6f}')
assert diff_00 < 1e-6 and diff_11 < 1e-6 and diff_33 < 1e-6, 'Position embeddings not correctly transferred!'
print('  All transferred positions match!')

# 4. Verify the model can forward pass on standard game
print()
print('=== Step 4: Forward pass on Standard game ===')
import engine
pgn = '[Board "Standard"]\n[Mode "5D"]\n'
state = engine.create_state_from_pgn(pgn)
with torch.no_grad():
    _, ge, ne, nf = std_agent.encode_state(state)
    v = std_agent.value_head(ge.unsqueeze(0)).item()
print(f'  Standard game value prediction: {v:.4f}')
print(f'  Global embed shape: {ge.shape}')
print(f'  Node embeds: {nf.shape}')

# 5. Test that standard agent can play a game with self-play
print()
print('=== Step 5: Self-play on Standard game ===')
from ml.self_play import SelfPlayWorker
worker = SelfPlayWorker(std_agent, STANDARD_CONFIG, use_agent=True, epsilon=0.5)
record = worker.play_game(temperature=1.0)
print(f'  Game: {record.num_moves} moves, outcome={record.outcome}, term={record.termination}')
assert record.num_moves > 0, "Game should have moves"

# Cleanup
os.remove(ckpt_path)
print()
print('=== ALL TRANSFER TESTS PASSED ===')
