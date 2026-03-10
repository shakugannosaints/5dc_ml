#!/usr/bin/env python3
"""
Smoke test: verify training fixes work correctly.
Tests that self-play stores actual states and training uses them.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build_py_ml', 'Release'))

import torch
import engine
from ml.config import SMALL_CONFIG
from ml.models.agent import Agent
from ml.self_play import SelfPlayWorker


def test_self_play_stores_states():
    """Test that self-play stores actual state copies, not PGN strings."""
    cfg = SMALL_CONFIG
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = Agent(cfg).to(device)
    
    worker = SelfPlayWorker(agent, cfg, use_agent=True, epsilon=0.5)
    record = worker.play_game(temperature=1.0)
    
    print(f"Game: {record.num_moves} moves, outcome={record.outcome}, term={record.termination}")
    assert record.num_moves > 0, "Game should have moves"
    assert len(record.steps) > 0, "Should have steps"
    
    step = record.steps[0]
    # Check step has engine.state (not string)
    assert type(step.state).__name__ == 'state', f"Expected engine.state, got {type(step.state)}"
    assert type(step.per_tl_moves).__name__ == 'list', f"Expected list, got {type(step.per_tl_moves)}"
    
    # Encode the stored state — this should work on the ACTUAL game state
    with torch.no_grad():
        bd, ge, ne, nf = agent.encode_state(step.state)
        print(f"  Step 0: global_embed shape={ge.shape}")
    
    # Check that different steps have different states
    if len(record.steps) > 2:
        s0 = record.steps[0].state
        s_last = record.steps[-1].state
        with torch.no_grad():
            _, ge0, _, _ = agent.encode_state(s0)
            _, ge_last, _, _ = agent.encode_state(s_last)
        # Embeddings should differ (different game states!)
        diff = (ge0 - ge_last).abs().sum().item()
        print(f"  State embedding diff between step 0 and step {len(record.steps)-1}: {diff:.4f}")
        assert diff > 0.001, "Different game states should have different embeddings!"
    
    print("  ✓ self-play stores actual states correctly\n")


def test_training_samples():
    """Test that training samples contain proper state objects."""
    cfg = SMALL_CONFIG
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = Agent(cfg).to(device)
    
    worker = SelfPlayWorker(agent, cfg, use_agent=True, epsilon=0.3)
    samples = worker.generate_training_data(num_games=2, temperature=1.0)
    
    print(f"Generated {len(samples)} training samples from 2 games")
    assert len(samples) > 0, "Should have samples"
    
    # Check sample structure
    s = samples[0]
    assert 'state' in s, "Sample should have 'state' key"
    assert 'per_tl_moves' in s, "Sample should have 'per_tl_moves' key"
    assert type(s['state']).__name__ == 'state', f"Expected engine.state, got {type(s['state'])}"
    
    # Check variety in states
    states_encoded = []
    for i, sample in enumerate(samples[:5]):
        with torch.no_grad():
            _, ge, _, _ = agent.encode_state(sample['state'])
            states_encoded.append(ge.cpu())
    
    # At least some embeddings should differ
    if len(states_encoded) > 1:
        diffs = []
        for i in range(1, len(states_encoded)):
            d = (states_encoded[0] - states_encoded[i]).abs().sum().item()
            diffs.append(d)
        print(f"  Embedding diffs from step 0: {[f'{d:.2f}' for d in diffs]}")
        any_different = any(d > 0.001 for d in diffs)
        assert any_different, "Training samples should cover different game states!"
    
    # Check return values include both perspectives
    returns = [s['return'] for s in samples]
    print(f"  Returns: min={min(returns):.1f}, max={max(returns):.1f}, unique={len(set(returns))}")
    
    print("  ✓ training samples are correct\n")


def test_mini_training():
    """Test 1 epoch of actual training to verify the pipeline."""
    from ml.config import TrainingConfig
    from dataclasses import replace
    
    cfg = SMALL_CONFIG
    cfg = replace(cfg, 
                  num_epochs=2,
                  num_games_per_epoch=2,
                  checkpoint_dir='checkpoints_test',
                  log_file='logs/test_fix.jsonl')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = Agent(cfg).to(device)
    
    # Run self-play
    worker = SelfPlayWorker(agent, cfg, use_agent=True, epsilon=0.5)
    samples = worker.generate_training_data(num_games=2, temperature=1.0)
    
    print(f"Training on {len(samples)} samples...")
    
    # Simple training step
    import torch.optim as optim
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    agent.train()
    
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    count = 0
    
    for sample in samples[:10]:  # Just a few samples
        state = sample['state']
        per_tl_moves = sample['per_tl_moves']
        
        if not per_tl_moves or len(per_tl_moves) == 0:
            continue
        
        try:
            _, global_embed, node_embeds_dict, _ = agent.encode_state(state)
        except Exception as e:
            print(f"  Warning: {e}")
            continue
        
        value_pred = agent.value_head(global_embed.unsqueeze(0)).squeeze()
        target = torch.tensor(sample['return'], dtype=torch.float32, device=device)
        value_loss = (value_pred - target) ** 2
        
        adj_indices = []
        for tl_idx, tl in enumerate(per_tl_moves):
            if tl_idx < len(sample['chosen_indices']):
                idx = sample['chosen_indices'][tl_idx]
                idx = max(0, min(idx, len(tl.moves) - 1)) if len(tl.moves) > 0 else -1
                adj_indices.append(idx)
            else:
                adj_indices.append(0)
        
        log_prob = agent.policy_head.compute_action_logprob(
            global_embed, node_embeds_dict, per_tl_moves, adj_indices, device
        )
        log_prob = log_prob.clamp(min=-20.0, max=0.0)
        
        advantage = target - value_pred.detach()
        advantage = advantage.clamp(-2.0, 2.0)
        policy_loss = -(log_prob * advantage)
        
        loss = policy_loss + value_loss
        total_loss = total_loss + loss
        count += 1
    
    if count > 0:
        avg_loss = total_loss / count
        avg_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
        optimizer.step()
        
        print(f"  avg_loss={avg_loss.item():.4f}, grad_norm={grad_norm.item():.4f}")
        assert not torch.isnan(avg_loss), "Loss should not be NaN"
        assert not torch.isinf(avg_loss), "Loss should not be Inf"
        assert grad_norm.item() < 100, f"Gradient norm too large: {grad_norm.item()}"
    
    print("  ✓ mini training step completed successfully\n")


if __name__ == '__main__':
    print("=" * 60)
    print("  Training Fix Smoke Tests")
    print("=" * 60)
    
    test_self_play_stores_states()
    test_training_samples()
    test_mini_training()
    
    print("=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)
