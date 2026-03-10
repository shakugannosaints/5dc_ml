# ml/smoke_test.py
"""
Smoke test: verify the full ML pipeline works end-to-end.
Tests each component individually, then runs a minimal training loop.
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def test_engine_import():
    """Test that the engine module loads."""
    print("1. Engine import... ", end="", flush=True)
    import engine
    state = engine.create_state_from_pgn('[Board "Very Small - Open"]\n[Mode "5D"]\n')
    assert state is not None, "Failed to create state"
    print(f"OK (present={state.get_present()}, boards={len(state.get_all_board_tensors())})")
    return state


def test_board_encoder(state):
    """Test BoardEncoder on real board tensors."""
    print("2. BoardEncoder... ", end="", flush=True)
    from ml.models.board_encoder import BoardEncoder
    from ml.config import BoardEncoderConfig

    cfg = BoardEncoderConfig()
    encoder = BoardEncoder(cfg)
    
    board_tensors = state.get_all_board_tensors()
    keys, embeds = encoder.encode_boards(board_tensors, torch.device('cpu'))
    
    assert len(keys) > 0, "No boards encoded"
    assert embeds.shape == (len(keys), cfg.embed_dim), f"Bad shape: {embeds.shape}"
    print(f"OK ({len(keys)} boards → [{embeds.shape[0]}, {embeds.shape[1]}])")
    return encoder, keys, embeds


def test_multiverse_encoder(state, board_encoder):
    """Test MultiverseEncoder with real graph."""
    print("3. MultiverseEncoder... ", end="", flush=True)
    from ml.models.multiverse_encoder import MultiverseEncoder
    from ml.config import MultiverseEncoderConfig

    cfg = MultiverseEncoderConfig()
    gnn = MultiverseEncoder(cfg)
    
    device = torch.device('cpu')
    
    # Get board embeddings
    board_tensors = state.get_all_board_tensors()
    keys, embeds = board_encoder.encode_boards(board_tensors, device)
    board_embeds_dict = {tuple(k): embeds[i] for i, k in enumerate(keys)}
    
    # Get graph structure
    graph = state.get_graph_structure()
    
    # Build GNN inputs
    be, ns, ei, et, nk = gnn.build_graph_from_engine(graph, board_embeds_dict, device)
    
    # Forward
    node_embeds, global_embed = gnn(be, ns, ei, et)
    
    assert node_embeds.shape[0] == len(nk), "Node count mismatch"
    assert global_embed.shape == (1, cfg.global_dim), f"Bad global shape: {global_embed.shape}"
    print(f"OK ({len(nk)} nodes → node[{node_embeds.shape}], global[{global_embed.shape}])")
    return gnn


def test_policy_head(state):
    """Test FactoredPolicyHead with real moves."""
    print("4. PolicyHead... ", end="", flush=True)
    import engine
    from ml.models.policy_head import FactoredPolicyHead
    from ml.config import PolicyConfig

    cfg = PolicyConfig()
    policy = FactoredPolicyHead(cfg, board_size=4)  # Very Small = 4x4
    
    device = torch.device('cpu')
    
    per_tl_moves = engine.get_per_timeline_moves(state)
    assert len(per_tl_moves) > 0, "No per-timeline moves"
    
    # Create dummy embeddings
    global_embed = torch.randn(cfg.state_dim)
    node_embeds_dict = {(0, 0, 0): torch.randn(cfg.board_embed_dim)}
    
    # Sample
    indices, log_prob, entropy = policy.sample_action(
        global_embed, node_embeds_dict, per_tl_moves, 1.0, device
    )
    
    assert len(indices) == len(per_tl_moves), "Wrong number of indices"
    assert log_prob.requires_grad or log_prob.item() <= 0, "Bad log_prob"
    print(f"OK ({len(per_tl_moves)} timelines, log_p={log_prob.item():.3f}, H={entropy.item():.3f})")


def test_value_head():
    """Test ValueHead."""
    print("5. ValueHead... ", end="", flush=True)
    from ml.models.value_head import ValueHead
    from ml.config import ValueConfig

    cfg = ValueConfig()
    vh = ValueHead(cfg)
    
    x = torch.randn(2, cfg.global_dim)
    v = vh(x)
    
    assert v.shape == (2, 1), f"Bad shape: {v.shape}"
    assert (v.abs() <= 1.0).all(), f"Value out of range: {v}"
    print(f"OK (values={v.detach().numpy().flatten()})")


def test_agent(state):
    """Test full Agent pipeline."""
    print("6. Agent (full pipeline)... ", end="", flush=True)
    import engine
    from ml.models.agent import Agent
    from ml.config import SMALL_CONFIG

    agent = Agent(SMALL_CONFIG)
    
    # Encode state
    bed, ge, ned, nef = agent.encode_state(state)
    assert ge.shape[-1] == SMALL_CONFIG.multiverse_encoder.global_dim
    
    # Value estimate
    val = agent.evaluate(state)
    assert -1.0 <= val <= 1.0, f"Bad value: {val}"
    
    # Select action
    action, lp, ent, v = agent.select_action(state, engine, temperature=1.0)
    
    if action is not None:
        print(f"OK (value={val:.3f}, action_moves={len(action.get_moves())}, lp={lp.item():.3f})")
    else:
        print(f"OK (value={val:.3f}, no valid action reconstructed)")


def test_self_play():
    """Test self-play with random policy."""
    print("7. Self-play (random, 1 game)... ", end="", flush=True)
    from ml.self_play import SelfPlayWorker
    from ml.config import SMALL_CONFIG

    worker = SelfPlayWorker(agent=None, cfg=SMALL_CONFIG, use_agent=False)
    record = worker.play_game(temperature=1.0)
    
    print(f"OK ({record.num_moves} moves, outcome={record.outcome:+.0f}, term={record.termination})")


def test_mini_training():
    """Test a minimal training step (1 epoch, 1 game)."""
    print("8. Mini training (1 epoch, 1 game)... ", end="", flush=True)
    from ml.config import TrainingConfig, BoardEncoderConfig, MultiverseEncoderConfig, PolicyConfig, ValueConfig
    
    # Tiny config for smoke test
    cfg = TrainingConfig(
        variant="Very Small - Open",
        variant_pgn='[Board "Very Small - Open"]\n[Mode "5D"]\n',
        board_size_x=4,
        board_size_y=4,
        num_games_per_epoch=1,
        max_game_length=20,
        num_action_samples=16,
        num_epochs=1,
        batch_size=4,
        device="cpu",
        log_file="logs/smoke_test.jsonl",
        checkpoint_dir="checkpoints/smoke_test",
        board_encoder=BoardEncoderConfig(embed_dim=32, num_res_blocks=1, inner_channels=16),
        multiverse_encoder=MultiverseEncoderConfig(
            board_embed_dim=32, gnn_hidden_dim=64, gnn_num_layers=1, gnn_num_heads=2, global_dim=64
        ),
        policy=PolicyConfig(state_dim=64, board_embed_dim=32, hidden_dim=64, move_embed_dim=16),
        value=ValueConfig(global_dim=64, hidden_dim=32),
    )
    
    from ml.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.train()
    print("OK")


def main():
    print("=" * 50)
    print("  5D Chess ML - Smoke Test")
    print("=" * 50)

    try:
        state = test_engine_import()
        encoder, keys, embeds = test_board_encoder(state)
        test_multiverse_encoder(state, encoder)
        test_policy_head(state)
        test_value_head()
        test_agent(state)
        test_self_play()
        test_mini_training()

        print("\n" + "=" * 50)
        print("  ALL TESTS PASSED ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n\nFAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
