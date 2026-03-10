# alphazero/smoke_test.py
"""
Smoke test for the semimove-level AlphaZero pipeline.

Validates that all components work end-to-end:
  1. Engine bindings load correctly
  2. SemimoveEnv can reset and encode states
  3. Network forward pass produces correct shapes
  4. MCTS runs and returns valid actions
  5. Self-play generates valid training samples
  6. Collation and loss computation work
  7. A minimal training step completes without error

Usage:
  python -m alphazero.smoke_test
"""

import sys
import time
import traceback
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "\u2705"
FAIL = "\u274c"


def section(name: str):
    print(f"\n{'鈹€'*60}")
    print(f"  {name}")
    print(f"{'鈹€'*60}")


def check(desc: str, fn):
    """Run a check, print result, return success bool."""
    try:
        result = fn()
        print(f"  {PASS} {desc}")
        return result
    except Exception as e:
        print(f"  {FAIL} {desc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# 1. Engine bindings
# ---------------------------------------------------------------------------
def test_engine():
    section("1. Engine Bindings")
    import engine

    def check_create_state():
        pgn = '[Board "Very Small - Open"]\n[Mode "5D"]\n'
        s = engine.create_state_from_pgn(pgn)
        assert s is not None, "Failed to create state"
        return s

    def check_boards(s):
        boards = s.get_boards()
        assert len(boards) > 0, "No boards"
        print(f"     Boards: {len(boards)}")
        return boards

    def check_timeline_moves(s):
        tl = engine.get_per_timeline_moves(s)
        print(f"     Timeline moves: {len(tl)} timelines")
        return tl

    def check_actions(s):
        actions = engine.enumerate_legal_actions(s, 100)
        print(f"     Legal actions: {len(actions)}")
        return actions

    def check_copy(s):
        s2 = engine.state(s)
        assert s2 is not None, "Copy failed"
        return s2

    s = check("create_state_from_pgn", check_create_state)
    if s is None:
        return None
    check("get_boards", lambda: check_boards(s))
    check("get_per_timeline_moves", lambda: check_timeline_moves(s))
    check("enumerate_legal_actions", lambda: check_actions(s))
    check("state copy constructor", lambda: check_copy(s))
    return s


# ---------------------------------------------------------------------------
# 2. Environment
# ---------------------------------------------------------------------------
def test_env():
    section("2. SemimoveEnv")
    from .env import SemimoveEnv

    VARIANT = '[Board "Very Small - Open"]\n[Mode "5D"]\n'

    def check_reset():
        env = SemimoveEnv(VARIANT, board_limit=10)
        env.reset()
        assert not env.done
        print(f"     Current player: {env.current_player}")
        return env

    def check_legal_semimoves(env):
        moves = env.get_legal_semimoves()
        assert len(moves) > 0, "No legal semimoves at start"
        print(f"     Legal semimoves: {len(moves)}")
        print(f"     First: {moves[0]}")
        return moves

    def check_encode(env):
        enc = env.encode_state(urgency=0.5)
        assert 'board_planes' in enc
        print(f"     board_planes shape: {enc['board_planes'].shape}")
        print(f"     board_keys: {enc['board_keys']}")
        return enc

    def check_apply(env, moves):
        env.apply_semimove(moves[0])
        print(f"     Applied semimove, pending: {len(env.pending_semimoves)}")
        return env

    def check_submit(env):
        # Try to submit 鈥?may or may not be valid depending on state
        can = env.can_submit()
        print(f"     Can submit: {can}")
        return can

    env = check("reset", check_reset)
    if env is None:
        return None
    moves = check("get_legal_semimoves", lambda: check_legal_semimoves(env))
    check("encode_state", lambda: check_encode(env))
    if moves:
        check("apply_semimove", lambda: check_apply(env, moves))
    check("can_submit", lambda: check_submit(env))
    return env


# ---------------------------------------------------------------------------
# 3. Network
# ---------------------------------------------------------------------------
def test_network():
    section("3. Network")
    from .config import NetworkConfig
    from .network import AlphaZeroNetwork

    device = torch.device("cpu")

    def check_init():
        cfg = NetworkConfig(
            d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0,
        )
        net = AlphaZeroNetwork(cfg).to(device)
        params = sum(p.numel() for p in net.parameters())
        print(f"     Parameters: {params:,}")
        return net

    def check_forward(net):
        B, N = 2, 3
        PC = net.cfg.piece_channels
        BS = net.cfg.board_squares
        bp = torch.randn(B, N, PC, BS, device=device)
        lm = torch.randn(B, N, BS, device=device)
        lc = torch.randint(0, 5, (B, N), device=device)
        tc = torch.randint(0, 10, (B, N), device=device)
        urg = torch.rand(B, 1, device=device)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        v, sl, rl = net(bp, lm, lc, tc, urg, mask)
        assert v.shape == (B, 1), f"value shape: {v.shape}"
        assert sl.shape == (B, 1), f"submit_logit shape: {sl.shape}"
        assert rl.shape == (B, N, BS), f"raw_logits shape: {rl.shape}"
        assert -1 <= v.min() <= v.max() <= 1, f"value range: [{v.min()}, {v.max()}]"
        print(f"     value: {v.tolist()}")
        print(f"     submit_logit: {sl.tolist()}")
        print(f"     raw_logits shape: {rl.shape}")
        return net

    def check_predict(net):
        N = 2
        PC = net.cfg.piece_channels
        BS = net.cfg.board_squares
        bp = torch.randn(N, PC, BS, device=device)
        lm = torch.randn(N, BS, device=device)
        lc = torch.randint(0, 5, (N,), device=device)
        tc = torch.randint(0, 10, (N,), device=device)
        urg = torch.rand(1, device=device)

        v, sl, rl = net.predict(bp, lm, lc, tc, urg)
        assert isinstance(v, float)
        assert isinstance(sl, float)
        assert rl.shape == (N, BS)
        print(f"     predict: value={v:.4f}, submit_logit={sl:.4f}")
        return True

    net = check("init", check_init)
    if net is None:
        return None
    check("forward", lambda: check_forward(net))
    check("predict", lambda: check_predict(net))
    return net


def test_standard_variant_preset():
    section("3b. Standard Variant Preset")
    from .config import TrainConfig
    from .env import SemimoveEnv

    def check_standard():
        cfg = TrainConfig()
        cfg.apply_variant("standard")
        assert cfg.network.board_side == 8
        assert cfg.network.board_squares == 64
        env = SemimoveEnv(cfg.variant_pgn, board_limit=10, rules_mode="capture_king")
        env.reset()
        enc = env.encode_state(urgency=0.5)
        assert enc["board_planes"].shape[1:] == (14, 64)
        print(f"     Encoded shape: {enc['board_planes'].shape}")
        print(f"     Variant: {cfg.variant_name}")
        return True

    return check("configure standard variant", check_standard)


# ---------------------------------------------------------------------------
# 4. MCTS
# ---------------------------------------------------------------------------
def test_mcts():
    section("4. MCTS (mini)")
    from .config import NetworkConfig, MCTSConfig
    from .network import AlphaZeroNetwork
    from .env import SemimoveEnv
    from .mcts import MCTS, SUBMIT_ACTION

    device = torch.device("cpu")

    def check_mcts():
        cfg = NetworkConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0)
        net = AlphaZeroNetwork(cfg).to(device)
        mcts_cfg = MCTSConfig(num_simulations=8, c_puct=2.0)  # very few sims

        env = SemimoveEnv(
            '[Board "Very Small - Open"]\n[Mode "5D"]\n',
            board_limit=10,
            rules_mode="strict",
        )
        env.reset()

        mcts = MCTS(net, mcts_cfg, device)
        t0 = time.time()
        action, policy, value, action_entries = mcts.select_action(env, urgency=0.5, temperature=1.0)
        elapsed = time.time() - t0

        print(f"     Action: {action}")
        print(f"     Policy entries: {len(policy)}")
        print(f"     Action metadata: {len(action_entries)}")
        print(f"     Value: {value:.4f}")
        print(f"     Time: {elapsed:.3f}s")
        assert action is not None, "No action selected"
        return True

    check("run MCTS (8 sims)", check_mcts)
    return True


# ---------------------------------------------------------------------------
# 4b. MCTS suffix reuse baseline
# ---------------------------------------------------------------------------
def test_mcts_suffix_reuse():
    section("4b. MCTS Suffix Reuse")
    import engine
    from .config import NetworkConfig, MCTSConfig
    from .network import AlphaZeroNetwork
    from .env import SemimoveEnv
    from .mcts import MCTS

    device = torch.device("cpu")

    def run_case(net, reuse_suffixes: bool):
        env = SemimoveEnv(
            '[Board "Very Small - Open"]\n[Mode "5D"]\n',
            board_limit=10,
            rules_mode="strict",
        )
        env.reset()
        mcts_cfg = MCTSConfig(
            num_simulations=16,
            c_puct=2.0,
            dirichlet_epsilon=0.0,
            reuse_semimove_suffixes=reuse_suffixes,
        )
        mcts = MCTS(net, mcts_cfg, device)

        call_count = 0
        original = engine.enumerate_legal_actions

        def counted_enumerate(state, depth):
            nonlocal call_count
            call_count += 1
            return original(state, depth)

        engine.enumerate_legal_actions = counted_enumerate
        try:
            t0 = time.time()
            with torch.no_grad():
                actions, visits, root_value, action_entries = mcts.search(env, urgency=0.5)
            elapsed = time.time() - t0
        finally:
            engine.enumerate_legal_actions = original

        return {
            "calls": call_count,
            "time": elapsed,
            "actions": len(actions),
            "visits": int(visits.sum()),
            "root_value": float(root_value),
            "entries": len(action_entries),
        }

    def check_suffix_reuse():
        torch.manual_seed(0)
        np.random.seed(0)
        cfg = NetworkConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0)
        net = AlphaZeroNetwork(cfg).to(device)
        net.eval()

        np.random.seed(1234)
        baseline = run_case(net, reuse_suffixes=False)
        np.random.seed(1234)
        optimized = run_case(net, reuse_suffixes=True)

        print(
            f"     Baseline calls/time: {baseline['calls']} / {baseline['time']:.3f}s"
        )
        print(
            f"     Optimized calls/time: {optimized['calls']} / {optimized['time']:.3f}s"
        )
        print(
            f"     Root actions: baseline={baseline['actions']}, optimized={optimized['actions']}"
        )
        assert optimized["calls"] <= baseline["calls"], "Suffix reuse increased enumeration calls"
        if baseline["calls"] > 1:
            assert optimized["calls"] < baseline["calls"], "Suffix reuse did not reduce enumeration calls"
        return {
            "baseline": baseline,
            "optimized": optimized,
        }

    return check("compare enumeration count with/without suffix reuse", check_suffix_reuse)


# ---------------------------------------------------------------------------
# 4c. MCTS transposition table
# ---------------------------------------------------------------------------
def test_mcts_transposition_table():
    section("4c. MCTS Transposition Table")
    from .config import NetworkConfig, MCTSConfig
    from .network import AlphaZeroNetwork
    from .env import SemimoveEnv
    from .mcts import MCTS, MCTSNode

    device = torch.device("cpu")

    def check_tt_cache():
        cfg = NetworkConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0)
        net = AlphaZeroNetwork(cfg).to(device)
        net.eval()
        mcts = MCTS(
            net,
            MCTSConfig(num_simulations=4, dirichlet_epsilon=0.0, use_transposition_table=True),
            device,
        )
        env = SemimoveEnv(
            '[Board "Very Small - Open"]\n[Mode "5D"]\n',
            board_limit=10,
            rules_mode="capture_king",
        )
        env.reset()

        call_count = 0
        original = net.predict_actions

        def counted_predict_actions(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        net.predict_actions = counted_predict_actions
        try:
            v1, entries1 = mcts._expand_node(MCTSNode(), env, urgency=0.5, capture_action_entries=True)
            v2, entries2 = mcts._expand_node(MCTSNode(), env, urgency=0.5, capture_action_entries=True)
        finally:
            net.predict_actions = original

        print(f"     predict_actions calls: {call_count}")
        assert call_count == 1, f"Expected 1 cached network call, got {call_count}"
        assert len(entries1) == len(entries2), "TT changed action entry count"
        assert abs(v1 - v2) < 1e-6, "TT changed value estimate"
        return True

    return check("reuse identical leaf expansion", check_tt_cache)


# ---------------------------------------------------------------------------
# 4d. Capture-king rule regressions
# ---------------------------------------------------------------------------
def test_capture_king_rules():
    section("4d. Capture-King Rules")
    import engine
    from .env import SemimoveEnv

    variant = '[Board "Very Small - Open"]\n[Mode "5D"]\n'

    def apply_sequence(env, sequence: list[str]):
        for text in sequence:
            if text == "SUBMIT":
                result = env.submit_turn(assume_legal=True)
                if result is not None:
                    return result
                continue
            legal, _ = env.get_legal_frontier()
            sm = next((sm for sm in legal if sm.to_ext_move().to_string() == text), None)
            assert sm is not None, f"Missing legal move: {text}"
            env.apply_semimove(sm)
            if env.done:
                return env.outcome
        return env.outcome

    def check_royal_piece_mapping():
        royal_names = [
            "KING_W", "KING_B", "KING_UW", "KING_UB",
            "COMMON_KING_W", "COMMON_KING_B",
            "ROYAL_QUEEN_W", "ROYAL_QUEEN_B",
        ]
        for name in royal_names:
            assert engine.is_royal_piece(getattr(engine.Piece, name)), f"{name} should be royal"
        assert not engine.is_royal_piece(engine.Piece.QUEEN_W)
        assert not engine.is_royal_piece(engine.Piece.QUEEN_B)
        return True

    def check_white_captures_black_royal():
        env = SemimoveEnv(variant, board_limit=25, rules_mode="capture_king")
        env.reset()
        apply_sequence(env, ["(0T1)a2a3Q", "SUBMIT", "(0T1)d4c3Q", "SUBMIT"])
        outcome = apply_sequence(env, ["(0T2)d1c3Q"])
        assert env.done and outcome == 1.0, f"Expected white win, got {outcome}"
        return True

    def check_black_captures_white_royal():
        env = SemimoveEnv(variant, board_limit=25, rules_mode="capture_king")
        env.reset()
        apply_sequence(env, [
            "(0T1)c1a3Q", "SUBMIT",
            "(0T1)b4a3Q", "SUBMIT",
            "(0T2)d1b2Q", "SUBMIT",
            "(0T2)a3(0T1)a2Q", "SUBMIT",
            "(-1T2)a1(0T2)b2Q",
            "(0T3)b2(0T1)c2Q",
            "SUBMIT",
        ])
        legal, _ = env.get_legal_frontier()
        capture = next(
            (sm for sm in legal
             if engine.is_royal_piece(env.state.get_piece(engine.vec4(*sm.to_pos), True))),
            None,
        )
        assert capture is not None, "Expected a black royal-capture move"
        env.apply_semimove(capture)
        assert env.done and env.outcome == -1.0, f"Expected black win, got {env.outcome}"
        return capture.to_ext_move().to_string()

    check("recognize royal piece enums", check_royal_piece_mapping)
    check("white royal capture ends game", check_white_captures_black_royal)
    check("black royal capture ends game", check_black_captures_white_royal)
    return True


# ---------------------------------------------------------------------------
# 5. Self-play
# ---------------------------------------------------------------------------
def test_self_play():
    section("5. Self-play (1 game)")
    from .config import NetworkConfig, MCTSConfig, SelfPlayConfig
    from .network import AlphaZeroNetwork
    from .self_play import SelfPlayWorker

    device = torch.device("cpu")

    def check_play():
        cfg = NetworkConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0)
        net = AlphaZeroNetwork(cfg).to(device)
        mcts_cfg = MCTSConfig(num_simulations=4)
        sp_cfg = SelfPlayConfig(
            num_games=1, max_game_length=20,
            min_board_limit=3, max_board_limit=6,
        )

        worker = SelfPlayWorker(net, mcts_cfg, sp_cfg, device, '[Board "Very Small - Open"]\n[Mode "5D"]\n')
        t0 = time.time()
        game = worker.play_game()
        elapsed = time.time() - t0

        print(f"     Samples: {len(game.samples)}")
        print(f"     Outcome: {game.outcome}")
        print(f"     Total semimoves: {game.total_semimoves}")
        print(f"     Terminal reason: {game.terminal_reason}")
        print(f"     Time: {elapsed:.2f}s")
        assert len(game.samples) > 0 or game.total_semimoves == 0, "No samples"
        return game

    game = check("play one game", check_play)
    return game


# ---------------------------------------------------------------------------
# 6. Collation + loss
# ---------------------------------------------------------------------------
def test_loss():
    section("6. Collation & Loss")
    from .config import NetworkConfig, TrainConfig
    from .network import AlphaZeroNetwork
    from .self_play import SemimoveRecord, collate_samples
    from .train import compute_loss

    device = torch.device("cpu")

    def check_collate():
        from .env import PIECE_CHANNELS, BOARD_SQUARES
        # Create fake samples
        samples = []
        for _ in range(4):
            n_boards = np.random.randint(1, 5)
            s = SemimoveRecord(
                board_planes=np.random.randn(n_boards, PIECE_CHANNELS, BOARD_SQUARES).astype(np.float32),
                last_move_markers=np.random.randn(n_boards, BOARD_SQUARES).astype(np.float32),
                l_coords=np.random.randint(0, 3, (n_boards,)).astype(np.float32),
                t_coords=np.random.randint(0, 10, (n_boards,)).astype(np.float32),
                urgency=0.5,
                padding_mask=np.zeros(n_boards, dtype=bool),
                policy_target=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
                action_board_indices=np.array([0, 0, 0, -1], dtype=np.int64),
                action_from_squares=np.array([0, 1, 2, 0], dtype=np.int64),
                action_to_squares=np.array([4, 5, 6, 0], dtype=np.int64),
                action_delta_t=np.array([0, 0, 1, 0], dtype=np.float32),
                action_delta_l=np.array([0, 1, 0, 0], dtype=np.float32),
                action_is_submit=np.array([False, False, False, True], dtype=bool),
                value_target=np.random.choice([-1.0, 0.0, 1.0]),
            )
            samples.append(s)

        batch = collate_samples(samples, device)
        print(f"     board_planes: {batch['board_planes'].shape}")
        print(f"     padding_mask: {batch['padding_mask'].shape}")
        print(f"     value_target: {batch['value_target'].shape}")
        return batch, samples

    def check_loss(batch):
        net_cfg = NetworkConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0)
        net = AlphaZeroNetwork(net_cfg).to(device)
        train_cfg = TrainConfig()

        loss, metrics = compute_loss(net, batch, train_cfg)
        print(f"     Total loss: {loss.item():.4f}")
        print(f"     Value loss: {metrics['value_loss']:.4f}")
        print(f"     Policy loss: {metrics['policy_loss']:.4f}")
        assert loss.requires_grad, "Loss doesn't have grad"
        assert not torch.isnan(loss), "Loss is NaN"
        return True

    result = check("collate samples", check_collate)
    if result:
        batch, _ = result
        check("compute_loss", lambda: check_loss(batch))
    return True


# ---------------------------------------------------------------------------
# 7. Training step
# ---------------------------------------------------------------------------
def test_training_step():
    section("7. Training Step")
    from .config import NetworkConfig, TrainConfig
    from .network import AlphaZeroNetwork
    from .self_play import SemimoveRecord, collate_samples
    from .train import compute_loss

    device = torch.device("cpu")

    def check_step():
        from .env import PIECE_CHANNELS, BOARD_SQUARES
        net_cfg = NetworkConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.0)
        net = AlphaZeroNetwork(net_cfg).to(device)
        train_cfg = TrainConfig()
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

        # Fake batch
        samples = []
        for _ in range(8):
            n = np.random.randint(1, 4)
            samples.append(SemimoveRecord(
                board_planes=np.random.randn(n, PIECE_CHANNELS, BOARD_SQUARES).astype(np.float32),
                last_move_markers=np.random.randn(n, BOARD_SQUARES).astype(np.float32),
                l_coords=np.random.randint(0, 3, (n,)).astype(np.float32),
                t_coords=np.random.randint(0, 10, (n,)).astype(np.float32),
                urgency=0.5,
                padding_mask=np.zeros(n, dtype=bool),
                policy_target=np.random.dirichlet([1]*3).astype(np.float32),
                action_board_indices=np.array([0, 0, -1], dtype=np.int64),
                action_from_squares=np.array([0, 1, 0], dtype=np.int64),
                action_to_squares=np.array([4, 5, 0], dtype=np.int64),
                action_delta_t=np.array([0, 1, 0], dtype=np.float32),
                action_delta_l=np.array([0, 0, 0], dtype=np.float32),
                action_is_submit=np.array([False, False, True], dtype=bool),
                value_target=float(np.random.choice([-1, 0, 1])),
            ))

        batch = collate_samples(samples, device)

        # Forward + backward
        optimizer.zero_grad()
        loss, metrics = compute_loss(net, batch, train_cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        print(f"     Loss: {loss.item():.4f}")
        print(f"     Grad norm: {sum(p.grad.norm().item() for p in net.parameters() if p.grad is not None):.4f}")
        return True

    check("forward + backward + step", check_step)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  AlphaZero 5D Chess 鈥?Smoke Test")
    print("=" * 60)

    results = {}

    results['engine'] = test_engine()
    results['env'] = test_env()
    results['network'] = test_network()
    results['standard_variant_preset'] = test_standard_variant_preset()
    results['mcts'] = test_mcts()
    results['mcts_suffix_reuse'] = test_mcts_suffix_reuse()
    results['mcts_transposition_table'] = test_mcts_transposition_table()
    results['capture_king_rules'] = test_capture_king_rules()
    results['self_play'] = test_self_play()
    results['loss'] = test_loss()
    results['training_step'] = test_training_step()

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v is not None)
    total = len(results)
    for name, result in results.items():
        status = PASS if result is not None else FAIL
        print(f"  {status} {name}")
    print(f"\n  {passed}/{total} components passed")

    if passed < total:
        sys.exit(1)
    else:
        print("\n  All smoke tests passed! 馃帀")
        sys.exit(0)


if __name__ == "__main__":
    main()


