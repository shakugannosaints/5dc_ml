# alphazero/mcts.py
"""
Semimove-level Monte Carlo Tree Search (MCTS) for 5D Chess AlphaZero.

Each node is a partial-turn state. Actions are legal semimoves plus optional
submit. Player changes only when submit is taken.
"""

import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

from .config import MCTSConfig
from .env import SemimoveEnv, Semimove


SUBMIT_ACTION = "SUBMIT"


@dataclass(slots=True)
class TranspositionEntry:
    """Cached leaf expansion for one semimove-search state."""

    value: float
    child_specs: list[tuple]
    terminal: bool
    terminal_value: float
    action_entries: list[dict]


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = [
        "parent", "action", "prior", "visit_count", "value_sum",
        "children", "is_expanded", "is_terminal", "terminal_value",
        "legal_action_move_lists",
    ]

    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        action=None,
        prior: float = 0.0,
        legal_action_move_lists=None,
    ):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: list[MCTSNode] = []
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0
        self.legal_action_move_lists = legal_action_move_lists

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        parent_visits = self.parent.visit_count if self.parent else 1
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u

    def select_child(self, c_puct: float) -> "MCTSNode":
        best = None
        best_score = -float("inf")
        for child in self.children:
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best = child
        return best

    def expand(self, child_specs: list[tuple], terminal: bool = False, terminal_value: float = 0.0):
        self.is_expanded = True
        self.is_terminal = terminal
        self.terminal_value = terminal_value

        if terminal:
            return

        for action, prior, legal_action_move_lists in child_specs:
            self.children.append(
                MCTSNode(
                    parent=self,
                    action=action,
                    prior=prior,
                    legal_action_move_lists=legal_action_move_lists,
                )
            )


class MCTS:
    """MCTS search engine for semimove-level decisions."""

    def __init__(self, network, cfg: MCTSConfig, device: torch.device):
        self.network = network
        self.cfg = cfg
        self.device = device
        self._transposition_table: dict[tuple, TranspositionEntry] = {}

    def search(self, env: SemimoveEnv, urgency: float = 0.0):
        """
        Returns:
            actions: list of root actions (ordered)
            visits: np.ndarray [A] root visit counts aligned to actions
            root_value: float root node q-value
            action_entries: list of per-action metadata aligned to actions
        """
        self._transposition_table = {}
        root = MCTSNode()

        root_value, root_action_entries = self._expand_node(
            root, env, urgency, capture_action_entries=True
        )

        if len(root.children) > 0:
            noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(root.children))
            eps = self.cfg.dirichlet_epsilon
            for i, child in enumerate(root.children):
                child.prior = (1 - eps) * child.prior + eps * noise[i]

        for _ in range(self.cfg.num_simulations):
            node = root
            scratch_env = self._clone_env(env)

            path = [node]
            while node.is_expanded and (not node.is_terminal) and len(node.children) > 0:
                node = node.select_child(self.cfg.c_puct)
                path.append(node)

                if node.action == SUBMIT_ACTION:
                    outcome = scratch_env.submit_turn(assume_legal=True)
                    if outcome is not None:
                        node.is_terminal = True
                        node.terminal_value = self._white_to_current_player_value(
                            outcome, scratch_env.current_player
                        )
                        break
                else:
                    # Action came from tree expansion legal set; skip re-validation.
                    known_suffixes = None
                    if self.cfg.reuse_semimove_suffixes and scratch_env.uses_strict_legal_enumeration:
                        known_suffixes = node.legal_action_move_lists
                    applied = scratch_env.apply_semimove(
                        node.action,
                        validate=False,
                        known_suffixes=known_suffixes,
                    )
                    if not applied:
                        raise RuntimeError("Expanded semimove failed during MCTS replay")

            if node.is_terminal:
                value = node.terminal_value
            elif not node.is_expanded:
                value, _ = self._expand_node(node, scratch_env, urgency, capture_action_entries=False)
            else:
                value = 0.0

            # Value stored at each node is from that node state's current-player perspective.
            # Player changes only after submit action.
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                if n.action == SUBMIT_ACTION:
                    value = -value

        actions = [child.action for child in root.children]
        visits = np.array([child.visit_count for child in root.children], dtype=np.float64)
        return actions, visits, root.q_value, root_action_entries

    def _find_board_index(self, board_keys: list[tuple], sm: Semimove, player: int) -> int:
        """
        Find source board index for semimove.
        Match by (l,t,c), fallback to (l,t) if color match is missing.
        """
        target_l = sm.from_pos[3]
        target_t = sm.from_pos[2]
        target_c = bool(player)

        for i, (l, t, c) in enumerate(board_keys):
            if l == target_l and t == target_t and c == target_c:
                return i

        for i, (l, t, c) in enumerate(board_keys):
            if l == target_l and t == target_t:
                return i

        return -1

    def _build_action_entries(
        self,
        env: SemimoveEnv,
        board_keys: list[tuple],
        legal_semimoves: list[Semimove],
        can_submit: bool,
    ) -> list[dict]:
        entries = []
        player = env.current_player

        for sm in legal_semimoves:
            fx, fy, ft, fl = sm.from_pos
            tx, ty, tt, tl = sm.to_pos
            entries.append({
                "action": sm,
                "board_idx": self._find_board_index(board_keys, sm, player),
                "from_sq": fx + fy * env.board_side,
                "to_sq": tx + ty * env.board_side,
                "delta_t": tt - ft,
                "delta_l": tl - fl,
                "is_submit": False,
            })

        if can_submit:
            entries.append({
                "action": SUBMIT_ACTION,
                "board_idx": -1,
                "from_sq": 0,
                "to_sq": 0,
                "delta_t": 0,
                "delta_l": 0,
                "is_submit": True,
            })

        return entries

    def _entries_to_tensors(self, entries: list[dict]) -> dict[str, torch.Tensor]:
        if not entries:
            return {
                "board_idx": torch.zeros((0,), dtype=torch.long, device=self.device),
                "from_sq": torch.zeros((0,), dtype=torch.long, device=self.device),
                "to_sq": torch.zeros((0,), dtype=torch.long, device=self.device),
                "delta_t": torch.zeros((0,), dtype=torch.float32, device=self.device),
                "delta_l": torch.zeros((0,), dtype=torch.float32, device=self.device),
                "is_submit": torch.zeros((0,), dtype=torch.bool, device=self.device),
            }

        return {
            "board_idx": torch.tensor([e["board_idx"] for e in entries], dtype=torch.long, device=self.device),
            "from_sq": torch.tensor([e["from_sq"] for e in entries], dtype=torch.long, device=self.device),
            "to_sq": torch.tensor([e["to_sq"] for e in entries], dtype=torch.long, device=self.device),
            "delta_t": torch.tensor([e["delta_t"] for e in entries], dtype=torch.float32, device=self.device),
            "delta_l": torch.tensor([e["delta_l"] for e in entries], dtype=torch.float32, device=self.device),
            "is_submit": torch.tensor([e["is_submit"] for e in entries], dtype=torch.bool, device=self.device),
        }

    @staticmethod
    def _white_to_current_player_value(outcome_white: float, current_player: int) -> float:
        return float(outcome_white if current_player == 0 else -outcome_white)

    def _expand_from_tt(
        self,
        node: MCTSNode,
        entry: TranspositionEntry,
        capture_action_entries: bool,
    ) -> tuple[float, list[dict] | None]:
        node.expand(
            list(entry.child_specs),
            terminal=entry.terminal,
            terminal_value=entry.terminal_value,
        )
        return entry.value, (entry.action_entries if capture_action_entries else None)

    def _expand_node(
        self,
        node: MCTSNode,
        env: SemimoveEnv,
        urgency: float,
        capture_action_entries: bool = False,
    ) -> tuple[float, list[dict] | None]:
        """
        Expand a leaf node using the network.

        Returns:
            value estimate (current-player perspective at this node state),
            and optional action metadata list aligned with node.children order.
        """
        tt_key = None
        if self.cfg.use_transposition_table:
            tt_key = env.get_mcts_transposition_key()
            cached = self._transposition_table.get(tt_key)
            if cached is not None:
                return self._expand_from_tt(node, cached, capture_action_entries)

        if env.done:
            terminal_value = self._white_to_current_player_value(env.outcome or 0.0, env.current_player)
            node.expand([], terminal=True, terminal_value=terminal_value)
            if tt_key is not None:
                self._transposition_table[tt_key] = TranspositionEntry(
                    value=terminal_value,
                    child_specs=[],
                    terminal=True,
                    terminal_value=terminal_value,
                    action_entries=[],
                )
            return terminal_value, ([] if capture_action_entries else None)

        frontier_suffixes = None
        use_suffix_reuse = self.cfg.reuse_semimove_suffixes and env.uses_strict_legal_enumeration
        if use_suffix_reuse:
            frontier_suffixes, can_submit = env.get_legal_frontier_with_suffixes()
            legal_semimoves = [sm for sm, _ in frontier_suffixes]
        else:
            legal_semimoves, can_submit = env.get_legal_frontier()

        if not legal_semimoves and not can_submit:
            node.expand([], terminal=True, terminal_value=0.0)
            if tt_key is not None:
                self._transposition_table[tt_key] = TranspositionEntry(
                    value=0.0,
                    child_specs=[],
                    terminal=True,
                    terminal_value=0.0,
                    action_entries=[],
                )
            return 0.0, ([] if capture_action_entries else None)

        encoded = env.encode_state(urgency=urgency)
        board_keys = encoded["board_keys"]

        action_entries = self._build_action_entries(env, board_keys, legal_semimoves, can_submit)
        action_tensors = self._entries_to_tensors(action_entries)

        board_planes = torch.from_numpy(encoded["board_planes"]).to(self.device)
        last_markers = torch.from_numpy(encoded["last_move_markers"]).to(self.device)
        l_coords = torch.from_numpy(encoded["l_coords"]).to(self.device)
        t_coords = torch.from_numpy(encoded["t_coords"]).to(self.device)
        urg_tensor = torch.tensor([urgency], dtype=torch.float32, device=self.device)

        value, action_logits = self.network.predict_actions(
            board_planes=board_planes,
            last_move_markers=last_markers,
            l_coords=l_coords,
            t_coords=t_coords,
            urgency=urg_tensor,
            action_board_indices=action_tensors["board_idx"],
            action_from_squares=action_tensors["from_sq"],
            action_to_squares=action_tensors["to_sq"],
            action_delta_t=action_tensors["delta_t"],
            action_delta_l=action_tensors["delta_l"],
            action_is_submit=action_tensors["is_submit"],
        )

        if action_logits.numel() == 0:
            node.expand([], terminal=True, terminal_value=0.0)
            if tt_key is not None:
                self._transposition_table[tt_key] = TranspositionEntry(
                    value=0.0,
                    child_specs=[],
                    terminal=True,
                    terminal_value=0.0,
                    action_entries=[],
                )
            return 0.0, ([] if capture_action_entries else None)

        logits_arr = action_logits.detach().cpu().numpy().astype(np.float32)
        logits_arr = np.clip(logits_arr, -20.0, 20.0)
        logits_arr = logits_arr - np.max(logits_arr)
        exp_logits = np.exp(logits_arr)
        priors = exp_logits / (np.sum(exp_logits) + 1e-8)

        child_suffixes = [None] * len(action_entries)
        if frontier_suffixes is not None:
            for i, (_, suffixes) in enumerate(frontier_suffixes):
                child_suffixes[i] = suffixes

        child_specs = [
            (e["action"], float(p), child_suffixes[i])
            for i, (e, p) in enumerate(zip(action_entries, priors))
        ]
        node.expand(child_specs)

        if tt_key is not None:
            self._transposition_table[tt_key] = TranspositionEntry(
                value=float(value),
                child_specs=list(child_specs),
                terminal=False,
                terminal_value=0.0,
                action_entries=list(action_entries),
            )

        return float(value), (action_entries if capture_action_entries else None)

    def _clone_env(self, env: SemimoveEnv) -> SemimoveEnv:
        return env.clone()

    def select_action(self, env: SemimoveEnv, urgency: float = 0.0, temperature: float = 1.0):
        """
        Run MCTS and select an action based on visit counts.

        Returns:
            action: selected action (Semimove or SUBMIT_ACTION)
            policy_probs: [A] probability vector aligned to action_entries
            root_value: root q-value
            action_entries: aligned metadata list for training
        """
        actions, visits, root_value, action_entries = self.search(env, urgency)

        if len(actions) == 0:
            return None, np.zeros((0,), dtype=np.float32), root_value, []

        if temperature < 1e-3:
            idx = int(np.argmax(visits))
            policy_probs = np.zeros_like(visits, dtype=np.float64)
            policy_probs[idx] = 1.0
        else:
            scaled = visits.copy()
            if temperature != 1.0:
                scaled = np.power(scaled, 1.0 / temperature)
            total = scaled.sum()
            if total <= 0:
                policy_probs = np.ones_like(scaled, dtype=np.float64) / len(scaled)
            else:
                policy_probs = scaled / total
            idx = int(np.random.choice(len(actions), p=policy_probs))

        return actions[idx], policy_probs.astype(np.float32), root_value, action_entries
