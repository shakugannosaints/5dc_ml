# alphazero/env.py
"""
Semimove-level environment wrapper for the 5D Chess engine.

Decomposes a full action (multi-timeline move combination) into a sequence
of semimoves. Each semimove is one atomic decision:
  - Pick a piece on one timeline and choose its destination, OR
  - Submit the turn (end current player's action).

Enforces lexicographic ordering among commutable semimoves to avoid
duplicate states in the MCTS tree.

Uses engine.game as primary state container (provides apply_move, can_submit,
submit).  engine.state is extracted for board encoding and legal-move queries.
"""

import math
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional

import engine  # resolved via alphazero/__init__.py
from .variants import VERY_SMALL_PROFILE, infer_variant_profile_from_pgn


# 鈹€鈹€鈹€ Constants 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

ENGINE_BOARD_LENGTH = 8   # engine's internal BOARD_LENGTH (always 8)
BOARD_SIDE = VERY_SMALL_PROFILE.board_side
BOARD_SQUARES = VERY_SMALL_PROFILE.board_squares
NUM_PIECE_TYPES = VERY_SMALL_PROFILE.num_piece_types
PIECE_CHANNELS = VERY_SMALL_PROFILE.piece_channels
# Engine channel indices we extract:
#   White K,Q,R,B,N,P = engine ch 0-5   鈫?our ch 0-5
#   Black K,Q,R,B,N,P = engine ch 12-17 鈫?our ch 6-11
#   Unmoved flag      = engine ch 24    鈫?our ch 12
#   Occupied mask     = engine ch 26    鈫?our ch 13
_ENGINE_CH_WHITE = list(range(0, 6))     # [0,1,2,3,4,5]
_ENGINE_CH_BLACK = list(range(12, 18))   # [12,13,14,15,16,17]
_ENGINE_CH_UNMOVED = 24
_ENGINE_CH_OCCUPIED = 26

# For backward compat with imports elsewhere
BOARD_LENGTH = BOARD_SIDE

MoveKey = tuple[int, int, int, int, int, int, int, int]
ActionMoveList = tuple[MoveKey, ...]

RULE_STRICT = "strict"
RULE_CAPTURE_KING = "capture_king"


# 鈹€鈹€鈹€ Semimove representation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

@dataclass
class Semimove:
    """
    One atomic semimove: a piece move on a single timeline.
    """
    line_idx: int          # timeline L
    from_pos: tuple[int, int, int, int]  # (x, y, t, l)
    to_pos: tuple[int, int, int, int]    # (x, y, t, l)

    @property
    def sort_key(self):
        """Key for lexicographic ordering (line_idx, from_xy, to_xy)."""
        return (self.line_idx, self.from_pos, self.to_pos)

    def is_commutable_with(self, other: 'Semimove') -> bool:
        """
        Two semimoves are commutable if they are on different timelines
        AND neither is a departing move whose arrival is on the other's timeline.
        """
        if self.line_idx == other.line_idx:
            return False
        self_local = (self.from_pos[3] == self.to_pos[3])
        other_local = (other.from_pos[3] == other.to_pos[3])
        if self_local and other_local:
            return True
        if self.to_pos[3] == other.line_idx or other.to_pos[3] == self.line_idx:
            return False
        return True

    def to_ext_move(self) -> 'engine.ext_move':
        """Convert back to engine ext_move."""
        return engine.ext_move(
            engine.vec4(*self.from_pos),
            engine.vec4(*self.to_pos),
        )

    def __hash__(self):
        return hash((self.line_idx, self.from_pos, self.to_pos))

    def __eq__(self, other):
        if not isinstance(other, Semimove):
            return False
        return (self.line_idx == other.line_idx and
                self.from_pos == other.from_pos and
                self.to_pos == other.to_pos)


# 鈹€鈹€鈹€ Semimove Environment 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class SemimoveEnv:
    """
    Wraps the engine to provide a semimove-level interface.

    Internal state:
      - self.game: engine.game object (authoritative state)
      - self.pending_semimoves: list of Semimove applied in current partial turn
      - self.last_semimove: previous semimove (for lexicographic filtering)
      - self.turn_history: list of completed turn records [(sm1, sm2, ...), ...]

    At each step, the agent either:
      (a) Picks a semimove from the legal set, or
      (b) Submits the turn.
    """

    def __init__(self, variant_pgn: str, board_limit: int = 999,
                 shared_legal_cache: Optional[OrderedDict] = None,
                 legal_cache_max_entries: int = 20000,
                 rules_mode: str = RULE_CAPTURE_KING):
        self.variant_pgn = variant_pgn
        self.variant = infer_variant_profile_from_pgn(variant_pgn)
        self.board_side = self.variant.board_side
        self.board_squares = self.variant.board_squares
        self.piece_channels = self.variant.piece_channels
        self.l_coord_shift = self.variant.l_coord_shift
        self.board_limit = board_limit
        self.rules_mode = rules_mode
        self.game: Optional[engine.game] = None
        self.pending_semimoves: list[Semimove] = []
        self.last_semimove: Optional[Semimove] = None
        self.turn_history: list[list[Semimove]] = []  # completed turns
        self.done = False
        self.outcome: Optional[float] = None
        self.total_semimoves = 0
        self._state_version = 0
        self._legal_action_cache_version = -1
        self._legal_action_move_lists: list[ActionMoveList] = []
        self._legal_prefix_index_version = -1
        self._legal_prefix_index: dict[MoveKey, list[ActionMoveList]] = {}
        self._shared_legal_cache = shared_legal_cache if shared_legal_cache is not None else OrderedDict()
        self._legal_cache_max_entries = legal_cache_max_entries
        self._frontier_cache_version = -1
        self._frontier_semimoves: list[Semimove] = []
        self._frontier_keys: set[MoveKey] = set()
        self._frontier_can_submit = False

    def reset(self):
        """Reset to initial game state."""
        self.game = engine.game.from_pgn(self.variant_pgn)
        self.pending_semimoves = []
        self.last_semimove = None
        self.turn_history = []
        self.done = False
        self.outcome = None
        self.total_semimoves = 0
        self._state_version = 0
        self._invalidate_legal_cache()

    @property
    def current_player(self) -> int:
        """0=white, 1=black"""
        _, is_black = self.game.get_current_present()
        return 1 if is_black else 0

    @property
    def state(self) -> 'engine.state':
        """Get the current engine state (for encoding/queries)."""
        return self.game.get_current_state()

    @property
    def board_count(self) -> int:
        return len(self.state.get_boards())

    def _invalidate_legal_cache(self):
        self._legal_action_cache_version = -1
        self._legal_action_move_lists = []
        self._legal_prefix_index_version = -1
        self._legal_prefix_index = {}
        self._frontier_cache_version = -1
        self._frontier_semimoves = []
        self._frontier_keys = set()
        self._frontier_can_submit = False

    def _set_legal_action_move_lists(self, move_lists: list[ActionMoveList]):
        self._legal_action_move_lists = list(move_lists)
        self._legal_action_cache_version = self._state_version
        self._legal_prefix_index_version = -1
        self._legal_prefix_index = {}
        self._frontier_cache_version = -1
        self._frontier_semimoves = []
        self._frontier_keys = set()
        self._frontier_can_submit = False

    @property
    def uses_strict_legal_enumeration(self) -> bool:
        return self.rules_mode == RULE_STRICT

    def _state_cache_key(self):
        """
        A hashable state signature for cross-clone legal-action cache.
        """
        present_t, present_c = self.state.get_present()
        return (present_t, bool(present_c), self.state.show_fen())

    def get_mcts_transposition_key(self):
        """
        Hashable semimove-search state key.

        This includes the engine state plus the in-turn semimove prefix, because
        semimove legality and submit availability can depend on the partial-turn
        history even when the board position matches.
        """
        pending = tuple(self._semimove_key(sm) for sm in self.pending_semimoves)
        return (
            self.rules_mode,
            self._state_cache_key(),
            pending,
            self.done,
            self.outcome,
        )

    @staticmethod
    def _ext_move_key(m) -> MoveKey:
        f = m.get_from()
        t = m.get_to()
        return (f.x(), f.y(), f.t(), f.l(), t.x(), t.y(), t.t(), t.l())

    @staticmethod
    def _semimove_key(sm: Semimove) -> MoveKey:
        fx, fy, ft, fl = sm.from_pos
        tx, ty, tt, tl = sm.to_pos
        return (fx, fy, ft, fl, tx, ty, tt, tl)

    @staticmethod
    def _move_key_to_semimove(k: MoveKey) -> Semimove:
        return Semimove(
            line_idx=k[3],
            from_pos=(k[0], k[1], k[2], k[3]),
            to_pos=(k[4], k[5], k[6], k[7]),
        )

    def _get_legal_prefix_index(self) -> dict[MoveKey, list[ActionMoveList]]:
        if self._legal_prefix_index_version == self._state_version:
            return self._legal_prefix_index

        index: dict[MoveKey, list[ActionMoveList]] = {}
        for moves in self._get_legal_action_move_lists():
            if len(moves) == 0:
                continue
            first = moves[0]
            index.setdefault(first, []).append(moves[1:])
        self._legal_prefix_index = index
        self._legal_prefix_index_version = self._state_version
        return index

    def _suffix_actions_for_semimove(self, sm: Semimove) -> list[ActionMoveList]:
        """
        Filter cached legal full actions to those whose next move matches `sm`,
        returning their remaining suffixes after consuming that move.
        """
        sm_key = self._semimove_key(sm)
        suffixes = self._get_legal_prefix_index().get(sm_key)
        if not suffixes:
            return []
        return list(suffixes)

    def _get_legal_action_move_lists(self) -> list[ActionMoveList]:
        """
        Enumerate legal full actions from the current state and cache them.

        This is the authoritative legality source and keeps semimove training
        aligned with engine action legality.
        """
        if self._legal_action_cache_version == self._state_version:
            return self._legal_action_move_lists

        key = self._state_cache_key()
        if key in self._shared_legal_cache:
            move_lists = self._shared_legal_cache.pop(key)
            self._shared_legal_cache[key] = move_lists  # move-to-end (LRU)
        else:
            actions = engine.enumerate_legal_actions(self.state, 0)
            move_lists: list[ActionMoveList] = []
            for a in actions:
                move_keys = tuple(self._ext_move_key(m) for m in a.get_moves())
                move_lists.append(move_keys)
            self._shared_legal_cache[key] = move_lists
            while len(self._shared_legal_cache) > self._legal_cache_max_entries:
                self._shared_legal_cache.popitem(last=False)

        self._legal_action_move_lists = move_lists
        self._legal_action_cache_version = self._state_version
        self._legal_prefix_index_version = -1
        self._legal_prefix_index = {}
        return move_lists

    def _get_capture_king_frontier(self) -> tuple[list[Semimove], bool]:
        if self._frontier_cache_version == self._state_version:
            return self._frontier_semimoves, self._frontier_can_submit

        semimoves = []
        seen_keys: set[MoveKey] = set()
        for line_moves in engine.get_per_timeline_pseudolegal_moves(self.state):
            for from_pos, to_pos in line_moves.moves:
                sm = Semimove(
                    line_idx=line_moves.line_idx,
                    from_pos=(from_pos.x(), from_pos.y(), from_pos.t(), from_pos.l()),
                    to_pos=(to_pos.x(), to_pos.y(), to_pos.t(), to_pos.l()),
                )
                sm_key = self._semimove_key(sm)
                if sm_key in seen_keys:
                    continue
                seen_keys.add(sm_key)
                semimoves.append(sm)

        semimoves = self._apply_lexicographic_filter(semimoves)
        self._frontier_semimoves = semimoves
        self._frontier_keys = {self._semimove_key(sm) for sm in semimoves}
        mandatory, _, _ = self.state.get_timeline_status()
        self._frontier_can_submit = len(mandatory) == 0
        self._frontier_cache_version = self._state_version
        return self._frontier_semimoves, self._frontier_can_submit

    def _is_capture_king_semimove_legal(self, sm: Semimove) -> bool:
        self._get_capture_king_frontier()
        return self._semimove_key(sm) in self._frontier_keys

    def _capture_royal_outcome(self, sm: Semimove) -> Optional[float]:
        mover_is_black = bool(self.current_player)
        piece = self.state.get_piece(engine.vec4(*sm.to_pos), mover_is_black)
        if piece in (engine.Piece.NO_PIECE, engine.Piece.WALL_PIECE):
            return None
        if not engine.is_royal_piece(piece):
            return None
        return -1.0 if mover_is_black else 1.0

    def _apply_lexicographic_filter(self, semimoves: list[Semimove]) -> list[Semimove]:
        if self.last_semimove is None:
            return semimoves
        filtered = []
        for sm in semimoves:
            if sm.is_commutable_with(self.last_semimove):
                if sm.sort_key < self.last_semimove.sort_key:
                    continue
            filtered.append(sm)
        return filtered

    def get_legal_frontier(self) -> tuple[list[Semimove], bool]:
        """
        Return legal next semimoves and whether submit is legal in one pass over
        authoritative legal full-action move-lists.
        """
        if not self.uses_strict_legal_enumeration:
            return self._get_capture_king_frontier()

        semimoves = []
        seen_keys: set[MoveKey] = set()
        can_submit = False

        for moves in self._get_legal_action_move_lists():
            if len(moves) == 0:
                can_submit = True
                continue
            first = moves[0]
            if first in seen_keys:
                continue
            seen_keys.add(first)
            semimoves.append(self._move_key_to_semimove(first))

        return self._apply_lexicographic_filter(semimoves), can_submit

    def get_legal_frontier_with_suffixes(self) -> tuple[list[tuple[Semimove, list[ActionMoveList]]], bool]:
        """
        Return legal next semimoves together with the remaining legal full-action
        suffixes that survive after consuming each semimove.
        """
        if not self.uses_strict_legal_enumeration:
            frontier, can_submit = self.get_legal_frontier()
            return [(sm, []) for sm in frontier], can_submit

        frontier_map = self._get_legal_prefix_index()
        semimoves = [self._move_key_to_semimove(k) for k in frontier_map.keys()]
        filtered_semimoves = self._apply_lexicographic_filter(semimoves)
        can_submit = any(len(moves) == 0 for moves in self._get_legal_action_move_lists())
        frontier = []
        for sm in filtered_semimoves:
            frontier.append((sm, list(frontier_map[self._semimove_key(sm)])))
        return frontier, can_submit

    def get_legal_semimoves(self) -> list[Semimove]:
        """
        Get legal next semimoves for the current partial-turn state.

        A semimove is legal iff it is the first move of at least one legal
        full action from the current state. This guarantees consistency with
        engine-level legality (including check/checkmate constraints).
        """
        semimoves, _ = self.get_legal_frontier()
        return semimoves

    def can_submit(self) -> bool:
        """
        Check if submit is legal from the current state.

        Submit is legal iff there exists a legal full action with 0 remaining
        moves (i.e., submit now).
        """
        if not self.uses_strict_legal_enumeration:
            _, can_submit = self._get_capture_king_frontier()
            return can_submit

        for moves in self._get_legal_action_move_lists():
            if len(moves) == 0:
                return True
        return False

    def apply_semimove(
        self,
        sm: Semimove,
        validate: bool = True,
        known_suffixes: Optional[list[ActionMoveList]] = None,
    ) -> bool:
        """
        Apply a semimove to the current state via game.apply_move().
        Returns True if successful.
        """
        if not self.uses_strict_legal_enumeration:
            if validate and (not self._is_capture_king_semimove_legal(sm)):
                return False
            capture_outcome = self._capture_royal_outcome(sm)
            result = self.game.apply_move_unsafe(sm.to_ext_move())
            if result:
                self.pending_semimoves.append(sm)
                self.last_semimove = sm
                self.total_semimoves += 1
                self._state_version += 1
                self._invalidate_legal_cache()
                if capture_outcome is not None:
                    self.done = True
                    self.outcome = capture_outcome
            return result

        # Keep transitions inside engine-legal action prefixes and derive the
        # next cached legal-action suffixes incrementally.
        suffixes = known_suffixes
        if validate:
            if suffixes is None:
                suffixes = self._suffix_actions_for_semimove(sm)
            if not suffixes:
                return False

        em = sm.to_ext_move()
        result = self.game.apply_move(em)
        if result:
            self.pending_semimoves.append(sm)
            self.last_semimove = sm
            self.total_semimoves += 1
            self._state_version += 1
            if suffixes is not None:
                self._set_legal_action_move_lists(suffixes)
            else:
                self._invalidate_legal_cache()
        return result

    def submit_turn(self, assume_legal: bool = False) -> Optional[float]:
        """
        Submit the accumulated pending moves as a complete action.
        Returns the outcome if the game ends, else None.
        """
        if (not assume_legal) and (not self.can_submit()):
            return None

        submit_fn = self.game.submit if self.uses_strict_legal_enumeration else self.game.submit_unsafe
        if not submit_fn():
            return None
        self.turn_history.append(list(self.pending_semimoves))
        self.pending_semimoves = []
        self.last_semimove = None
        self._state_version += 1
        self._invalidate_legal_cache()
        return self._check_terminal()

    def _check_terminal(self) -> Optional[float]:
        """Check if game has ended. Returns outcome or None."""
        if self.done:
            return self.outcome

        # Board count limit 鈫?forced termination with material scoring
        if self.board_count >= self.board_limit:
            self.done = True
            self.outcome = self._material_score()
            return self.outcome

        if not self.uses_strict_legal_enumeration:
            return None

        # Engine match status
        status = self.game.get_match_status()
        if status == engine.match_status_t.WHITE_WINS:
            self.done = True
            self.outcome = 1.0
            return 1.0
        elif status == engine.match_status_t.BLACK_WINS:
            self.done = True
            self.outcome = -1.0
            return -1.0
        elif status == engine.match_status_t.STALEMATE:
            self.done = True
            self.outcome = 0.0
            return 0.0

        return None

    def _material_score(self) -> float:
        """Material-based score in [-1, 1] for forced termination."""
        w, b = self.state.material_count()
        total = w + b + 1e-8
        diff = w - b
        return math.tanh(diff / total * 3.0)

    # 鈹€鈹€鈹€ Cloning for MCTS 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def clone(self) -> 'SemimoveEnv':
        """
        Create a clone of this environment by replaying history.
        engine.game doesn't support copy, so we recreate from PGN and replay.
        game.from_pgn is ~4碌s, apply_move + submit ~fast, so this is acceptable.
        """
        clone = SemimoveEnv.__new__(SemimoveEnv)
        clone.variant_pgn = self.variant_pgn
        clone.variant = self.variant
        clone.board_side = self.board_side
        clone.board_squares = self.board_squares
        clone.piece_channels = self.piece_channels
        clone.l_coord_shift = self.l_coord_shift
        clone.board_limit = self.board_limit
        clone.rules_mode = self.rules_mode
        clone.done = self.done
        clone.outcome = self.outcome
        clone.total_semimoves = self.total_semimoves
        clone._state_version = self._state_version
        clone._legal_action_cache_version = -1
        clone._legal_action_move_lists = []
        clone._legal_prefix_index_version = -1
        clone._legal_prefix_index = {}
        clone._shared_legal_cache = self._shared_legal_cache
        clone._legal_cache_max_entries = self._legal_cache_max_entries
        clone._frontier_cache_version = -1
        clone._frontier_semimoves = []
        clone._frontier_keys = set()
        clone._frontier_can_submit = False

        clone.game = engine.game.from_pgn(self.variant_pgn)

        # Replay completed turns
        for turn in self.turn_history:
            for sm in turn:
                if self.uses_strict_legal_enumeration:
                    clone.game.apply_move(sm.to_ext_move())
                else:
                    clone.game.apply_move_unsafe(sm.to_ext_move())
            if self.uses_strict_legal_enumeration:
                clone.game.submit()
            else:
                clone.game.submit_unsafe()

        # Replay pending semimoves in current turn
        for sm in self.pending_semimoves:
            if self.uses_strict_legal_enumeration:
                clone.game.apply_move(sm.to_ext_move())
            else:
                clone.game.apply_move_unsafe(sm.to_ext_move())

        clone.turn_history = [list(t) for t in self.turn_history]
        clone.pending_semimoves = list(self.pending_semimoves)
        clone.last_semimove = self.last_semimove
        return clone

    # 鈹€鈹€鈹€ Tensor encoding for network 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    def encode_state(self, urgency: float = 0.0) -> dict:
        """
        Encode current game state into network input tensors.

        Returns dict with:
          'board_planes':      [N, C, S] float32
          'last_move_markers': [N, S] float32
          'l_coords':          [N] int64
          't_coords':          [N] int64
          'urgency':           [1] float32
          'board_keys':        list of (l, t, c) tuples
        """
        s = self.state
        boards_info = s.get_boards()  # list of (l, t, c, fen_str)
        n = len(boards_info)

        board_planes = np.zeros((n, self.piece_channels, self.board_squares), dtype=np.float32)
        l_coords = np.zeros(n, dtype=np.int64)
        t_coords = np.zeros(n, dtype=np.int64)
        last_move_markers = np.zeros((n, self.board_squares), dtype=np.float32)
        board_keys = []

        for i, (l, t, c, fen) in enumerate(boards_info):
            l_coords[i] = l + self.l_coord_shift
            t_coords[i] = t
            board_keys.append((l, t, c))

            # Engine board tensor: [27, 8, 8]
            # Crop to the active board size and flatten to [ch, board_squares].
            b_tensor = s.get_board_tensor(l, t, c)  # [27, 8, 8] numpy
            crop = b_tensor[:, :self.board_side, :self.board_side]

            # White pieces (engine ch 0-5 鈫?our ch 0-5)
            board_planes[i, 0:6, :] = crop[0:6].reshape(6, self.board_squares)
            # Black pieces (engine ch 12-17 鈫?our ch 6-11)
            board_planes[i, 6:12, :] = crop[12:18].reshape(6, self.board_squares)
            # Unmoved flag (engine ch 24 鈫?our ch 12)
            board_planes[i, 12, :] = crop[24].reshape(self.board_squares)
            # Occupied mask (engine ch 26 鈫?our ch 13)
            board_planes[i, 13, :] = crop[26].reshape(self.board_squares)

        # Last-semimove marker
        if self.last_semimove is not None:
            sm = self.last_semimove
            player_color = bool(self.current_player)
            for i, (l, t, c) in enumerate(board_keys):
                if l == sm.from_pos[3] and t == sm.from_pos[2] and c == player_color:
                    pos = sm.from_pos[0] + sm.from_pos[1] * self.board_side
                    if 0 <= pos < self.board_squares:
                        last_move_markers[i, pos] = 1.0
                if l == sm.to_pos[3] and t == sm.to_pos[2] and c == player_color:
                    pos = sm.to_pos[0] + sm.to_pos[1] * self.board_side
                    if 0 <= pos < self.board_squares:
                        last_move_markers[i, pos] = -1.0

        return {
            'board_planes': board_planes,
            'last_move_markers': last_move_markers,
            'l_coords': l_coords,
            't_coords': t_coords,
            'urgency': np.array([urgency], dtype=np.float32),
            'board_keys': board_keys,
        }

    def get_legal_mask(self, board_keys: list, semimoves: list[Semimove]) -> np.ndarray:
        """Build (N, S) mask of legal source squares."""
        n = len(board_keys)
        mask = np.zeros((n, self.board_squares), dtype=np.float32)
        player_color = bool(self.current_player)
        for sm in semimoves:
            for i, (l, t, c) in enumerate(board_keys):
                if l == sm.from_pos[3] and t == sm.from_pos[2] and c == player_color:
                    pos = sm.from_pos[0] + sm.from_pos[1] * self.board_side
                    if 0 <= pos < self.board_squares:
                        mask[i, pos] = 1.0
                    break
        return mask



