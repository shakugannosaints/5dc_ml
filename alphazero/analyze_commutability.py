from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import alphazero  # noqa: F401 - ensures local engine bindings are on sys.path
import engine

from .env import RULE_CAPTURE_KING, Semimove, SemimoveEnv


ROOT = Path(__file__).resolve().parents[1]
PGN_DIR = ROOT / "test" / "pgn"
OUTPUT_PATH = ROOT / "alphazero" / "logs" / "commutability_audit.md"


@dataclass
class SemimoveAuditRow:
    action_index: int
    action_text: str
    semimove_index: int
    semimove_text: str
    previous_text: str
    commutable_with_previous: str
    previous_target_playable: str
    current_target_playable: str
    destination_hits_other_source: str


MOVE_LINE_RE = re.compile(r"^\s*\d+\.")


def split_pgn_sections(full_pgn: str) -> tuple[str, list[str]]:
    headers: list[str] = []
    action_fragments: list[str] = []
    in_moves = False
    for raw_line in full_pgn.splitlines():
        if not in_moves and MOVE_LINE_RE.match(raw_line):
            in_moves = True
        if in_moves:
            if raw_line.strip():
                line = raw_line.rstrip()
                if " / " in line:
                    left, right = line.split(" / ", 1)
                    action_fragments.append(left.rstrip())
                    action_fragments.append("/ " + right.lstrip())
                else:
                    action_fragments.append(line)
        else:
            if raw_line.strip():
                headers.append(raw_line.rstrip())
    return "\n".join(headers) + "\n", action_fragments


def ext_move_to_semimove(move: engine.ext_move) -> Semimove:
    src = move.get_from()
    dst = move.get_to()
    return Semimove(
        line_idx=src.l(),
        from_pos=(src.x(), src.y(), src.t(), src.l()),
        to_pos=(dst.x(), dst.y(), dst.t(), dst.l()),
    )


def rewind_to_root(game: engine.game) -> None:
    while game.has_parent():
        game.visit_parent()


def walk_mainline_actions(game: engine.game):
    while True:
        children = game.get_child_actions()
        if not children:
            break
        action, text = children[0]
        player = bool(game.get_current_state().get_present()[1])
        moves = [ext_move_to_semimove(move) for move in action.get_moves()]
        move_texts = [move.to_string() for move in action.get_moves()]
        yield player, text, moves, move_texts, len(children)
        game.visit_child(action)


def build_prefix_pgn(variant_pgn: str, action_fragments: list[str], action_index: int) -> str:
    if action_index <= 1:
        return variant_pgn

    prefix_lines = action_fragments[: action_index - 1]
    return variant_pgn.rstrip() + "\n" + "\n".join(prefix_lines) + "\n"


def make_env_at_prefix(variant_pgn: str, action_fragments: list[str], action_index: int) -> SemimoveEnv:
    env = SemimoveEnv(variant_pgn, board_limit=999, rules_mode=RULE_CAPTURE_KING)
    env.game = engine.game.from_pgn(build_prefix_pgn(variant_pgn, action_fragments, action_index))
    env.pending_semimoves = []
    env.last_semimove = None
    env.turn_history = []
    env.done = False
    env.outcome = None
    env.terminal_reason = None
    env.total_semimoves = 0
    env._state_version = 0
    env._legal_action_cache_version = -1
    env._legal_prefix_index_version = -1
    env._frontier_cache_version = -1
    env._frontier_semimoves = []
    env._frontier_keys = set()
    env._frontier_can_submit = False
    return env


def audit_pgn_file(path: Path) -> tuple[list[SemimoveAuditRow], list[str]]:
    full_pgn = path.read_text(encoding="utf-8")
    variant_pgn, action_fragments = split_pgn_sections(full_pgn)

    tree = engine.game.from_pgn(full_pgn)
    rewind_to_root(tree)
    mainline = list(walk_mainline_actions(tree))

    rows: list[SemimoveAuditRow] = []
    notes: list[str] = []

    for action_index, (player_is_black, action_text, moves, move_texts, alternatives) in enumerate(mainline, start=1):
        if alternatives > 1:
            notes.append(f"action {action_index}: mainline chosen from {alternatives} children")

        env = make_env_at_prefix(variant_pgn, action_fragments, action_index)

        previous_sm: Semimove | None = None
        previous_text = "-"
        for semimove_index, (sm, current_text) in enumerate(zip(moves, move_texts), start=1):
            if previous_sm is None:
                row = SemimoveAuditRow(
                    action_index=action_index,
                    action_text=action_text,
                    semimove_index=semimove_index,
                    semimove_text=current_text,
                    previous_text="-",
                    commutable_with_previous="N/A",
                    previous_target_playable="-",
                    current_target_playable=str(env._is_playable_board_coord(sm.to_pos[2], sm.to_pos[3])),
                    destination_hits_other_source="-",
                )
            else:
                row = SemimoveAuditRow(
                    action_index=action_index,
                    action_text=action_text,
                    semimove_index=semimove_index,
                    semimove_text=current_text,
                    previous_text=previous_text,
                    commutable_with_previous=str(env._semimoves_are_commutable(sm, previous_sm)),
                    previous_target_playable=str(env._is_playable_board_coord(previous_sm.to_pos[2], previous_sm.to_pos[3])),
                    current_target_playable=str(env._is_playable_board_coord(sm.to_pos[2], sm.to_pos[3])),
                    destination_hits_other_source=str(env._destination_hits_other_source(sm, previous_sm)),
                )

            rows.append(row)
            if not env.apply_semimove(sm, validate=False):
                notes.append(f"action {action_index}: failed to replay semimove {current_text}")
                break
            previous_sm = sm
            previous_text = current_text
        else:
            env.submit_turn(assume_legal=True)

    return rows, notes


def render_report(results: list[tuple[Path, list[SemimoveAuditRow], list[str]]]) -> str:
    lines: list[str] = ["# Commutability Audit", ""]
    for path, rows, notes in results:
        lines.append(f"## {path.name}")
        lines.append("")
        lines.append(f"- total semimoves audited: {len(rows)}")
        if notes:
            for note in notes:
                lines.append(f"- note: {note}")
        else:
            lines.append("- note: none")
        lines.append("")
        lines.append("| action | action_text | semimove | move | prev | commutable_with_prev | prev_target_playable | current_target_playable | destination_hits_other_source |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row.action_index} | `{row.action_text}` | {row.semimove_index} | `{row.semimove_text}` | "
                f"`{row.previous_text}` | {row.commutable_with_previous} | {row.previous_target_playable} | "
                f"{row.current_target_playable} | {row.destination_hits_other_source} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    pgn_files = sorted(p for p in PGN_DIR.glob("*.5dpgn"))
    results: list[tuple[Path, list[SemimoveAuditRow], list[str]]] = []
    for path in pgn_files:
        rows, notes = audit_pgn_file(path)
        results.append((path, rows, notes))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_report(results), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
