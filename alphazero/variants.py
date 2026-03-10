"""Variant presets for semimove AlphaZero training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantProfile:
    name: str
    board_label: str
    variant_pgn: str
    board_side: int
    board_squares: int
    num_piece_types: int = 6
    piece_channels: int = 14
    l_coord_shift: int = 5


VERY_SMALL_PROFILE = VariantProfile(
    name="very_small",
    board_label="Very Small - Open",
    variant_pgn='[Board "Very Small - Open"]\n[Mode "5D"]\n',
    board_side=4,
    board_squares=16,
)

STANDARD_PROFILE = VariantProfile(
    name="standard",
    board_label="Standard",
    variant_pgn='[Board "Standard"]\n[Mode "5D"]\n',
    board_side=8,
    board_squares=64,
)

STANDARD_TURN_ZERO_PROFILE = VariantProfile(
    name="standard_turn_zero",
    board_label="Standard - Turn Zero",
    variant_pgn='[Board "Standard - Turn Zero"]\n[Mode "5D"]\n',
    board_side=8,
    board_squares=64,
)

VARIANT_PROFILES = {
    VERY_SMALL_PROFILE.name: VERY_SMALL_PROFILE,
    STANDARD_PROFILE.name: STANDARD_PROFILE,
    STANDARD_TURN_ZERO_PROFILE.name: STANDARD_TURN_ZERO_PROFILE,
}


def get_variant_profile(name: str) -> VariantProfile:
    try:
        return VARIANT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown variant profile: {name}") from exc


def infer_variant_profile_from_pgn(variant_pgn: str) -> VariantProfile:
    if 'Board "Standard - Turn Zero"' in variant_pgn:
        return STANDARD_TURN_ZERO_PROFILE
    if 'Board "Standard"' in variant_pgn:
        return STANDARD_PROFILE
    if 'Board "Very Small - Open"' in variant_pgn:
        return VERY_SMALL_PROFILE
    raise ValueError(f"Unsupported variant PGN for AlphaZero profile inference: {variant_pgn!r}")
