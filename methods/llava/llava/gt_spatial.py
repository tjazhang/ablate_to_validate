# Shared slot-permutation helpers; the LLaVA (llava/gt_spatial.py) and Qwen (qwenvl/gt_spatial.py) copies are identical.
"""SPATIAL GT-injection operators for the continuous depth span (2026-09-02).

Three eval-side arms live here, all of them variants of the existing ``gt`` arm
(``modeling_qwen2_5_vl.py``'s generate-mode depth block) that keep THIS image's GT
encoder embeddings but change WHICH slot receives WHICH vector, or whether a slot is
given the GT vector at all:

  ``gt_permuted``  every slot ``s`` receives ``GT[perm[s]]`` for ONE fixed permutation
                   shared by every row.  Content is unchanged and its spatial layout is
                   destroyed, so a cell that reads *where* the depth is loses the oracle
                   lift while a cell that reads only the span's global statistics keeps
                   it.  The permutation is a DERANGEMENT and is additionally required to
                   move at least ``PERM_MIN_MOVED`` of the ``grid*grid`` slots by a
                   Chebyshev distance of ``PERM_MIN_CHEBYSHEV`` cells or more, so
                   "permuted" cannot degenerate into "nudged".
  ``gt_marked``    only the MARKED slots (the 8x8 cells under this row's queried point
                   markers, dilated by ``radius`` in Chebyshev cells) receive GT; every
                   other slot keeps the model's OWN vector.
  ``gt_unmarked``  the exact complement.

Everything in this module is PURE (no torch device work beyond indexing the tensors it
is handed, no RNG outside ``build_slot_permutation``'s explicit seed) so the decode-time
branch can be unit-tested on CPU without the backbone.

THE SLOT ORDER IS ROW-MAJOR, AND THAT IS NOT AN ASSUMPTION.  The K=64 span is the
encoder's 16x16 patch grid bilinearly resized to 8x8 by
``methods/llava/model_vqa_depth_continuous.py:interpolate_features_bilinear``, whose
last two operations are ``permute(1, 2, 0)`` then ``.view(target_len, dim)`` on a
``(dim, H, W)`` tensor — i.e. slot index ``= row * W + col``.  ``slot_of`` is that
identity.
"""

from __future__ import annotations

import hashlib

# --- the grid the K=64 continuous span tiles ---------------------------------------
GRID = 8                      # 8x8 = 64 slots
K_DEFAULT = GRID * GRID

# --- the ONE permutation these arms use (pre-registered constants) ------------------
PERM_SEED = 20260902          # numpy default_rng seed; never derived from anything
PERM_MIN_MOVED = 60           # of 64 slots ...
PERM_MIN_CHEBYSHEV = 2        # ... must move by at least this many cells

SPATIAL_MODES = ("gt_permuted", "gt_marked", "gt_unmarked")
MARKED_MODES = ("gt_marked", "gt_unmarked")
MARKED_RADII = (0, 1)

# Per-slot provenance tags.  `own_zero_fallback` and `gt_exhausted` are FAILURES that
# the engagement checkers refuse — they exist so a degenerate slot is named in the
# artifact rather than silently indistinguishable from a legitimate one.
SRC_GT = "gt"
SRC_OWN = "own"
SRC_OWN_ZERO = "own_zero_fallback"
SRC_GT_EXHAUSTED = "gt_exhausted"


# =================================================================================== #
# grid geometry
# =================================================================================== #
def rowcol(slot: int, grid: int = GRID):
    """Slot index -> (row, col) under the row-major flatten (see module docstring)."""
    slot = int(slot)
    if not 0 <= slot < grid * grid:
        raise ValueError(f"slot {slot} out of range for a {grid}x{grid} grid")
    return slot // grid, slot % grid


def slot_of(row: int, col: int, grid: int = GRID) -> int:
    """(row, col) -> slot index.  Row-major: ``row * grid + col``."""
    row, col = int(row), int(col)
    if not (0 <= row < grid and 0 <= col < grid):
        raise ValueError(f"cell ({row},{col}) out of range for a {grid}x{grid} grid")
    return row * grid + col


def chebyshev(a: int, b: int, grid: int = GRID) -> int:
    """Chebyshev (chessboard) distance between two SLOTS on the grid."""
    ra, ca = rowcol(a, grid)
    rb, cb = rowcol(b, grid)
    return max(abs(ra - rb), abs(ca - cb))


def point_to_cell(x_eval, y_eval, png_w, png_h, bbox, img_size, grid: int = GRID):
    """Queried-point pixel (eval frame) -> the 8x8 cell that contains it.

    Two mappings compose, and BOTH are taken from code that already runs:

      1. eval frame -> oracle depth PNG.  Identical arithmetic to
         ``extract_points.sample_oracle`` (``x = bx0 + x/IMG_SIZE * (bx1 - bx0)``): the
         eval image is the scene rendered at ``img_size``, and the 389x389 oracle PNG
         holds that same scene inside a fixed white border whose measured content box is
         ``common.ORACLE_DEPTH_BBOX``.
      2. PNG -> cell.  The encoder is handed the WHOLE PNG (``extract_patch_tokens``
         opens the file and the DINOv2 processor resizes shortest-edge to 224 and
         centre-crops 224 — a square input is resized to exactly 224x224 and the crop is
         a no-op), so the patch grid, and therefore the 8x8 span, tiles the full PNG:
         ``cell = (floor(y_png / H * grid), floor(x_png / W * grid))``.

    Returns ``(row, col)``, clamped into the grid (a marker centre can round onto the
    last pixel, which would otherwise index ``grid``).
    """
    bx0, by0, bx1, by1 = bbox
    x_png = bx0 + (float(x_eval) / float(img_size)) * (bx1 - bx0)
    y_png = by0 + (float(y_eval) / float(img_size)) * (by1 - by0)
    col = int(x_png / float(png_w) * grid)
    row = int(y_png / float(png_h) * grid)
    col = min(grid - 1, max(0, col))
    row = min(grid - 1, max(0, row))
    return row, col


def dilate(cells, radius: int, grid: int = GRID):
    """Chebyshev dilation of ``(row, col)`` cells -> sorted list of SLOT indices.

    ``radius=0`` is the containing cell itself; ``radius=1`` is its 3x3 block, clipped
    at the grid edge (no wrap — a marker near the border simply marks fewer cells).
    """
    if radius not in MARKED_RADII:
        raise ValueError(f"radius {radius!r} must be one of {MARKED_RADII}")
    out = set()
    for row, col in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < grid and 0 <= c < grid:
                    out.add(slot_of(r, c, grid))
    return sorted(out)


# =================================================================================== #
# the permutation
# =================================================================================== #
def permutation_facts(perm, grid: int = GRID) -> dict:
    """Re-derive (never trust) the two properties the permutation must have."""
    perm = [int(x) for x in perm]
    k = len(perm)
    moved = [chebyshev(i, perm[i], grid) for i in range(k)]
    return {
        "k": k,
        "is_permutation": sorted(perm) == list(range(k)),
        "is_derangement": all(perm[i] != i for i in range(k)),
        "n_moved_cheb_ge": sum(1 for d in moved if d >= PERM_MIN_CHEBYSHEV),
        "min_chebyshev": min(moved) if moved else None,
        "max_chebyshev": max(moved) if moved else None,
        "mean_chebyshev": (sum(moved) / len(moved)) if moved else None,
    }


def permutation_sha(perm) -> str:
    """Stable fingerprint of a permutation, for cross-artifact equality checks."""
    return hashlib.sha256(
        ",".join(str(int(x)) for x in perm).encode("utf-8")).hexdigest()


def build_slot_permutation(k: int = K_DEFAULT, seed: int = PERM_SEED,
                           grid: int = GRID, min_moved: int = PERM_MIN_MOVED,
                           min_cheb: int = PERM_MIN_CHEBYSHEV,
                           max_draws: int = 100000):
    """The ONE fixed slot permutation, rejection-sampled from ``seed``.

    Deterministic by construction: ``numpy.random.default_rng(seed)`` is drawn from
    repeatedly and the FIRST draw that is both a derangement and moves at least
    ``min_moved`` slots by Chebyshev ``>= min_cheb`` is returned.  The same seed
    therefore always yields the same tuple, on any machine — which is what lets the
    engagement checker re-derive the permutation and compare it to what the run
    actually injected instead of taking the artifact's word for it.

    Returns a ``tuple`` (immutable — this object is shared by every row).
    """
    import numpy as np

    if k != grid * grid:
        raise ValueError(
            f"k={k} does not tile a {grid}x{grid} grid; the Chebyshev criterion is "
            "only defined on the square span")
    rng = np.random.default_rng(seed)
    for _ in range(max_draws):
        cand = tuple(int(x) for x in rng.permutation(k))
        if any(cand[i] == i for i in range(k)):
            continue
        if sum(1 for i in range(k)
               if chebyshev(i, cand[i], grid) >= min_cheb) < min_moved:
            continue
        return cand
    raise RuntimeError(
        f"no permutation satisfying (derangement, >={min_moved} slots moved by "
        f">={min_cheb}) in {max_draws} draws from seed {seed}")


# =================================================================================== #
# the per-slot decision — the pure core of the decode-time branch
# =================================================================================== #
def uses_gt(slot: int, mode: str, marked=None) -> bool:
    """Does slot ``slot`` get a GT vector under ``mode``?

    ``marked`` is this ROW's marked slot set; it is ignored by ``gt_permuted`` and
    REQUIRED by the two marked modes (a missing set would silently turn ``gt_marked``
    into an identity pass and ``gt_unmarked`` into the plain ``gt`` arm).
    """
    if mode == "gt_permuted":
        return True
    if mode not in MARKED_MODES:
        raise ValueError(f"{mode!r} is not one of {SPATIAL_MODES}")
    if marked is None:
        raise ValueError(
            f"mode {mode!r} needs this row's marked slot set; None would make the arm "
            "silently equal to identity (gt_marked) or to the plain gt arm "
            "(gt_unmarked)")
    inside = int(slot) in marked
    return inside if mode == "gt_marked" else not inside


def gt_source_slot(slot: int, mode: str, perm=None) -> int:
    """Which GT sequence position feeds ``slot`` when the slot is GT-fed.

    ``gt_permuted`` reads ``perm[slot]``; both marked modes read ``slot`` itself.

    A slot BEYOND the permutation's length gets ``-1``, i.e. "no GT position": the
    model reopened its span and is writing past the trained K, and the caller's
    range test then routes it down the same exhausted branch the plain ``gt`` arm
    uses (``modeling_qwen2_5_vl.py``'s ``gt_idx < shape[0]`` fallback).  Crashing a
    372-row eval on a reopened span would be a worse failure than recording it.
    """
    if mode == "gt_permuted":
        if perm is None:
            raise ValueError("gt_permuted needs the slot permutation")
        return int(perm[int(slot)]) if 0 <= int(slot) < len(perm) else -1
    if mode not in MARKED_MODES:
        raise ValueError(f"{mode!r} is not one of {SPATIAL_MODES}")
    return int(slot)


def needs_own_vector(slot: int, mode: str, marked, gt_len: int, perm=None) -> bool:
    """True iff this slot must be filled from the model's OWN depth head.

    Split out of ``resolve_slot`` so the caller can skip the ``_apply_depth_head`` call
    on a GT slot.  The CONTRACT — "``needs_own_vector`` is True exactly when
    ``resolve_slot`` would not return source ``gt``" — must hold per (mode, slot), so
    the two cannot drift apart.
    """
    if not uses_gt(slot, mode, marked):
        return True
    return not 0 <= gt_source_slot(slot, mode, perm) < int(gt_len)


def resolve_slot(slot: int, mode: str, gt_seq, own_vec, marked=None, perm=None):
    """The per-slot operator: ``-> (vector, source, source_slot)``.

    ``gt_seq``   this image's GT embeddings, ``[K_gt, D]`` (the same object the plain
                 ``gt`` arm injects from).
    ``own_vec``  the model's OWN D-space vector for this slot, or ``None`` when the
                 caller could not compute one (no ``prev_hidden``).
    ``marked``   this row's marked slot set (ignored by ``gt_permuted``).
    ``perm``     the fixed slot permutation (required by ``gt_permuted``).

    ``source`` is one of ``gt`` / ``own`` / ``gt_exhausted`` / ``own_zero_fallback``.
    The last two are DEGENERATE and every engagement checker refuses them; they exist so
    that a short GT sequence or a missing hidden state is a named, recorded fact instead
    of a slot that merely looks like the other kind.
    """
    slot = int(slot)
    if gt_seq is None:
        raise ValueError(f"mode {mode!r} requires this image's GT embeddings")
    gt_len = int(gt_seq.shape[0])
    if uses_gt(slot, mode, marked):
        src = gt_source_slot(slot, mode, perm)
        if 0 <= src < gt_len:
            return gt_seq[src], SRC_GT, src
        # the GT sequence is shorter than the span the model is writing
        if own_vec is not None:
            return own_vec, SRC_GT_EXHAUSTED, None
        return gt_seq.new_zeros(gt_seq.shape[-1]), SRC_GT_EXHAUSTED, None
    if own_vec is not None:
        return own_vec, SRC_OWN, None
    return gt_seq.new_zeros(gt_seq.shape[-1]), SRC_OWN_ZERO, None
