"""Exploration map — fog of war unlocked by completed lesson sections + XP."""

from __future__ import annotations

from functools import lru_cache
from collections import OrderedDict
import threading
import hashlib
import heapq
import json
import math
from pathlib import Path

from database import MapSnapshot, CourseOutline, MapTileProvenance, SectionRewardClaim, SessionLocal, UserMapState

# Must match mapTerrainTypes.js TERRAIN enum (0–14).
T_OCEAN = 1
T_SHALLOW = 2
T_REEF = 3
T_BEACH = 4
T_DEEP_FOREST = 8
T_MOUNTAIN = 9
T_PEAK = 10
T_SWAMP = 11
T_LAVA = 12
T_PATH = 14

# XP awarded on section / lesson completion (persisted on UserMapState).
XP_PER_SECTION = 100
XP_LESSON_COMPLETE_BONUS = 500
# Extra map unlock budget beyond section estimated minutes.
BONUS_UNLOCK_PER_SECTION = 35
BONUS_UNLOCK_LESSON_COMPLETE = 120

XP_PER_LEVEL = 400


def _is_land_terrain(t: int) -> bool:
    return T_BEACH <= t <= T_PATH


def _terrain_move_cost_type(t: int) -> float:
    if _is_land_terrain(t):
        if t in (T_MOUNTAIN, T_PEAK, T_DEEP_FOREST):
            return 0.58
        if t == T_LAVA:
            return 0.85
        if t == T_SWAMP:
            return 0.55
        return 0.48
    if t in (T_SHALLOW, T_REEF):
        return 0.72
    if t == T_OCEAN:
        return 1.05
    return 1.35


def _cell_discovery_hash(x: int, y: int) -> float:
    n = (x * 374761393 + y * 668265263) & 0xFFFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0


# ---------------------------------------------------------------------------
# Map levels. Each world is exported from the frontend generator
# (scripts/export-map-terrain.mjs) so the unlock order here matches the art:
#   level 1  The Lumen Reaches  map_terrain_types.json
#   level 2  Neon Meridian      map_terrain_types_l2.json
# Points chart level 1; once every tile of it is charted, further points
# chart level 2. Charted tiles grow linearly with points ("area" pacing), so
# every section uncovers about the same amount of land.
# ---------------------------------------------------------------------------

LEVEL_FILES = {1: "map_terrain_types.json", 2: "map_terrain_types_l2.json"}
# Unlock points to chart a whole level. A mastered four-section lesson banks
# about 4×(35+25) + 120 = 360, so ~12 lessons for level 1 and ~13 for level 2.
LEVEL_POINTS = {1: 4200, 2: 4800}
# Radius charted around a level's harbour before any study.
LEVEL_CLEAR = 9


class MapLevel:
    def __init__(self, level: int, world: str, size: int, origin: tuple[int, int], types: list[int] | None,
                 clear: float, full_points: int, chests: list[dict]):
        self.level = level
        self.world = world
        self.size = size
        self.ox, self.oy = origin
        self.types = types or [T_OCEAN] * (size * size)
        self.clear = float(clear)
        self.full_points = full_points
        self.pacing = "area"
        self.chests = chests
        self.full_radius = math.ceil(math.hypot(max(self.ox, size - self.ox), max(self.oy, size - self.oy))) + 2

    def terrain_at(self, x: int, y: int) -> int:
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            return 0
        return self.types[y * self.size + x]

    def pacing_payload(self) -> dict:
        return {"level": self.level, "world": self.world, "mode": self.pacing, "clear": self.clear,
                "full": self.full_radius, "points": self.full_points}


def _load_level(level_no: int) -> MapLevel | None:
    path = Path(__file__).with_name(LEVEL_FILES[level_no])
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    origin = data.get("origin") or {}
    return MapLevel(level_no, str(data.get("world") or level_no), int(data["size"]),
                    (int(origin["x"]), int(origin["y"])), [int(t) for t in data["types"]],
                    LEVEL_CLEAR, LEVEL_POINTS[level_no], data.get("chests", []))


LEVELS: dict[int, MapLevel] = {}
for _no in sorted(LEVEL_FILES):
    _lv = _load_level(_no)
    if not _lv:
        break  # levels are contiguous
    LEVELS[_no] = _lv
if 1 not in LEVELS:  # no exported terrain: an empty ocean keeps the API alive
    LEVELS[1] = MapLevel(1, "lumen", 160, (80, 80), None, LEVEL_CLEAR, LEVEL_POINTS[1], [])
MAX_MAP_LEVEL = max(LEVELS)
_LEVELS_REVISION = hashlib.sha256(json.dumps(
    [[n, lv.world, lv.size, lv.ox, lv.oy, lv.clear, lv.full_points, lv.types] for n, lv in sorted(LEVELS.items())],
).encode()).hexdigest()

# Level 1: the harbour every student starts from.
MAP_SIZE = LEVELS[1].size
ORIGIN_X, ORIGIN_Y = LEVELS[1].ox, LEVELS[1].oy


def _level_move_cost(level: MapLevel, x: int, y: int) -> float:
    return _terrain_move_cost_type(level.terrain_at(x, y)) * (0.85 + _cell_discovery_hash(x, y) * 0.3)


@lru_cache(maxsize=4)
def _level_order(level_no: int) -> tuple[tuple[int, int], ...]:
    """Every tile of a level in the order the fog lifts (shared by all students):
    a Dijkstra flood from the harbour where land is cheap and open sea expensive."""
    lv = LEVELS[level_no]
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    dist: dict[tuple[int, int], float] = {(lv.ox, lv.oy): 0.0}
    heap: list[tuple[float, int, int]] = [(0.0, lv.ox, lv.oy)]
    order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while heap:
        d, x, y = heapq.heappop(heap)
        if d > dist.get((x, y), float("inf")) or (x, y) in seen:
            continue
        seen.add((x, y))
        order.append((x, y))
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= lv.size or ny >= lv.size:
                continue
            nd = d + (1.414 if dx and dy else 1.0) * _level_move_cost(lv, nx, ny)
            if nd < dist.get((nx, ny), float("inf")):
                dist[(nx, ny)] = nd
                heapq.heappush(heap, (nd, nx, ny))
    return tuple(order)


@lru_cache(maxsize=4096)
def _disc_count(level_no: int, radius10: int) -> int:
    """Cells of a disc of radius radius10/10 around the level's origin, clipped to the grid."""
    lv = LEVELS[level_no]
    radius = radius10 / 10
    r = int(math.ceil(radius))
    count = 0
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy <= radius * radius:
                x, y = lv.ox + dx, lv.oy + dy
                if 0 <= x < lv.size and 0 <= y < lv.size:
                    count += 1
    return count


def _level_radius(level_no: int, points: int) -> float:
    """Reveal radius (1 decimal, the precision the client sees) for points banked in a level.
    The browser charts the first N tiles of the discovery order, N = cells in that disc."""
    lv = LEVELS[level_no]
    progress = min(1.0, max(0, points) / lv.full_points)
    total = lv.size * lv.size
    c0 = _disc_count(level_no, int(round(lv.clear * 10)))
    target = c0 + (total - c0) * progress
    lo, hi = int(round(lv.clear * 10)), int(lv.full_radius * 10)
    while lo < hi:  # smallest radius (in tenths) whose disc covers the target
        mid = (lo + hi) // 2
        if _disc_count(level_no, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo / 10


def _points_to_radius(unlock_points: int) -> float:
    """Level 1 reveal radius for unlock points."""
    return _level_radius(1, unlock_points)


def _level_cells(level_no: int, radius: float) -> set[tuple[int, int]]:
    count = _disc_count(level_no, int(round(radius * 10)))
    return set(_level_order(level_no)[:count])


def _levels_for_points(points: int, full_unlock: bool = False) -> dict:
    """Which level a student is on and how far each level is charted. Finished
    levels stay fully charted; `full_unlock` (admin) charts all of level 1."""
    points = max(0, int(points))
    radius: dict[int, float] = {}
    start = 0
    map_level = 1
    for no in sorted(LEVELS):
        lv = LEVELS[no]
        into = points - start
        if into >= lv.full_points and no < MAX_MAP_LEVEL:
            radius[no] = float(lv.full_radius)
            start += lv.full_points
            continue
        radius[no] = _level_radius(no, into)
        map_level = no
        break
    if full_unlock:
        radius[1] = float(LEVELS[1].full_radius)
    level_points = points - start
    return {"map_level": map_level, "radius": radius,
            "level_points": min(level_points, LEVELS[map_level].full_points)}


def _cells_for_points(points: int) -> dict[int, set[tuple[int, int]]]:
    """Charted cells per level after `points` (replay of the unlock, no admin override)."""
    return {no: _level_cells(no, r) for no, r in _levels_for_points(points)["radius"].items()}


def _level_start_cells(level_no: int) -> set[tuple[int, int]]:
    return _level_cells(level_no, LEVELS[level_no].clear)


def xp_to_level(total_xp: int) -> dict:
    xp = max(0, int(total_xp))
    level = max(1, xp // XP_PER_LEVEL + 1)
    xp_in_level = xp % XP_PER_LEVEL
    return {
        "level": level,
        "xp": xp_in_level,
        "xp_max": XP_PER_LEVEL,
        "total_xp": xp,
    }


def _user_map_row(db, user_id: int) -> UserMapState:
    row = db.query(UserMapState).filter(UserMapState.user_id == user_id).first()
    if not row:
        row = UserMapState(user_id=user_id, pos_x=ORIGIN_X, pos_y=ORIGIN_Y, pos_world=LEVELS[1].world)
        db.add(row)
        db.flush()
    return row


def _user_full_unlock(user_id: int) -> bool:
    db = SessionLocal()
    try:
        row = db.query(UserMapState).filter(UserMapState.user_id == user_id).first()
        return bool(row and row.full_unlock)
    finally:
        db.close()


def _bonus_unlock_points(user_id: int) -> int:
    db = SessionLocal()
    try:
        row = db.query(UserMapState).filter(UserMapState.user_id == user_id).first()
        return int(row.bonus_unlock_points or 0) if row else 0
    finally:
        db.close()


def _unlocked_cells(user_id: int, radius: float) -> set[tuple[int, int]]:
    """Charted level 1 cells (admin full unlock charts all of them)."""
    if _user_full_unlock(user_id):
        return set(_level_order(1))
    return _level_cells(1, radius)


def _claimed_section_keys(user_id: int) -> set[tuple[str, int]]:
    db = SessionLocal()
    try:
        rows = db.query(SectionRewardClaim).filter(
            SectionRewardClaim.user_id == user_id,
        ).all()
        return {(r.folder_name, int(r.section_index)) for r in rows}
    finally:
        db.close()


def _collect_unlock_points(user_id: int) -> tuple[int, list[dict]]:
    """Sum unlock budget from 100%-mastered sections + bonus points."""
    import lesson as lesson_mod

    claimed = _claimed_section_keys(user_id)
    db = SessionLocal()
    try:
        outlines = db.query(CourseOutline).filter(CourseOutline.user_id == user_id).all()
        total = _bonus_unlock_points(user_id)
        unlocks: list[dict] = []
        for outline in outlines:
            sections = json.loads(outline.outline_json or "[]")
            progress = lesson_mod.get_section_mastery_list(
                user_id, outline.folder_name, sections, outline.current_section,
            )
            for i, sec in enumerate(sections):
                p = progress[i] if i < len(progress) else {}
                if p.get("mastery_pct") != 100:
                    continue
                key = (outline.folder_name, i)
                if key in claimed:
                    continue
                mins = max(int(sec.get("estimated_minutes") or 20), 25)
                total += mins
                unlocks.append({
                    "folder": outline.folder_name,
                    "section_index": i,
                    "title": sec.get("title", ""),
                    "minutes": mins,
                })
        return total, unlocks
    finally:
        db.close()


HARBOR_FOLDER = "__harbor__"
HARBOR_SECTION_INDEX = -1
HARBOR_TITLE = "Harbor Home"


def _section_title_from_outline(folder_name: str, section_index: int, user_id: int) -> str:
    db = SessionLocal()
    try:
        outline = db.query(CourseOutline).filter(
            CourseOutline.user_id == user_id,
            CourseOutline.folder_name == folder_name,
        ).first()
        if not outline:
            return ""
        sections = json.loads(outline.outline_json or "[]")
        if 0 <= section_index < len(sections):
            return sections[section_index].get("title", "") or ""
        return ""
    finally:
        db.close()


def _upsert_tile_tags(
    user_id: int,
    cells: set[tuple[int, int]],
    folder_name: str,
    section_index: int,
    section_title: str,
    map_level: int = 1,
) -> None:
    """Bulk upsert without querying each tile; retain its original section."""
    if not cells:
        return
    from sqlalchemy.dialects.sqlite import insert
    values = [{"user_id": user_id, "map_level": map_level, "x": x, "y": y, "folder_name": folder_name,
               "section_index": section_index, "section_title": section_title or ""} for x, y in cells]
    statement = insert(MapTileProvenance)
    statement = statement.on_conflict_do_update(index_elements=['user_id', 'map_level', 'x', 'y'],
        set_={"folder_name": statement.excluded.folder_name, "section_index": statement.excluded.section_index,
              "section_title": statement.excluded.section_title},
        where=(MapTileProvenance.folder_name == HARBOR_FOLDER) & (statement.excluded.folder_name != HARBOR_FOLDER))
    with SessionLocal() as db:
        db.execute(statement, values)
        db.commit()


def _harbor_starter_cells() -> set[tuple[int, int]]:
    return _level_start_cells(1)


def _provenance_unlock_events(user_id: int) -> list[dict]:
    """Unlock waves in real completion order: claims chronologically, then unclaimed mastery."""
    import lesson as lesson_mod

    db = SessionLocal()
    try:
        claimed_keys: set[tuple[str, int]] = set()
        events: list[dict] = []

        claims = (
            db.query(SectionRewardClaim)
            .filter(SectionRewardClaim.user_id == user_id)
            .order_by(SectionRewardClaim.created_at)
            .all()
        )
        outline_by_folder = {
            o.folder_name: json.loads(o.outline_json or "[]")
            for o in db.query(CourseOutline).filter(CourseOutline.user_id == user_id).all()
        }

        for claim in claims:
            idx = int(claim.section_index)
            sections = outline_by_folder.get(claim.folder_name, [])
            title = ""
            if 0 <= idx < len(sections):
                title = sections[idx].get("title", "") or ""
            key = (claim.folder_name, idx)
            claimed_keys.add(key)
            events.append({
                "folder": claim.folder_name,
                "section_index": idx,
                "title": title,
                "points": int(claim.map_bonus_added or 0),
            })

        outlines = db.query(CourseOutline).filter(CourseOutline.user_id == user_id).all()
        unclaimed: list[dict] = []
        for outline in outlines:
            sections = json.loads(outline.outline_json or "[]")
            progress = lesson_mod.get_section_mastery_list(
                user_id, outline.folder_name, sections, outline.current_section,
            )
            for i, sec in enumerate(sections):
                p = progress[i] if i < len(progress) else {}
                if p.get("mastery_pct") != 100:
                    continue
                key = (outline.folder_name, i)
                if key in claimed_keys:
                    continue
                unclaimed.append({
                    "folder": outline.folder_name,
                    "section_index": i,
                    "title": sec.get("title", "") or "",
                    "points": max(int(sec.get("estimated_minutes") or 20), 25),
                })

        unclaimed.sort(key=lambda e: (e["folder"], e["section_index"]))
        events.extend(unclaimed)
        return events
    finally:
        db.close()


def _load_tile_tag_map(user_id: int) -> dict[tuple[int, int, int], dict]:
    """(map_level, x, y) -> the section that charted it."""
    db = SessionLocal()
    try:
        rows = db.query(MapTileProvenance).filter(MapTileProvenance.user_id == user_id).all()
        return {
            (int(r.map_level or 1), r.x, r.y): {
                "folder": r.folder_name,
                "section_index": int(r.section_index),
                "title": r.section_title or "",
            }
            for r in rows
        }
    finally:
        db.close()


def _replay_unlocks(user_id: int, tag) -> None:
    """Replay every unlock in completion order, calling tag(level, cells, folder, index, title)
    for the cells each event newly charted. Level 1 opens with the harbour; level 2 opens
    with its own harbour once level 1 is fully charted."""
    harbor = _harbor_starter_cells()
    tag(1, harbor, HARBOR_FOLDER, HARBOR_SECTION_INDEX, HARBOR_TITLE)
    cumulative: dict[int, set[tuple[int, int]]] = {1: set(harbor)}
    unlock_sim = 0
    events = _provenance_unlock_events(user_id)

    def advance(points: int, folder: str, index: int, title: str) -> None:
        for level, cells in _cells_for_points(points).items():
            before = cumulative.get(level)
            if before is None:
                start = _level_start_cells(level)
                tag(level, start, HARBOR_FOLDER, HARBOR_SECTION_INDEX, f"{HARBOR_TITLE} (level {level})")
                before = set(start)
            new_cells = cells - before
            if new_cells:
                tag(level, new_cells, folder, index, title)
            cumulative[level] = before | cells

    for event in events:
        unlock_sim += int(event["points"])
        advance(unlock_sim, event["folder"], int(event["section_index"]), event["title"])

    unlock_total, _ = _collect_unlock_points(user_id)
    orphan = unlock_total - unlock_sim
    if orphan > 0:
        if events:
            last = events[-1]
            advance(unlock_sim + orphan, last["folder"], int(last["section_index"]), last["title"])
        else:
            advance(unlock_sim + orphan, HARBOR_FOLDER, HARBOR_SECTION_INDEX, HARBOR_TITLE)


def _sections_with_tiles(user_id: int) -> set[tuple[str, int]]:
    """Sections that receive at least one tile during a full provenance replay."""
    tagged: set[tuple[str, int]] = set()

    def tag(level, cells, folder, index, title):
        if cells and folder != HARBOR_FOLDER:
            tagged.add((folder, int(index)))

    _replay_unlocks(user_id, tag)
    return tagged


def _charted_by_level(user_id: int, unlock_points: int) -> dict[int, set[tuple[int, int]]]:
    """Every charted cell on every level for this student right now."""
    state = _levels_for_points(unlock_points, _user_full_unlock(user_id))
    cells = {1: _unlocked_cells(user_id, state["radius"][1])}
    for no, radius in state["radius"].items():
        if no != 1:
            cells[no] = _level_cells(no, radius)
    return cells


def _provenance_needs_sync(user_id: int) -> bool:
    tag_map = _load_tile_tag_map(user_id)
    unlock_points, _ = _collect_unlock_points(user_id)
    tagged_cells = set(tag_map.keys())
    charted = {(level, x, y) for level, cells in _charted_by_level(user_id, unlock_points).items() for x, y in cells}
    # Untagged charted land, or tags left on land that is no longer charted
    # (e.g. from a world that has since been replaced).
    if charted - tagged_cells or tagged_cells - charted:
        return True

    harbor_tagged = sum(1 for (lv, _, _), v in tag_map.items() if lv == 1 and v["folder"] == HARBOR_FOLDER)
    if harbor_tagged > len(_harbor_starter_cells()) + 50:
        return True

    tagged_sections = {
        (v["folder"], v["section_index"])
        for v in tag_map.values()
        if v["folder"] != HARBOR_FOLDER
    }
    return tagged_sections != _sections_with_tiles(user_id)


def _sync_tile_provenance(user_id: int) -> None:
    """Rebuild tile tags: harbour starter, then one wave per section unlock, on every level."""
    pending = []

    def tag(level, cells, folder, index, title):
        pending.extend({"user_id": user_id, "map_level": level, "x": x, "y": y, "folder_name": folder,
                        "section_index": index, "section_title": title} for x, y in cells)

    _replay_unlocks(user_id, tag)
    from sqlalchemy import insert
    with SessionLocal() as db:
        db.query(MapTileProvenance).filter_by(user_id=user_id).delete()
        if pending:
            db.execute(insert(MapTileProvenance), pending)
        db.commit()
    invalidate_map_cache(user_id)


def _rebuild_tile_provenance(user_id: int) -> None:
    """Backfill or rebuild tile tags when mastery/outdated tags drift."""
    if _provenance_needs_sync(user_id):
        _sync_tile_provenance(user_id)


def _assign_tiles_to_section(
    user_id: int,
    folder_name: str,
    section_index: int,
    section_title: str,
    points_before: int,
    points_after: int,
) -> None:
    """Tag newly revealed cells (on any level) with the section that unlocked them."""
    before = _cells_for_points(points_before)
    for level, cells in _cells_for_points(points_after).items():
        prev = before.get(level)
        if prev is None:
            start = _level_start_cells(level)
            _upsert_tile_tags(user_id, start, HARBOR_FOLDER, HARBOR_SECTION_INDEX, f"{HARBOR_TITLE} (level {level})", level)
            prev = start
        _upsert_tile_tags(user_id, cells - prev, folder_name, section_index, section_title, level)


def _tile_sections_map(user_id: int) -> dict[str, dict]:
    """Level 1 tiles are keyed "x,y"; later levels "<level>:x,y"."""
    db = SessionLocal()
    try:
        rows = db.query(MapTileProvenance).filter(MapTileProvenance.user_id == user_id).all()
        return {
            (f"{r.x},{r.y}" if int(r.map_level or 1) == 1 else f"{int(r.map_level)}:{r.x},{r.y}"): {
                "folder": r.folder_name,
                "section_index": int(r.section_index),
                "title": r.section_title or "",
            }
            for r in rows
        }
    finally:
        db.close()


def _reward_payload(
    user_id: int,
    *,
    xp_gained: int,
    total_xp: int,
    section_title: str,
    lesson_complete: bool,
    points_before: int,
    points_after: int,
) -> dict:
    full_unlock = _user_full_unlock(user_id)
    state_before = _levels_for_points(points_before, full_unlock)
    state_after = _levels_for_points(points_after, full_unlock)
    before = _charted_by_level(user_id, points_before)
    after = _charted_by_level(user_id, points_after)
    tiles_delta = sum(max(0, len(after[lv]) - len(before.get(lv, ()))) for lv in after)
    level = state_after["map_level"]
    radius_before = state_before["radius"].get(level, LEVELS[level].clear)
    radius_after = state_after["radius"][level]
    total_cells = LEVELS[level].size ** 2
    explored_before = len(before.get(level, ()))
    explored_after = len(after[level])
    level_info = xp_to_level(total_xp)
    return {
        "xp_gained": xp_gained,
        "total_xp": total_xp,
        "level": level_info["level"],
        "xp_in_level": level_info["xp"],
        "xp_max": level_info["xp_max"],
        "section_title": section_title,
        "lesson_complete": lesson_complete,
        "map": {
            "map_level": level,
            "level_up": level > state_before["map_level"],
            "reveal_radius": round(radius_after, 1),
            "radius_delta": round(max(0.0, radius_after - radius_before), 1),
            "explored_pct": round(explored_after / total_cells * 100, 1),
            "explored_delta_pct": round(
                max(0.0, (explored_after - explored_before) / total_cells * 100), 1,
            ),
            "tiles_unlocked": explored_after,
            "tiles_unlocked_delta": tiles_delta,
            "unlock_points": points_after,
        },
    }


_reward_locks = [threading.RLock() for _ in range(64)]


def claim_section_reward(user_id, folder_name, section_index, **kwargs):
    with _reward_locks[int(user_id) % len(_reward_locks)]:
        return _claim_section_reward(user_id, folder_name, section_index, **kwargs)


def _claim_section_reward(
    user_id: int,
    folder_name: str,
    section_index: int,
    *,
    section_title: str = "",
    lesson_complete: bool = False,
    section_minutes: int = 25,
) -> dict:
    """Grant XP + map expansion when Pedro marks [SECTION_COMPLETE]. Idempotent."""
    db = SessionLocal()
    try:
        from sqlalchemy import text
        db.execute(text("BEGIN IMMEDIATE"))
        existing = db.query(SectionRewardClaim).filter(
            SectionRewardClaim.user_id == user_id,
            SectionRewardClaim.folder_name == folder_name,
            SectionRewardClaim.section_index == section_index,
        ).first()
        if existing:
            existing_xp = int(existing.xp_gained or 0)
            db.rollback()  # Release writer before building/repairing a map snapshot.
            state = get_map_state(user_id)
            points = int(state.get("unlock_points") or 0)
            return {
                "already_claimed": True,
                **_reward_payload(
                    user_id,
                    xp_gained=existing_xp,
                    total_xp=int(state.get("total_xp") or 0),
                    section_title=section_title,
                    lesson_complete=lesson_complete,
                    points_before=points,
                    points_after=points,
                ),
            }

        unlock_before, _ = _collect_unlock_points(user_id)

        xp_gained = XP_PER_SECTION
        map_bonus = BONUS_UNLOCK_PER_SECTION + max(section_minutes, 25)
        if lesson_complete:
            xp_gained += XP_LESSON_COMPLETE_BONUS
            map_bonus += BONUS_UNLOCK_LESSON_COMPLETE

        row = _user_map_row(db, user_id)
        row.total_xp = int(row.total_xp or 0) + xp_gained
        row.bonus_unlock_points = int(row.bonus_unlock_points or 0) + map_bonus
        db.add(SectionRewardClaim(
            user_id=user_id,
            folder_name=folder_name,
            section_index=section_index,
            xp_gained=xp_gained,
            map_bonus_added=map_bonus,
        ))
        db.commit()
        total_xp = int(row.total_xp)
    finally:
        db.close()

    unlock_after, _ = _collect_unlock_points(user_id)
    title = section_title or _section_title_from_outline(folder_name, section_index, user_id)
    _assign_tiles_to_section(
        user_id, folder_name, section_index, title, unlock_before, unlock_after,
    )
    if _provenance_needs_sync(user_id):
        _sync_tile_provenance(user_id)
    invalidate_map_cache(user_id)
    return _reward_payload(
        user_id,
        xp_gained=xp_gained,
        total_xp=total_xp,
        section_title=section_title,
        lesson_complete=lesson_complete,
        points_before=unlock_before,
        points_after=unlock_after,
    )


_map_cache = OrderedDict()
_map_cache_lock = threading.RLock()
_MAP_CACHE_LIMIT = 32


def invalidate_map_cache(user_id):
    with _map_cache_lock:
        _map_cache.pop(user_id, None)
    with SessionLocal() as db:
        db.query(MapSnapshot).filter_by(user_id=user_id).delete()
        db.commit()


def _map_signature(db, user_id):
    from sqlalchemy import func
    row = db.get(UserMapState, user_id)
    stamp = db.query(func.max(CourseOutline.updated_at), func.count(CourseOutline.id)).filter_by(user_id=user_id).one()
    # Version changes deliberately rebuild saved projections after geometry/schema updates.
    signature = json.dumps([4, _LEVELS_REVISION,
        row.bonus_unlock_points, row.total_xp, row.full_unlock, str(stamp)]) if row else ''
    position = {"x": row.pos_x, "y": row.pos_y} if row else None
    return signature, position


def _compact_map(state):
    catalog, ids, tiles = [], {}, {}
    for cell, section in state.get('tile_sections', {}).items():
        key = (section.get('folder'), section.get('section_index'), section.get('title'))
        if key not in ids:
            ids[key] = len(catalog)
            catalog.append(section)
        tiles[cell] = ids[key]
    return {**state, 'section_catalog': catalog, 'tile_sections': tiles}


def get_map_state(user_id: int, *, compact: bool = False) -> dict:
    # Same local lock as completion: don't publish a partial reward/provenance update.
    with _reward_locks[int(user_id) % len(_reward_locks)]:
        return _get_map_state(user_id, compact=compact)


def _get_map_state(user_id, *, compact=False):
    with SessionLocal() as db:
        signature, position = _map_signature(db, user_id)
        saved = db.get(MapSnapshot, user_id)
        saved_payload = saved.payload if saved and saved.signature == signature else None
    with _map_cache_lock:
        cached = _map_cache.get(user_id)
        if cached and cached[0] == signature:
            state = dict(cached[1])
            _map_cache.move_to_end(user_id)
        else:
            state = None
    if state is None and saved_payload:
        try:
            state = json.loads(saved_payload)
            catalog = state.pop('section_catalog')
            state['tile_sections'] = {cell: catalog[index] for cell, index in state['tile_sections'].items()}
        except (ValueError, KeyError, IndexError, TypeError):
            state = None  # A cache can always be rebuilt from durable learning records.
    if state is None:
        state = _build_map_state(user_id)
        position = None
        with SessionLocal() as db:
            signature, _ = _map_signature(db, user_id)
            db.merge(MapSnapshot(user_id=user_id, signature=signature,
                payload=json.dumps(_compact_map(state), separators=(',', ':'))))
            db.commit()
    with _map_cache_lock:
        _map_cache[user_id] = (signature, state)
        _map_cache.move_to_end(user_id)
        while len(_map_cache) > _MAP_CACHE_LIMIT:
            _map_cache.popitem(last=False)
    state = dict(state)
    if position:
        state['player'] = position
    import treasure
    state['treasures'] = treasure.get_treasure_state(user_id)
    return _compact_map(state) if compact else state


def is_charted(user_id: int, level: int, x: int, y: int) -> bool:
    """Whether a tile on a given level is charted for this student."""
    unlock_points, _ = _collect_unlock_points(user_id)
    cells = _charted_by_level(user_id, unlock_points).get(int(level))
    return bool(cells) and (int(x), int(y)) in cells


def _level_now(user_id: int, unlock_points: int | None = None) -> tuple[dict, MapLevel, set[tuple[int, int]]]:
    """The student's current level, its geometry and its charted cells."""
    if unlock_points is None:
        unlock_points, _ = _collect_unlock_points(user_id)
    state = _levels_for_points(unlock_points, _user_full_unlock(user_id))
    level = LEVELS[state["map_level"]]
    radius = state["radius"][level.level]
    cells = _unlocked_cells(user_id, radius) if level.level == 1 else _level_cells(level.level, radius)
    return state, level, cells


def _build_map_state(user_id: int) -> dict:
    _rebuild_tile_provenance(user_id)
    unlock_points, recent_unlocks = _collect_unlock_points(user_id)
    events = _provenance_unlock_events(user_id)
    recent_unlocks = [{"folder": e["folder"], "section_index": e["section_index"],
                       "title": e["title"], "minutes": e["points"]} for e in events]
    state, level, unlocked = _level_now(user_id, unlock_points)
    radius = state["radius"][level.level]

    db = SessionLocal()
    try:
        row = _user_map_row(db, user_id)
        db.commit()
        db.refresh(row)
        px, py = row.pos_x, row.pos_y
        total_xp = int(row.total_xp or 0)
        # A position saved in another world (or in the fog) goes back to this world's harbour.
        if row.pos_world != level.world or (px, py) not in unlocked:
            px, py = level.ox, level.oy
            row.pos_x, row.pos_y, row.pos_world = px, py, level.world
            db.commit()
    finally:
        db.close()

    explored_pct = round(len(unlocked) / (level.size * level.size) * 100, 1)
    level_info = xp_to_level(total_xp)

    import treasure
    treasure_state = treasure.get_treasure_state(user_id)

    return {
        "size": level.size,
        "origin": {"x": level.ox, "y": level.oy},
        "origins": {str(k): {"x": v.ox, "y": v.oy} for k, v in LEVELS.items()},
        "player": {"x": px, "y": py},
        "map_level": level.level,
        "map_world": level.world,
        "max_map_level": MAX_MAP_LEVEL,
        "reveal_radius": round(radius, 1),
        "reveal_pacing": level.pacing_payload(),
        "level_points": state["level_points"],
        "unlock_points": unlock_points,
        "sections_mastered": len(recent_unlocks),
        "recent_unlocks": recent_unlocks[-8:],
        "explored_pct": explored_pct,
        "total_xp": level_info["total_xp"],
        "level": level_info["level"],
        "xp": level_info["xp"],
        "xp_max": level_info["xp_max"],
        "treasures": treasure_state,
        "tile_sections": _tile_sections_map(user_id),
    }


def _player_in_unlocked(user_id: int) -> tuple[int, int, set[tuple[int, int]]]:
    _, level, unlocked = _level_now(user_id)
    db = SessionLocal()
    try:
        row = _user_map_row(db, user_id)
        db.commit()
        px, py = row.pos_x, row.pos_y
        if row.pos_world != level.world or (px, py) not in unlocked:
            px, py = level.ox, level.oy
            row.pos_x, row.pos_y, row.pos_world = px, py, level.world
            db.commit()
        return px, py, unlocked
    finally:
        db.close()


def move_player(user_id: int, dx: int, dy: int) -> dict:
    dx = max(-1, min(1, int(dx)))
    dy = max(-1, min(1, int(dy)))
    px, py, unlocked = _player_in_unlocked(user_id)
    nx = px + dx
    ny = py + dy
    if (nx, ny) not in unlocked:
        state = get_map_state(user_id)
        return {"ok": False, "error": "That area is still hidden in the fog.", **state}
    db = SessionLocal()
    try:
        row = db.query(UserMapState).filter(UserMapState.user_id == user_id).first()
        if row:
            row.pos_x, row.pos_y = nx, ny
            db.commit()
    finally:
        db.close()
    return {"ok": True, **get_map_state(user_id)}


def teleport_player(user_id: int, x: int, y: int) -> dict:
    _, _, unlocked = _player_in_unlocked(user_id)
    if (x, y) not in unlocked:
        state = get_map_state(user_id)
        return {"ok": False, "error": "Cannot move into fog.", **state}
    db = SessionLocal()
    try:
        row = db.query(UserMapState).filter(UserMapState.user_id == user_id).first()
        if row:
            row.pos_x, row.pos_y = int(x), int(y)
            db.commit()
    finally:
        db.close()
    return {"ok": True, **get_map_state(user_id)}


def set_full_unlock(user_id: int, enabled: bool = True) -> dict:
    db = SessionLocal()
    try:
        row = _user_map_row(db, user_id)
        row.full_unlock = enabled
        db.commit()
    finally:
        db.close()
    return get_map_state(user_id)
