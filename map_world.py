"""Exploration map — fog of war unlocked by completed lesson sections + XP."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Optional

from database import CourseOutline, MapTileProvenance, SectionRewardClaim, SessionLocal, UserMapState

MAP_SEED = 0xC04A57

# Grid geometry + terrain come from map_terrain_types.json, exported from the
# frontend generator (scripts/export-map-terrain.mjs) so the organic unlock
# shape matches the rendered world exactly.
_MAP_FILE_PATH = Path(__file__).with_name("map_terrain_types.json")


def _read_map_file() -> dict:
    if _MAP_FILE_PATH.exists():
        with _MAP_FILE_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


_map_file = _read_map_file()
MAP_SIZE = int(_map_file.get("size") or 144)
_origin_meta = _map_file.get("origin") or {}
ORIGIN_X = int(_origin_meta.get("x", MAP_SIZE // 2))
ORIGIN_Y = int(_origin_meta.get("y", MAP_SIZE // 2))
ORIGIN_CLEAR = 4
FULL_REVEAL_RADIUS = math.ceil(math.hypot(
    max(ORIGIN_X, MAP_SIZE - ORIGIN_X),
    max(ORIGIN_Y, MAP_SIZE - ORIGIN_Y),
)) + 2

# Must match coastWorldMap.js TERRAIN enum (0–14).
T_BEACH = 4
T_SHALLOW = 2
T_REEF = 3
T_OCEAN = 1
T_PATH = 14
T_MOUNTAIN = 9
T_PEAK = 10
T_DEEP_FOREST = 8
T_LAVA = 12
T_SWAMP = 11

# XP awarded on section / lesson completion (persisted on UserMapState).
XP_PER_SECTION = 100
XP_LESSON_COMPLETE_BONUS = 500
# Extra map unlock budget beyond section estimated minutes.
BONUS_UNLOCK_PER_SECTION = 35
BONUS_UNLOCK_LESSON_COMPLETE = 120

# Unlock points for a full map (~10 mastered four-section lessons:
# 4×(35+25) + 120 = 360 per lesson → 3600 total).
FULL_MAP_UNLOCK_POINTS = 3600

XP_PER_LEVEL = 400

_terrain_types: list[int] | None = (
    [int(t) for t in _map_file["types"]] if "types" in _map_file else None
)


def _load_terrain_types() -> list[int]:
    global _terrain_types
    if _terrain_types is None:
        _terrain_types = [T_OCEAN] * (MAP_SIZE * MAP_SIZE)
    return _terrain_types


def _terrain_type_at(x: int, y: int) -> int:
    types = _load_terrain_types()
    if x < 0 or y < 0 or x >= MAP_SIZE or y >= MAP_SIZE:
        return 0
    return types[y * MAP_SIZE + x]


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


def _movement_cost_at(x: int, y: int, size: int = MAP_SIZE) -> float:
    base = _terrain_move_cost_type(_terrain_type_at(x, y))
    jitter = 0.85 + _cell_discovery_hash(x, y) * 0.3
    return base * jitter


def _cell_discovery_hash(x: int, y: int) -> float:
    n = (x * 374761393 + y * 668265263) & 0xFFFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0


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
        row = UserMapState(user_id=user_id, pos_x=ORIGIN_X, pos_y=ORIGIN_Y)
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


def _effective_radius(user_id: int, unlock_points: int) -> float:
    if _user_full_unlock(user_id):
        return float(FULL_REVEAL_RADIUS)
    return _points_to_radius(unlock_points)


def _cells_in_radius(cx: int, cy: int, radius: float) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    r = int(math.ceil(radius))
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy <= radius * radius:
                x, y = cx + dx, cy + dy
                if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                    out.add((x, y))
    return out


def _cells_organic_unlock(cx: int, cy: int, radius: float) -> set[tuple[int, int]]:
    """Noisy Dijkstra from HQ — same budget as a circle, organic coastal shape."""
    target = len(_cells_in_radius(cx, cy, radius))
    if target <= 0:
        return set()

    dirs = (
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    dist: dict[tuple[int, int], float] = {(cx, cy): 0.0}
    heap: list[tuple[float, int, int]] = [(0.0, cx, cy)]
    unlocked: set[tuple[int, int]] = set()

    while heap and len(unlocked) < target:
        d, x, y = heapq.heappop(heap)
        if d > dist.get((x, y), float("inf")):
            continue
        if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
            continue
        unlocked.add((x, y))

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= MAP_SIZE or ny >= MAP_SIZE:
                continue
            step = 1.414 if dx and dy else 1.0
            nd = d + step * _movement_cost_at(nx, ny)
            if nd < dist.get((nx, ny), float("inf")):
                dist[(nx, ny)] = nd
                heapq.heappush(heap, (nd, nx, ny))

    return unlocked


def _unlocked_cells(user_id: int, radius: float) -> set[tuple[int, int]]:
    if _user_full_unlock(user_id):
        return {(x, y) for x in range(MAP_SIZE) for y in range(MAP_SIZE)}
    return _cells_organic_unlock(ORIGIN_X, ORIGIN_Y, radius)


def _points_to_radius(unlock_points: int) -> float:
    """Map unlock points → reveal radius (linear to full map at FULL_MAP_UNLOCK_POINTS)."""
    if unlock_points <= 0:
        return float(ORIGIN_CLEAR)
    progress = min(1.0, unlock_points / FULL_MAP_UNLOCK_POINTS)
    return ORIGIN_CLEAR + (FULL_REVEAL_RADIUS - ORIGIN_CLEAR) * progress


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


def _write_tile_tags(
    user_id: int,
    cells: set[tuple[int, int]],
    folder_name: str,
    section_index: int,
    section_title: str,
) -> None:
    """Bulk-write provenance (used after a full wipe)."""
    if not cells:
        return
    db = SessionLocal()
    try:
        for x, y in cells:
            db.add(MapTileProvenance(
                user_id=user_id,
                x=x,
                y=y,
                folder_name=folder_name,
                section_index=section_index,
                section_title=section_title or "",
            ))
        db.commit()
    finally:
        db.close()


def _upsert_tile_tags(
    user_id: int,
    cells: set[tuple[int, int]],
    folder_name: str,
    section_index: int,
    section_title: str,
) -> None:
    """Insert provenance for cells; replace stale harbor tags when a section claims them."""
    if not cells:
        return
    db = SessionLocal()
    try:
        for x, y in cells:
            exists = db.query(MapTileProvenance).filter(
                MapTileProvenance.user_id == user_id,
                MapTileProvenance.x == x,
                MapTileProvenance.y == y,
            ).first()
            if exists:
                if exists.folder_name == HARBOR_FOLDER and folder_name != HARBOR_FOLDER:
                    exists.folder_name = folder_name
                    exists.section_index = section_index
                    exists.section_title = section_title or ""
                continue
            db.add(MapTileProvenance(
                user_id=user_id,
                x=x,
                y=y,
                folder_name=folder_name,
                section_index=section_index,
                section_title=section_title or "",
            ))
        db.commit()
    finally:
        db.close()


def _harbor_starter_cells() -> set[tuple[int, int]]:
    return _cells_organic_unlock(ORIGIN_X, ORIGIN_Y, float(ORIGIN_CLEAR))


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


def _load_tile_tag_map(user_id: int) -> dict[tuple[int, int], dict]:
    db = SessionLocal()
    try:
        rows = db.query(MapTileProvenance).filter(MapTileProvenance.user_id == user_id).all()
        return {
            (r.x, r.y): {
                "folder": r.folder_name,
                "section_index": int(r.section_index),
                "title": r.section_title or "",
            }
            for r in rows
        }
    finally:
        db.close()


def _clear_tile_provenance(user_id: int) -> None:
    db = SessionLocal()
    try:
        db.query(MapTileProvenance).filter(MapTileProvenance.user_id == user_id).delete()
        db.commit()
    finally:
        db.close()


def _sections_with_tiles(user_id: int) -> set[tuple[str, int]]:
    """Sections that receive at least one tile during a full provenance replay."""
    cumulative = set(_harbor_starter_cells())
    unlock_sim = 0
    tagged: set[tuple[str, int]] = set()
    for event in _provenance_unlock_events(user_id):
        unlock_sim += int(event["points"])
        radius_after = _points_to_radius(unlock_sim)
        after = _cells_organic_unlock(ORIGIN_X, ORIGIN_Y, radius_after)
        if after - cumulative:
            tagged.add((event["folder"], int(event["section_index"])))
        cumulative = after
    return tagged


def _provenance_needs_sync(user_id: int) -> bool:
    tag_map = _load_tile_tag_map(user_id)
    unlock_points, _ = _collect_unlock_points(user_id)
    radius = _effective_radius(user_id, unlock_points)
    unlocked = _unlocked_cells(user_id, radius)
    if unlocked - set(tag_map.keys()):
        return True

    harbor_tagged = sum(1 for v in tag_map.values() if v["folder"] == HARBOR_FOLDER)
    if harbor_tagged > len(_harbor_starter_cells()) + 50:
        return True

    tagged_sections = {
        (v["folder"], v["section_index"])
        for v in tag_map.values()
        if v["folder"] != HARBOR_FOLDER
    }
    return tagged_sections != _sections_with_tiles(user_id)


def _sync_tile_provenance(user_id: int) -> None:
    """Rebuild tile tags: harbor starter, then one wave per section unlock."""
    _clear_tile_provenance(user_id)

    harbor_cells = _harbor_starter_cells()
    _write_tile_tags(user_id, harbor_cells, HARBOR_FOLDER, HARBOR_SECTION_INDEX, HARBOR_TITLE)

    cumulative = set(harbor_cells)
    unlock_sim = 0
    events = _provenance_unlock_events(user_id)
    for event in events:
        unlock_sim += int(event["points"])
        radius_after = _points_to_radius(unlock_sim)
        after = _cells_organic_unlock(ORIGIN_X, ORIGIN_Y, radius_after)
        new_cells = after - cumulative
        if new_cells:
            _write_tile_tags(
                user_id,
                new_cells,
                event["folder"],
                int(event["section_index"]),
                event["title"],
            )
        cumulative = after

    unlock_total, _ = _collect_unlock_points(user_id)
    orphan = unlock_total - unlock_sim
    if orphan > 0:
        radius_after = _points_to_radius(unlock_sim + orphan)
        after = _cells_organic_unlock(ORIGIN_X, ORIGIN_Y, radius_after)
        new_cells = after - cumulative
        if new_cells:
            if events:
                last = events[-1]
                _write_tile_tags(
                    user_id,
                    new_cells,
                    last["folder"],
                    int(last["section_index"]),
                    last["title"],
                )
            else:
                _write_tile_tags(
                    user_id, new_cells, HARBOR_FOLDER, HARBOR_SECTION_INDEX, HARBOR_TITLE,
                )


def _rebuild_tile_provenance(user_id: int) -> None:
    """Backfill or rebuild tile tags when mastery/outdated tags drift."""
    if _provenance_needs_sync(user_id):
        _sync_tile_provenance(user_id)


def _assign_tiles_to_section(
    user_id: int,
    folder_name: str,
    section_index: int,
    section_title: str,
    radius_before: float,
    radius_after: float,
) -> None:
    """Tag newly revealed cells with the section that unlocked them."""
    before = _cells_organic_unlock(ORIGIN_X, ORIGIN_Y, radius_before)
    after = _cells_organic_unlock(ORIGIN_X, ORIGIN_Y, radius_after)
    new_cells = after - before
    if not new_cells:
        return
    _upsert_tile_tags(user_id, new_cells, folder_name, section_index, section_title)


def _tile_sections_map(user_id: int) -> dict[str, dict]:
    db = SessionLocal()
    try:
        rows = db.query(MapTileProvenance).filter(MapTileProvenance.user_id == user_id).all()
        return {
            f"{r.x},{r.y}": {
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
    radius_before: float,
    radius_after: float,
) -> dict:
    explored_before = len(_unlocked_cells(user_id, radius_before))
    explored_after = len(_unlocked_cells(user_id, radius_after))
    total_cells = MAP_SIZE * MAP_SIZE
    level_info = xp_to_level(total_xp)
    unlock_after, _ = _collect_unlock_points(user_id)
    return {
        "xp_gained": xp_gained,
        "total_xp": total_xp,
        "level": level_info["level"],
        "xp_in_level": level_info["xp"],
        "xp_max": level_info["xp_max"],
        "section_title": section_title,
        "lesson_complete": lesson_complete,
        "map": {
            "reveal_radius": round(radius_after, 1),
            "radius_delta": round(max(0.0, radius_after - radius_before), 1),
            "explored_pct": round(explored_after / total_cells * 100, 1),
            "explored_delta_pct": round(
                max(0.0, (explored_after - explored_before) / total_cells * 100), 1,
            ),
            "unlock_points": unlock_after,
        },
    }


def claim_section_reward(
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
        existing = db.query(SectionRewardClaim).filter(
            SectionRewardClaim.user_id == user_id,
            SectionRewardClaim.folder_name == folder_name,
            SectionRewardClaim.section_index == section_index,
        ).first()
        if existing:
            state = get_map_state(user_id)
            return {
                "already_claimed": True,
                **_reward_payload(
                    user_id,
                    xp_gained=int(existing.xp_gained or 0),
                    total_xp=int(state.get("total_xp") or 0),
                    section_title=section_title,
                    lesson_complete=lesson_complete,
                    radius_before=state.get("reveal_radius", ORIGIN_CLEAR),
                    radius_after=state.get("reveal_radius", ORIGIN_CLEAR),
                ),
            }

        unlock_before, _ = _collect_unlock_points(user_id)
        radius_before = _effective_radius(user_id, unlock_before)

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
    radius_after = _effective_radius(user_id, unlock_after)
    title = section_title or _section_title_from_outline(folder_name, section_index, user_id)
    _assign_tiles_to_section(
        user_id, folder_name, section_index, title, radius_before, radius_after,
    )
    if _provenance_needs_sync(user_id):
        _sync_tile_provenance(user_id)
    return _reward_payload(
        user_id,
        xp_gained=xp_gained,
        total_xp=total_xp,
        section_title=section_title,
        lesson_complete=lesson_complete,
        radius_before=radius_before,
        radius_after=radius_after,
    )


def get_map_state(user_id: int) -> dict:
    _rebuild_tile_provenance(user_id)
    unlock_points, recent_unlocks = _collect_unlock_points(user_id)
    radius = _effective_radius(user_id, unlock_points)

    db = SessionLocal()
    try:
        row = _user_map_row(db, user_id)
        db.commit()
        db.refresh(row)
        px, py = row.pos_x, row.pos_y
        total_xp = int(row.total_xp or 0)
        unlocked = _unlocked_cells(user_id, radius)
        if (px, py) not in unlocked:
            px, py = ORIGIN_X, ORIGIN_Y
            row.pos_x, row.pos_y = px, py
            db.commit()
    finally:
        db.close()

    explored_pct = round(len(unlocked) / (MAP_SIZE * MAP_SIZE) * 100, 1)
    level_info = xp_to_level(total_xp)

    import treasure
    treasure_state = treasure.get_treasure_state(user_id)

    return {
        "size": MAP_SIZE,
        "origin": {"x": ORIGIN_X, "y": ORIGIN_Y},
        "player": {"x": px, "y": py},
        "reveal_radius": round(radius, 1),
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


def _player_in_unlocked(user_id: int, radius: float) -> tuple[int, int, set[tuple[int, int]]]:
    unlocked = _unlocked_cells(user_id, radius)
    db = SessionLocal()
    try:
        row = _user_map_row(db, user_id)
        db.commit()
        px, py = row.pos_x, row.pos_y
        if (px, py) not in unlocked:
            px, py = ORIGIN_X, ORIGIN_Y
            row.pos_x, row.pos_y = px, py
            db.commit()
        return px, py, unlocked
    finally:
        db.close()


def move_player(user_id: int, dx: int, dy: int) -> dict:
    dx = max(-1, min(1, int(dx)))
    dy = max(-1, min(1, int(dy)))
    unlock_points, _ = _collect_unlock_points(user_id)
    radius = _effective_radius(user_id, unlock_points)
    px, py, unlocked = _player_in_unlocked(user_id, radius)
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
    unlock_points, _ = _collect_unlock_points(user_id)
    radius = _effective_radius(user_id, unlock_points)
    _, _, unlocked = _player_in_unlocked(user_id, radius)
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
