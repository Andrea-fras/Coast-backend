#!/usr/bin/env python3
"""Map levels: level 1 is the Lumen Reaches, level 2 (Neon Meridian) opens once
level 1 is fully charted. Isolated SQLite."""
import json
import unittest
from datetime import datetime, timedelta, timezone

import test_http_integrity as fixture
from database import SessionLocal, SectionRewardClaim, UserMapState, MapTileProvenance, CourseOutline
import map_world

LUMEN_HARBOUR = {'x': 75, 'y': 107}
NEON_HARBOUR = {'x': 72, 'y': 110}


def claim(db, folder, index, points, minutes_ago):
    db.add(SectionRewardClaim(user_id=1, folder_name=folder, section_index=index, xp_gained=100,
                              map_bonus_added=points,
                              created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)))


class MapLevels(unittest.TestCase):
    def setUp(self):
        fixture.HttpIntegrity.setUp(self)
        self.assertIn(2, map_world.LEVELS, 'map_terrain_types_l2.json must be exported')

    def bank(self, claims, pos=None):
        """Record claims (folder, index, points) in order and bank their points."""
        total = 0
        with SessionLocal() as db:
            for k, (folder, index, points) in enumerate(claims):
                claim(db, folder, index, points, minutes_ago=len(claims) - k)
                total += points
            row = db.get(UserMapState, 1) or UserMapState(user_id=1, pos_x=75, pos_y=107, pos_world='lumen')
            if pos:
                row.pos_x, row.pos_y, row.pos_world = pos
            row.bonus_unlock_points = total
            db.merge(row)
            db.commit()
        map_world.invalidate_map_cache(1)
        return total

    def test_worlds_are_exported_in_order(self):
        self.assertEqual(map_world.LEVELS[1].world, 'lumen')
        self.assertEqual(map_world.LEVELS[2].world, 'neon')
        self.assertEqual((map_world.LEVELS[1].ox, map_world.LEVELS[1].oy), (75, 107))
        self.assertEqual((map_world.LEVELS[2].ox, map_world.LEVELS[2].oy), (72, 110))

    def test_level_one_is_the_lumen_reaches(self):
        self.bank([('Physics', 0, 60), ('Physics', 1, 60)])
        state = map_world.get_map_state(1, compact=True)
        self.assertEqual(state['map_level'], 1)
        self.assertEqual(state['map_world'], 'lumen')
        self.assertEqual(state['size'], 160)
        self.assertEqual(state['origin'], LUMEN_HARBOUR)
        self.assertEqual(state['reveal_radius'], round(map_world._level_radius(1, 120), 1))
        self.assertGreater(state['reveal_radius'], map_world.LEVELS[1].clear)
        self.assertEqual(state['reveal_pacing']['mode'], 'area')
        self.assertTrue(all(':' not in key for key in state['tile_sections']))

    def test_level_two_opens_after_level_one_is_charted(self):
        # 70 claims of 60 points = 4200 (level 1 complete), then 5 more sections.
        claims = [('Physics', i, 60) for i in range(70)] + [('Chemistry', i, 60) for i in range(5)]
        self.bank(claims)
        state = map_world.get_map_state(1, compact=True)
        self.assertEqual(state['map_level'], 2)
        self.assertEqual(state['map_world'], 'neon')
        self.assertEqual(state['size'], 160)
        self.assertEqual(state['origin'], NEON_HARBOUR)
        self.assertEqual(state['level_points'], 300)
        self.assertEqual(state['reveal_pacing']['mode'], 'area')
        self.assertEqual(state['reveal_pacing']['points'], 4800)
        self.assertGreater(state['reveal_radius'], map_world.LEVELS[2].clear)
        # The player moves to the new harbour.
        self.assertEqual(state['player'], NEON_HARBOUR)
        keys = state['tile_sections']
        catalog = state['section_catalog']
        level_two = [k for k in keys if k.startswith('2:')]
        self.assertTrue(level_two, 'level 2 tiles are tagged')
        folders = {catalog[keys[k]]['folder'] for k in level_two}
        self.assertIn('Chemistry', folders)
        self.assertNotIn('Physics', folders)
        # Level 1 stays fully charted and tagged.
        self.assertEqual(len([k for k in keys if ':' not in k]), 160 * 160)

    def test_area_pacing_uncovers_a_steady_amount_per_section(self):
        for level in (1, 2):
            counts = [map_world._disc_count(level, int(round(map_world._level_radius(level, p) * 10)))
                      for p in (0, 600, 1200, 1800)]
            steps = [b - a for a, b in zip(counts, counts[1:])]
            self.assertTrue(all(abs(s - steps[0]) / steps[0] < 0.08 for s in steps), (level, steps))

    def test_crossing_into_level_two_reports_level_up(self):
        self.bank([('Physics', i, 60) for i in range(69)])  # 4140 points
        with SessionLocal() as db:
            db.add(CourseOutline(user_id=1, folder_name='Biology', current_section=0, total_sections=1,
                                 outline_json=json.dumps([{'title': 'Cells'}])))
            db.commit()
        reward = map_world.claim_section_reward(1, 'Biology', 0, section_title='Cells', section_minutes=25)
        self.assertEqual(reward['map']['map_level'], 2)
        self.assertTrue(reward['map']['level_up'])
        self.assertGreater(reward['map']['tiles_unlocked_delta'], 0)
        again = map_world.claim_section_reward(1, 'Biology', 0, section_title='Cells', section_minutes=25)
        self.assertTrue(again['already_claimed'])
        self.assertFalse(again['map']['level_up'])

    def test_chests_carry_their_world_and_need_their_tile_charted(self):
        import treasure
        self.assertTrue(all(c['id'].startswith(('lumen:', 'neon:')) for c in treasure.TREASURE_CHESTS))
        chest = next(c for c in treasure.TREASURE_CHESTS if c['id'].startswith('neon:'))
        self.assertEqual(chest['level'], 2)
        self.bank([('Physics', 0, 60)])
        self.assertFalse(map_world.is_charted(1, 2, chest['x'], chest['y']))
        with SessionLocal() as db:
            db.query(SectionRewardClaim).delete()
            db.commit()
        self.bank([('History', i, 60) for i in range(71)])
        near = min((c for c in treasure.TREASURE_CHESTS if c['level'] == 2),
                   key=lambda c: (c['x'] - 72) ** 2 + (c['y'] - 110) ** 2)
        self.assertTrue(map_world.is_charted(1, 2, near['x'], near['y']))

    def test_position_from_a_retired_world_goes_back_to_the_harbour(self):
        # A player saved on the old level 1 map (no world recorded) starts at the new harbour.
        self.bank([('Physics', 0, 60)], pos=(72, 79, None))
        state = map_world.get_map_state(1, compact=True)
        self.assertEqual(state['player'], LUMEN_HARBOUR)
        with SessionLocal() as db:
            self.assertEqual(db.get(UserMapState, 1).pos_world, 'lumen')

    def test_tags_from_a_retired_world_are_rebuilt(self):
        self.bank([('Physics', 0, 60)])
        map_world.get_map_state(1, compact=True)
        with SessionLocal() as db:  # a tag far out in the fog, as the old map would have left
            db.merge(MapTileProvenance(user_id=1, map_level=1, x=2, y=2, folder_name='Physics',
                                       section_index=0, section_title='A'))
            db.commit()
        map_world.invalidate_map_cache(1)
        state = map_world.get_map_state(1, compact=True)
        self.assertNotIn('2,2', state['tile_sections'])
        self.assertIn(f"{LUMEN_HARBOUR['x']},{LUMEN_HARBOUR['y']}", state['tile_sections'])

    def test_old_provenance_table_is_migrated(self):
        from sqlalchemy import inspect, text
        from database import engine, _run_migrations
        with engine.begin() as conn:
            conn.execute(text('DROP TABLE map_tile_provenance'))
            conn.execute(text('CREATE TABLE map_tile_provenance (user_id INTEGER, x INTEGER, y INTEGER, '
                              'folder_name VARCHAR(100), section_index INTEGER, section_title VARCHAR(255), '
                              'created_at DATETIME, PRIMARY KEY (user_id, x, y))'))
        _run_migrations()
        cols = [c['name'] for c in inspect(engine).get_columns('map_tile_provenance')]
        self.assertIn('map_level', cols)
        self.bank([('Physics', 0, 60)])
        state = map_world.get_map_state(1, compact=True)
        self.assertTrue(state['tile_sections'])
        with SessionLocal() as db:
            self.assertTrue(db.query(MapTileProvenance).filter_by(user_id=1, map_level=1).count() > 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
