#!/usr/bin/env python3
"""Full-text rows are numbered by their item's id (stores/db.fts_rowid), so writing, replacing,
re-tagging and deleting an item finds its search row at once instead of reading the whole table."""
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coast_content_oma.stores.base import MemoryItem  # noqa: E402
from coast_content_oma.stores.content import ContentStore  # noqa: E402
from coast_content_oma.stores.db import connect_db, fts_rowid, key_fts_rows  # noqa: E402
from coast_content_oma import source_lifecycle  # noqa: E402


def item(n, ns="u1__c", text=None, doc="doc_a"):
    return MemoryItem(id=f"coi_{n}", namespace=ns, store="content", content=text or f"page {n} about entropy",
                      source_doc_id=doc, entities=["entropy"], tags=["definition"], store_specific={})


class SearchRows(unittest.TestCase):
    def setUp(self):
        self.db = Path(tempfile.mkdtemp()) / "oma.db"
        self.store = ContentStore(self.db)

    def rows(self):
        with connect_db(self.db) as conn:
            return conn.execute("select rowid, id, content from content_items_fts order by id").fetchall()

    def test_each_item_has_one_search_row_numbered_by_its_id(self):
        self.store.write_items_bulk([item(1), item(2)], embed=False)
        self.store.write_items_bulk([item(1, text="page 1 rewritten about enthalpy")], embed=False)  # replaced
        self.assertEqual([(r[0], r[1]) for r in self.rows()], [(fts_rowid("coi_1"), "coi_1"), (fts_rowid("coi_2"), "coi_2")])
        self.assertIn("enthalpy", self.rows()[0][2])
        self.store.delete("coi_2")
        self.assertEqual([r[1] for r in self.rows()], ["coi_1"])

    def test_deleting_a_source_removes_its_search_rows(self):
        self.store.write_items_bulk([item(1), item(2, doc="doc_b")], embed=False)
        source_lifecycle.remove_source_material(self.db, "u1__c", "a")
        self.assertEqual([r[1] for r in self.rows()], ["coi_2"])

    def test_rows_written_before_numbering_are_numbered_once_and_found_the_same(self):
        with connect_db(self.db) as conn:  # a table from before: insertion-order numbers, a duplicate
            conn.execute("delete from fts_keyed")
            for i, n in enumerate((1, 2, 1)):
                conn.execute("insert into content_items_fts (id, namespace, content, entities, tags) values (?,?,?,?,?)",
                             (f"coi_{n}", "u1__c", f"version {i} of page {n} entropy", "", ""))
        with connect_db(self.db) as conn:
            key_fts_rows(conn, "content_items", "content_items_fts")
        rows = self.rows()
        self.assertEqual([(r[0], r[1]) for r in rows], [(fts_rowid("coi_1"), "coi_1"), (fts_rowid("coi_2"), "coi_2")])
        self.assertIn("version 2", rows[0][2])  # the later of the two copies
        with connect_db(self.db) as conn:
            key_fts_rows(conn, "content_items", "content_items_fts")  # done once only
            matches = {r[0] for r in conn.execute("select id from content_items_fts where content_items_fts match 'entropy'")}
        self.assertEqual(matches, {"coi_1", "coi_2"})


if __name__ == "__main__":
    unittest.main(verbosity=1)
