"""Identifier-format checks for SQL input joins."""
from pathlib import Path
import tempfile
import unittest

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from atlas.identifiers import paper_id_sql


class IdentifierJoinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = duckdb.connect(':memory:')
        self.addCleanup(self.db.close)

    def table(self, name, values, column='paper_id'):
        path = self.root / (name + '.parquet')
        pq.write_table(pa.table({column: values}), path)
        expression = paper_id_sql(path, column, allow_float=True)
        quoted = str(path).replace("'", "''")
        return path, f"SELECT {expression} AS id FROM read_parquet('{quoted}')"

    def test_integer_prefixed_and_url_sources_join(self):
        _, edges = self.table('edges', [1, 2, 3])
        _, core = self.table('core', ['W1', 'https://openalex.org/W2', '3'])
        _, scores = self.table('scores', ['1', 'W2', 'http://openalex.org/W3'])
        query = f'WITH e AS ({edges}), c AS ({core}), s AS ({scores}) '
        query += 'SELECT e.id FROM e JOIN c USING(id) JOIN s USING(id) ORDER BY e.id'
        self.assertEqual(self.db.execute(query).fetchall(), [(1,), (2,), (3,)])

    def test_exact_float_input_and_strict_edge_type(self):
        path, query = self.table('numeric', [1.0, 2.0])
        self.assertEqual(self.db.execute(query).fetchall(), [(1,), (2,)])
        with self.assertRaises(ValueError):
            paper_id_sql(path, 'paper_id')

    def test_unsafe_float_identifier_is_rejected(self):
        _, query = self.table('unsafe', [float(2**53)])
        with self.assertRaises(duckdb.Error):
            self.db.execute(query).fetchall()

    def test_quoted_column_name(self):
        _, query = self.table('quoted', ['W7'], column='paper"id')
        self.assertEqual(self.db.execute(query).fetchall(), [(7,)])


if __name__ == '__main__':
    unittest.main()
