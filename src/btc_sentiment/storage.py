import json
import sqlite3
from pathlib import Path

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
 timestamp TEXT PRIMARY KEY, open REAL, high REAL, low REAL, close REAL,
 retrieved_at TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS posts (
 id TEXT PRIMARY KEY, timestamp TEXT, text TEXT, likes INTEGER,
 reposts INTEGER, views INTEGER, followers INTEGER, author_id TEXT,
 retrieved_at TEXT, raw_json TEXT);
CREATE TABLE IF NOT EXISTS coverage (
 day TEXT PRIMARY KEY, run_id TEXT, status TEXT, returned_count INTEGER,
 retained_count INTEGER, cap INTEGER, capped INTEGER, retrieved_at TEXT);
CREATE TABLE IF NOT EXISTS news (
 url TEXT PRIMARY KEY, title TEXT, content TEXT, published_at TEXT,
 retrieved_at TEXT, relevance REAL);
CREATE TABLE IF NOT EXISTS provenance (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def upsert(con, table, records):
    if not records:
        return
    if table not in {"prices", "posts", "coverage", "news", "provenance"}:
        raise ValueError("Unknown table")
    cols = list(records[0])
    allowed = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    if not set(cols) <= allowed:
        raise ValueError("Unknown column")
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})"
    with con:
        con.executemany(sql, [[r.get(k) for k in cols] for r in records])


def metadata(con, key, value):
    upsert(con, "provenance", [{"key": key, "value": json.dumps(value)}])


def read(con, table):
    if table not in {"prices", "posts", "coverage", "news", "provenance"}:
        raise ValueError("Unknown table")
    return pd.read_sql_query(f"SELECT * FROM {table}", con)
