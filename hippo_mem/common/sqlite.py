# Copyright (c) 2025 Arne Deutsch, itemis AG, MIT License
"""Small SQLite execution helper mixin."""

from __future__ import annotations

from typing import Any, Iterable, Literal, Optional

Fetch = Optional[Literal["one", "all"]]


class SQLiteExecMixin:
    """Mixin providing a thin wrapper around ``cursor.execute``."""

    conn: Any  # expected to be a ``sqlite3.Connection``

    def _exec(self, sql: str, params: Iterable = (), *, fetch: Fetch = None):
        """Execute ``sql`` with ``params`` and optionally fetch results.

        Parameters
        ----------
        sql
            SQL statement to execute.
        params
            Parameters bound to the SQL statement.
        fetch
            If ``"one"`` or ``"all"``, returns ``cursor.fetchone()`` or
            ``cursor.fetchall()`` respectively. Otherwise returns ``None``.
        """

        cur = self.conn.cursor()
        cur.execute(sql, params)
        result = None
        if fetch == "one":
            result = cur.fetchone()
        elif fetch == "all":
            result = cur.fetchall()
        self.conn.commit()
        return result


__all__ = ["SQLiteExecMixin"]
