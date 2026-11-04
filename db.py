import os
import re
import pymysql
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()


# 数据库连接（从 .env 读取，带默认值）
def _conn():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "langgraph_demo"),
        charset="utf8mb4",
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )


# 危险语句黑名单：只允许查询
_BLOCK_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "SET", "ATTACH", "PRAGMA",
)


def _check_readonly(stmt: str):
    """校验语句是否只读 SELECT；不合法则返回原因，合法返回 None。"""
    if not stmt:
        return "SQL 为空"
    if not (stmt.upper().startswith("SELECT") or stmt.upper().startswith("WITH")):
        return "仅允许 SELECT 查询，请检查你的 SQL"
    for kw in _BLOCK_KEYWORDS:
        if re.search(rf"\b{kw}\b", stmt, re.IGNORECASE):
            return f"检测到禁止的关键字 {kw}，只能执行只读 SELECT 查询"
    return None


def _format_rows(rows, limit=50):
    if not rows:
        return "查询无结果（0 行）"
    truncated = len(rows) > limit
    rows = rows[:limit]

    cols = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)

    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join("" if r.get(c) is None else str(r[c]) for c in cols))

    body = "\n".join(lines)
    if truncated:
        body += f"\n...仅显示前 {limit} 行，共 {len(rows)} 行"
    return body


@tool
def list_tables() -> str:
    """列出当前数据库中所有的表，包括表名和注释，帮助你了解有哪些数据对象。

    在编写 SQL 之前，先调用本工具确认要查询的表名是否真实存在、
    名称拼写是否正确。表的注释（table_comment）会说明该表存什么。
    """
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET SESSION TRANSACTION READ ONLY")
            cur.execute(
                "SELECT table_name AS 表名, table_comment AS 注释"
                " FROM information_schema.tables"
                " WHERE table_schema = DATABASE()"
                " ORDER BY table_name"
            )
            rows = cur.fetchall()
        if not rows:
            return "当前数据库中没有表"
        return "\n".join(f"{r['表名']}\t{r['注释']}" for r in rows)
    except Exception as e:
        return f"获取表清单失败：{e}"
    finally:
        conn.close()


@tool
def get_table_schema(table_name: str) -> str:
    """获取指定表的字段结构（列名、类型、注释）以及外键关系。

    Args:
        table_name: 表名（用 list_tables 确认过的真实表名）

    返回该表的：
    - 字段列表：列名、类型、注释（注释为空则省略说明）
    - 外键关系：本表的某列 引用 -> 目标表.目标列（用于正确写 JOIN）
    """
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET SESSION TRANSACTION READ ONLY")
            cur.execute(
                "SELECT column_name AS col, column_type AS type, column_comment AS comment"
                " FROM information_schema.columns"
                " WHERE table_schema = DATABASE() AND table_name = %s"
                " ORDER BY ordinal_position",
                (table_name,),
            )
            cols = cur.fetchall()
            if not cols:
                return f"表「{table_name}」不存在或没有字段，请先用 list_tables 确认表名"

            cur.execute(
                "SELECT column_name AS col, referenced_table_name AS ref_tbl,"
                "       referenced_column_name AS ref_col"
                " FROM information_schema.key_column_usage"
                " WHERE table_schema = DATABASE() AND table_name = %s"
                "   AND referenced_table_name IS NOT NULL",
                (table_name,),
            )
            fks = cur.fetchall()
        # 字段
        lines = [f"表 {table_name} 的字段："]
        for c in cols:
            extra = f" 注释: {c['comment']}" if c["comment"] else ""
            lines.append(f"  - {c['col']}  {c['type']}{extra}")
        # 外键
        if fks:
            lines.append("外键关系：")
            for f in fks:
                lines.append(f"  - {f['col']}  ->  {f['ref_tbl']}.{f['ref_col']}")
        return "\n".join(lines)
    except Exception as e:
        return f"获取表结构失败：{e}"
    finally:
        conn.close()


@tool
def query_database(sql_query: str) -> str:
    """在数据库中执行一条只读 SELECT 查询并返回结果文本，供你向用户作答。

    执行前请先看 mysql 中对应表的结构：先用 list_tables 确认表名，
    再用 get_table_schema 看字段和外键，再写本工具的 SQL。
    若查询出错，我会返回错误原因，你可据此修正 SQL 后重试。

    Args:
        sql_query: 一条 SELECT 语句（勿以分号结尾）
    """
    stmt = sql_query.strip().rstrip(";").strip()
    err = _check_readonly(stmt)
    if err:
        return err

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET SESSION TRANSACTION READ ONLY")
            cur.execute(stmt)
            rows = cur.fetchall()
        return _format_rows(rows)
    except Exception as e:
        # 把错误返回给模型，让它自行修正 SQL 后重试
        return f"SQL 执行出错：{e}"
    finally:
        conn.close()