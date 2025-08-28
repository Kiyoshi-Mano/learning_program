# chatbot_app.py
import re
import textwrap
from typing import List, Tuple

import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
from openai import OpenAI

# ========== 基本設定 ==========
MODEL_NAME = "gpt-5-nano"  # 必要に応じて上位へ
CSV_PATH = "data/sample_sales.csv"
APP_TITLE = "売上データ分析AIチャットボット (DuckDB + NL→SQL)"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

client = OpenAI()  # 環境変数 OPENAI_API_KEY を使用

# ========== データ読み込み ==========
@st.cache_data(show_spinner=True)
def load_sales_data() -> pd.DataFrame:
    dtypes = {
        "category": "string",
        "region": "string",
        "sales_channel": "string",
        "customer_segment": "string",
    }
    df = pd.read_csv(
        CSV_PATH,
        dtype=dtypes,
        parse_dates=["date"],
        infer_datetime_format=True,
        dayfirst=False,
    )
    # revenue が欠損/0 → units * unit_price で補完
    if "revenue" in df.columns and {"units", "unit_price"}.issubset(df.columns):
        missing = df["revenue"].isna() | (df["revenue"] == 0)
        if missing.any():
            df.loc[missing, "revenue"] = df.loc[missing, "units"] * df.loc[missing, "unit_price"]
    return df

sales_df = load_sales_data()

# DuckDB 接続
conn = duckdb.connect(database=":memory:")
conn.register("sales", sales_df)

# ========== ガードレール ==========
ALLOWED_TABLE = "sales"
ALLOWED_COLUMNS = [
    "date", "category", "units", "unit_price", "region",
    "sales_channel", "customer_segment", "revenue",
]
FORBIDDEN_KEYWORDS = [
    "ATTACH", "COPY", "CREATE", "DELETE", "DROP", "EXPORT",
    "INSERT", "LOAD", "PRAGMA", "REPLACE", "UPDATE", "ALTER",
]
SQL_COMMENT_RE = re.compile(r"--.*?$|/\*.*?\*/", re.DOTALL | re.MULTILINE)

def sanitize_sql(sql: str) -> str:
    no_comment = re.sub(SQL_COMMENT_RE, "", sql)
    no_comment = no_comment.strip().rstrip(";")
    return no_comment

def validate_sql(sql: str) -> Tuple[bool, str]:
    s = sql.strip()
    su = s.upper()
    if not (su.startswith("SELECT ") or su.startswith("WITH ")):
        return False, "SELECT/WITH で始まるSQLのみ許可します。"
    for kw in FORBIDDEN_KEYWORDS:
        if kw in su:
            return False, f"禁止キーワードを検出: {kw}"
    if " JOIN " in su:
        return False, "JOIN は許可しません（単一テーブル集計のみ）。"

    m = re.search(r"FROM\s+([a-zA-Z_][\w\.]*)", s, flags=re.IGNORECASE)
    if not m:
        return False, "FROM 句が見つかりません。"
    table = m.group(1).split(".")[-1]
    if table.lower() != ALLOWED_TABLE:
        return False, f"利用可能なテーブルは '{ALLOWED_TABLE}' のみです。"

    # カラム名の簡易チェック（厳密パーサではない）
    # 許可外識別子は警告に留める（AS別名などがあるため）
    return True, ""

def ensure_limit(sql: str, default_limit: int = 500) -> str:
    if re.search(r"\bLIMIT\b", sql, flags=re.IGNORECASE):
        return sql
    return sql + f"\nLIMIT {default_limit}"

# ========== NL→SQL ==========
def fewshot_examples() -> List[dict]:
    return [
        {"role": "user", "content": "月毎のカテゴリー別の売上を出して"},
        {"role": "assistant", "content": textwrap.dedent("""
            WITH m AS (
              SELECT strftime(date, '%Y-%m') AS month,
                     category,
                     SUM(revenue) AS total_revenue
              FROM sales
              GROUP BY 1,2
            )
            SELECT month, category, total_revenue
            FROM m
            ORDER BY month, category
        """).strip()},
        {"role": "user", "content": "チャネルごとの売上合計を見たい"},
        {"role": "assistant", "content": textwrap.dedent("""
            SELECT sales_channel,
                   SUM(revenue) AS total_revenue
            FROM sales
            GROUP BY sales_channel
            ORDER BY total_revenue DESC
        """).strip()},
        {"role": "user", "content": "地域ごとの売上の合計は？"},
        {"role": "assistant", "content": textwrap.dedent("""
            SELECT region,
                   SUM(revenue) AS total_revenue
            FROM sales
            GROUP BY region
            ORDER BY total_revenue DESC
        """).strip()},
    ]

SCHEMA_JSON = {
    "tables": [{
        "name": ALLOWED_TABLE,
        "columns": [
            {"name": c, "type": ("DATE" if c=="date" else "INTEGER" if c in ["units","unit_price","revenue"] else "TEXT")}
            for c in ALLOWED_COLUMNS
        ],
    }]
}

SYSTEM_PROMPT = f"""
あなたは DuckDB 用の SQL アシスタントです。ユーザーの自然言語の質問を **DuckDB で実行可能な安全な SQL** に変換します。
必ず以下を守る：
- 使ってよいテーブルは sales のみ。カラムは {ALLOWED_COLUMNS} のみ。
- DDL/DML（CREATE/INSERT/UPDATE/DELETE/ALTER/DROP）、PRAGMA、ATTACH、COPY、EXPORT は禁止。
- JOIN 禁止（単一テーブル集計）。
- 月次集計は strftime(date, '%Y-%m') AS month を用いる。
- 集計は SUM/COUNT/AVG/MIN/MAX を用いる。
- 返答は **SQL 文字列のみ**（説明文・自然文は返さない）。
- 比較時は必要に応じて ORDER BY を付ける。

スキーマ（JSON）:
{SCHEMA_JSON}
""".strip()

@st.cache_data(show_spinner=False)
def nl_to_sql(question: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(fewshot_examples())
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    sql = resp.choices[0].message.content.strip()
    # コードブロック除去
    if sql.startswith("```"):
        sql = re.sub(r"^```(?:sql)?\n", "", sql)
        sql = re.sub(r"\n```$", "", sql)
    return sql

# ========== 実行 & 可視化 ==========
def run_query(sql: str) -> Tuple[pd.DataFrame, str]:
    try:
        df = conn.execute(sql).df()
        return df, ""
    except Exception as e:
        return pd.DataFrame(), str(e)

def guess_chart(df: pd.DataFrame) -> Tuple[str, dict]:
    cols_lower = [c.lower() for c in df.columns]
    # 月次がある → 折れ線
    if "month" in cols_lower:
        y_candidates = [c for c in df.columns if c.lower() in ["total_revenue", "revenue", "sum"]]
        if y_candidates:
            return "line", {"x": "month", "y": y_candidates[0]}
    # 2列（カテゴリ×値）→ 棒
    if len(df.columns) == 2 and df.dtypes.iloc[0] == "object":
        return "bar", {"x": df.columns[0], "y": df.columns[1]}
    # 3列（カテゴリ×カテゴリ×値）→ グループ棒
    if len(df.columns) == 3 and df.dtypes.iloc[0] == "object" and df.dtypes.iloc[1] == "object":
        val_col = [c for c in df.columns if df[c].dtype != "object"]
        if val_col:
            return "bar_group", {"x": df.columns[0], "y": val_col[0], "color": df.columns[1]}
    return "table", {}

# ====== NEW: 100% Pythonの確定値要約（AIに依存しない）======
def summarize_result(df: pd.DataFrame, question: str) -> str:
    """結果DFだけを根拠に短く要約（最大3文）。AIに依存しないため『データがない』等は出さない。"""
    if df.empty:
        return "該当データはありませんでした。条件を見直してください。"

    # 数値列を特定
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    text_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    # 典型ケース（1~2カテゴリ + 1数値）
    if len(numeric_cols) >= 1:
        val = numeric_cols[0]
        # 上位3件
        df_sorted = df.sort_values(val, ascending=False)
        top = df_sorted.head(3)
        items = []
        for _, r in top.iterrows():
            key = " × ".join(str(r[c]) for c in text_cols) if text_cols else ""
            items.append(f"{key}: {int(r[val]):,}円" if 'revenue' in val or 'total' in val else f"{key}: {r[val]}")
        bullet = " / ".join(items)
        msg1 = f"上位値: {bullet}。"
        # 合計があれば全体感
        total = None
        try:
            total = int(df[val].sum())
        except Exception:
            pass
        msg2 = f"合計は {total:,} です。" if total is not None else ""
        return f"{msg1} {msg2}".strip()

    # 月次のみなど
    if "month" in df.columns and len(df.columns) == 1:
        return f"対象月は {', '.join(df['month'].astype(str).head(5))} … です。"
    return "集計結果を表示しました。必要に応じて条件を追加してください。"

# ========== サイドバー ==========
st.sidebar.header("データ概要")
st.sidebar.write(f"総レコード数: {len(sales_df):,}")
st.sidebar.write(f"期間: {sales_df['date'].min().date()} - {sales_df['date'].max().date()}")
st.sidebar.write("カテゴリ: " + ", ".join(sorted(map(str, sales_df["category"].dropna().unique()))))
st.sidebar.write("地域: " + ", ".join(sorted(map(str, sales_df["region"].dropna().unique()))))
st.sidebar.write("チャネル: " + ", ".join(sorted(map(str, sales_df["sales_channel"].dropna().unique()))))

# ========== チャット履歴 ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

DEFAULT_HINTS = "\n".join([
    "例) 月毎のカテゴリー別の売上",
    "例) チャネルごとの売上",
    "例) 地域ごとの売上の合計",
])

prompt = st.chat_input(f"売上データについて何でも質問してください…\n{DEFAULT_HINTS}")

# 定型フォールバック
FALLBACKS = {
    "月": """
        WITH m AS (
          SELECT strftime(date, '%Y-%m') AS month,
                 category,
                 SUM(revenue) AS total_revenue
          FROM sales
          GROUP BY 1,2
        )
        SELECT month, category, total_revenue
        FROM m
        ORDER BY month, category
    """,
    "チャネル": """
        SELECT sales_channel, SUM(revenue) AS total_revenue
        FROM sales
        GROUP BY sales_channel
        ORDER BY total_revenue DESC
    """,
    "地域": """
        SELECT region, SUM(revenue) AS total_revenue
        FROM sales
        GROUP BY region
        ORDER BY total_revenue DESC
    """,
}

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1) NL→SQL
        raw_sql = nl_to_sql(prompt)
        cleaned = sanitize_sql(raw_sql)
        valid, reason = validate_sql(cleaned)
        if not valid:
            # キーワードに応じてフォールバック
            fb = None
            for k, v in FALLBACKS.items():
                if k in prompt:
                    fb = textwrap.dedent(v).strip()
                    break
            if fb is None:
                st.warning(f"安全ではない/不正なSQLが生成されました（{reason}）。定型クエリに該当しないため、質問をもう少し具体化してください。")
                st.code(raw_sql or "(SQLなし)", language="sql")
                st.stop()
            cleaned = fb

        final_sql = ensure_limit(cleaned)
        st.caption("生成されたSQL")
        st.code(final_sql, language="sql")

        # 2) 実行
        df, err = run_query(final_sql)
        if err:
            st.error(f"SQL実行エラー: {err}")
        elif df.empty:
            st.info("該当データがありませんでした。条件を見直してください。")
        else:
            # 3) 可視化
            chart_type, args = guess_chart(df)
            if chart_type == "line":
                fig = px.line(df, x=args["x"], y=args["y"], markers=True)
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "bar":
                fig = px.bar(df, x=args["x"], y=args["y"])
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "bar_group":
                fig = px.bar(df, x=args["x"], y=args["y"], color=args["color"], barmode="group")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)

            # 4) NEW: 100% Python 要約（AI不使用）
            summary = summarize_result(df, prompt)
            st.write(summary)

    st.session_state.messages.append(
        {"role": "assistant", "content": "SQLを生成・実行し、結果を表示しました。"}
    )
