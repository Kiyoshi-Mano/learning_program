import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# ページ設定
st.set_page_config(page_title="販売データBIダッシュボード", layout="wide")

# タイトル
st.title("販売データBIダッシュボード")
st.write("data/sample_sales.csv の販売実績をもとに、売上状況を可視化します。")

# データ読み込み（キャッシュ）
@st.cache_data
def load_data():
    df = pd.read_csv("data/sample_sales.csv", parse_dates=["date"])
    return df

df = load_data()

# --- 🔍 サイドバー フィルター設定 ---
st.sidebar.header("フィルター")

# 日付範囲選択
min_date = df["date"].min()
max_date = df["date"].max()
date_range = st.sidebar.date_input("日付範囲を選択", [min_date, max_date], min_value=min_date, max_value=max_date)

# 入力値が2つない場合の対応
if len(date_range) != 2:
    st.warning("日付範囲を2つ選択してください。")
    st.stop()

start_date, end_date = date_range
df_filtered = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

# データ存在確認
if df_filtered.empty:
    st.warning("該当するデータがありません。別の日付範囲を選択してください。")
    st.stop()

# --- 📊 KPI 表示 ---
total_revenue = int(df_filtered["revenue"].sum())
total_units = int(df_filtered["units"].sum())
category_count = df_filtered["category"].nunique()

st.subheader("主要指標（KPI）")

col1, col2, col3 = st.columns(3)
col1.metric("売上合計（円）", f"¥{total_revenue:,}")
col2.metric("販売数量合計", f"{total_units:,} 個")
col3.metric("商品カテゴリ数", f"{category_count} 種類")

# --- 📈 カテゴリ別売上グラフ（棒グラフ） ---
st.subheader("カテゴリ別売上（棒グラフ）")
category_revenue = df_filtered.groupby("category")["revenue"].sum().reset_index().sort_values(by="revenue", ascending=False)

fig_bar = px.bar(
    category_revenue,
    x="category",
    y="revenue",
    color="category",
    title="カテゴリ別売上",
    labels={"revenue": "売上（円）", "category": "カテゴリ"},
)
st.plotly_chart(fig_bar, use_container_width=True)

# --- 📉 日別売上推移（折れ線グラフ） ---
st.subheader("日別売上推移（折れ線グラフ）")
daily_revenue = df_filtered.groupby("date")["revenue"].sum().reset_index()

fig_line = px.line(
    daily_revenue,
    x="date",
    y="revenue",
    title="日別売上推移",
    labels={"date": "日付", "revenue": "売上（円）"},
)
st.plotly_chart(fig_line, use_container_width=True)
