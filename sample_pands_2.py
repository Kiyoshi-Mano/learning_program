fig = px.bar(
    category_revenue,
    x='revenue',            # ← 横軸：売上
    y='category',           # ← 縦軸：カテゴリ
    color='category',       # ← 色分け（虹色）
    orientation='h',        # ← 横向きバーにする
    title='商品カテゴリごとの総売上（横棒グラフ）',
    labels={'category': '商品カテゴリ', 'revenue': '総売上 (円)'},
    color_discrete_sequence=px.colors.qualitative.Bold
)
import streamlit as st
import pandas as pd
import plotly.express as px

# アプリのタイトルと説明
st.title('Plotly基礎')
st.write('Plotlyを使ってインタラクティブなグラフを作成してみましょう！')

# CSVファイルを読み込む
df = pd.read_csv('data/sample_sales.csv')

st.subheader('カテゴリ別合計売上グラフ')

# Pandasでカテゴリごとの合計売上を計算
# 1. 'category' 列でデータをグループ化します。
# 2. グループごとに'revenue'列の合計値（sum）を計算します。
# 3. reset_index()でカテゴリを通常の列に戻します。
category_revenue = df.groupby('category')['revenue'].sum().reset_index()

# Plotlyで棒グラフを作成
# x軸にカテゴリ、y軸に合計売上を設定します。
fig = px.bar(
    category_revenue,
    x='revenue',            # ← 横軸：売上
    y='category',           # ← 縦軸：カテゴリ
    color='category',       # ← 色分け（虹色）
    orientation='h',        # ← 横向きバーにする
    title='商品カテゴリごとの総売上（横棒グラフ）',
    labels={'category': '商品カテゴリ', 'revenue': '総売上 (円)'},
    color_discrete_sequence=px.colors.qualitative.Bold
)


# Streamlitにグラフを表示
# st.plotly_chart()を使うと、Plotlyで作成したグラフをStreamlitアプリに埋め込めます。
st.plotly_chart(fig)

st.write('---')
st.write('このグラフはインタラクティブです！特定のカテゴリにカーソルを合わせると、そのカテゴリの正確な総売上が表示されます。')

