import streamlit as st
import pandas as pd
import plotly.express as px

# アプリのタイトルと説明
st.title('Plotly基礎')
st.write('Plotlyを使ってインタラクティブなグラフを作成してみましょう！')

# CSVファイルを読み込む
df = pd.read_csv('data/sample_sales.csv')

# 日付列をdatetime型に変換（もしまだなら）
df['date'] = pd.to_datetime(df['date'])

st.subheader('日別売上推移グラフ')

# 日毎の売上合計を集計
daily_revenue = df.groupby('date')['revenue'].sum().reset_index()

# 折れ線グラフを作成（線の色は赤）
fig = px.line(
    daily_revenue,
    x='date',
    y='revenue',
    title='日別売上推移',
    labels={'date': '日付', 'revenue': '総売上 (円)'},
    line_shape='linear'  # 折れ線の形状（デフォルトでOK）
)

# 線の色を赤に指定
fig.update_traces(line=dict(color='red'))

# グラフを表示
st.plotly_chart(fig)

st.write('---')
st.write('この折れ線グラフは、日別の売上推移を示しています。日付にカーソルを合わせると、その日の売上合計が表示されます。')
