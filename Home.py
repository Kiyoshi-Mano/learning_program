import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Streamlit BI x Claude Code Starter", layout="wide")

st.title("Streamlit BI x Claude Code Starter")
@st.cache_data
def load_data():
    orders_df = pd.read_csv("sample_data/orders.csv")
    users_df = pd.read_csv("sample_data/users.csv")
    return orders_df, users_df

@st.cache_data
def preprocess_orders_data(orders_df):
    orders_df = orders_df.copy()
    orders_df['created_at'] = pd.to_datetime(orders_df['created_at'], errors='coerce')
    orders_df = orders_df.dropna(subset=['created_at'])
    orders_df['year_month'] = orders_df['created_at'].dt.to_period('M')
    orders_df['status'] = orders_df['status'].str.strip().str.title()
    return orders_df

@st.cache_data
def calculate_monthly_metrics(orders_df):
    monthly_stats = orders_df.groupby('year_month').agg({
        'order_id': 'count',
        'status': lambda x: (x == 'Cancelled').sum()
    }).rename(columns={
        'order_id': 'total_orders',
        'status': 'cancelled_orders'
    })
    
    monthly_stats['cancellation_rate'] = (
        monthly_stats['cancelled_orders'] / monthly_stats['total_orders'] * 100
    )
    
    monthly_stats.index = monthly_stats.index.astype(str)
    return monthly_stats

def create_monthly_orders_chart(monthly_stats):
    fig = px.bar(
        x=monthly_stats.index,
        y=monthly_stats['total_orders'],
        title='月別オーダー数',
        labels={'x': '年月', 'y': 'オーダー数'}
    )
    fig.update_layout(
        xaxis_title='年月',
        yaxis_title='オーダー数',
        showlegend=False
    )
    return fig

def create_monthly_cancellation_chart(monthly_stats):
    fig = px.line(
        x=monthly_stats.index,
        y=monthly_stats['cancellation_rate'],
        title='月別キャンセル率',
        labels={'x': '年月', 'y': 'キャンセル率 (%)'},
        markers=True
    )
    fig.update_layout(
        xaxis_title='年月',
        yaxis_title='キャンセル率 (%)',
        showlegend=False
    )
    fig.update_traces(line_color='red', marker_color='red')
    return fig

orders_df, users_df = load_data()

st.header("📊 月別オーダー分析")

if not orders_df.empty:
    processed_orders = preprocess_orders_data(orders_df)
    
    if not processed_orders.empty:
        monthly_stats = calculate_monthly_metrics(processed_orders)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総オーダー数", f"{monthly_stats['total_orders'].sum():,}")
        with col2:
            st.metric("平均キャンセル率", f"{monthly_stats['cancellation_rate'].mean():.1f}%")
        with col3:
            st.metric("分析期間", f"{len(monthly_stats)}ヶ月")
        
        col1, col2 = st.columns(2)
        
        with col1:
            orders_chart = create_monthly_orders_chart(monthly_stats)
            st.plotly_chart(orders_chart, use_container_width=True)
        
        with col2:
            cancellation_chart = create_monthly_cancellation_chart(monthly_stats)
            st.plotly_chart(cancellation_chart, use_container_width=True)
        
        st.subheader("📈 月別統計詳細")
        st.dataframe(monthly_stats.round(2), use_container_width=True)
    else:
        st.error("有効な注文データがありません")
else:
    st.error("注文データを読み込めませんでした")

st.header("Orders Data (Top 10 rows)")
st.dataframe(orders_df.head(10))

st.header("Users Data (Top 10 rows)")
st.dataframe(users_df.head(10))