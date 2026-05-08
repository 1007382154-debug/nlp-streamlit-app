import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline
import torch

st.set_page_config(page_title="细粒度情感分析与舆情监测 Web 平台", page_icon="📊", layout="wide")

# --- CSS 全局样式 ---
st.markdown("""
<style>
.stApp {
    background-color: #f8f9fa;
}
.main-title {
    text-align: center;
    color: #1e3a8a;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 1rem;
}
.sub-title {
    text-align: center;
    color: #4b5563;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.expl_card {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
    border-left: 5px solid #3b82f6;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 细粒度情感分析与舆情监测平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">将自然语言评论转化为量化的情感极性，并聚合为商业决策图表</div>', unsafe_allow_html=True)

# --- 模型加载 ---
@st.cache_resource(show_spinner="正在加载 Hugging Face 情感分析模型 (大概需要1-2分钟)...")
def load_sentiment_model():
    """
    加载多语言轻量级情感分析模型
    """
    # lxyuan/distilbert-base-multilingual-cased-sentiments-student
    # 这款模型能够处理中英等多语言，输出 positive/negative/neutral
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    classifier = pipeline("sentiment-analysis", model=model_name)
    return classifier

classifier = load_sentiment_model()

# --- 标签页定义 ---
tab1, tab2, tab3 = st.tabs(["🎯 模块 1：基础分类与置信度", "🔍 模块 2：显式情感 vs 隐式情感", "📈 模块 3：舆情挖掘仪表盘"])

# ================================
# 模块 1：基础分类与置信度
# ================================
with tab1:
    st.header("基础情感分类与置信度量化")
    st.markdown("输入一段中文商品评论，体验模型对其极性（Positive/Negative/Neutral）的判定及置信度输出。")
    
    text_input = st.text_area("✍️ 请输入一小段商品评价：", "这款手机的屏幕色彩非常艳丽，拍照也很清晰，非常满意！", height=120)
    
    if st.button("🚀 开始分析", key="btn_mod1"):
        if text_input.strip():
            with st.spinner("情感分析计算中..."):
                results = classifier(text_input.strip())
                result = results[0]
                label = result['label'].capitalize()
                score = result['score']
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write("### 📌 分析结果")
                    if label == "Positive":
                        st.success(f"**极性**：{label} (正面 😊)")
                        bar_color = "green"
                    elif label == "Negative":
                        st.error(f"**极性**：{label} (负面 😡)")
                        bar_color = "red"
                    else:
                        st.warning(f"**极性**：{label} (中性 😐)")
                        bar_color = "gray"
                        
                    st.metric(label="置信度 (Confidence Score)", value=f"{score:.2%}")
                    
                with col2:
                    # 绘制漂亮的 Gauge Chart (半圆仪表盘)
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = score,
                        title = {'text': f"<b>{label}</b> 置信度"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 1]},
                            'bar': {'color': bar_color},
                            'steps' : [
                                {'range': [0, 0.4], 'color': "rgba(255, 99, 71, 0.2)"},
                                {'range': [0.4, 0.7], 'color': "rgba(255, 206, 86, 0.2)"},
                                {'range': [0.7, 1.0], 'color': "rgba(144, 238, 144, 0.2)"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.9}
                        }
                    ))
                    fig.update_layout(height=280, margin=dict(l=10, r=10, b=10, t=40))
                    st.plotly_chart(fig, use_container_width=True)
                    
            st.info("💡 **观察提示**：分类结果只表明了最可能的极性，而**置信度（概率值）**反映了模型的“确信程度”。在商业应用中，低置信度的文本可能包含复杂情绪，通常需要交由人工进行二次审核。")

# ================================
# 模块 2：显式情感 vs 隐式情感
# ================================
with tab2:
    st.header("显式情感 vs. 隐式情感识别")
    st.markdown("""
    **💡 知识科普：什么是显式与隐式情感？**
    - **显式情感**：句子中包含明显的褒贬情感词汇，如“太棒了”、“垃圾”、“难吃”、“喜欢”。
    - **隐式情感**：句子中**没有明显的情感词**，通常是客观地陈述事实，但该事实在具体场景下反映出了使用者的负面或正面态度。如“手机玩游戏半小时就没电了”（暗示续航极差），或“在太阳底下根本看不清屏幕上的字”（暗示亮度不足）。
    """)
    
    colA, colB = st.columns(2)
    with colA:
        explicit_text = st.text_area("📝 显式情感评价", "这屏幕画质太垃圾了，体验极差！", height=100)
    with colB:
        implicit_text = st.text_area("🕵️ 隐式客观描述", "在太阳底下根本看不清屏幕上的字。", height=100)
        
    if st.button("⚖️ 对比分析", key="btn_mod2"):
        r_expl = classifier(explicit_text)[0]
        r_impl = classifier(implicit_text)[0]
        
        cA, cB = st.columns(2)
        with cA:
            st.markdown('<div class="expl_card">', unsafe_allow_html=True)
            st.write(f"**模型识别极性**：{r_expl['label']}")
            st.write(f"**置信度**：{r_expl['score']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with cB:
            st.markdown('<div class="expl_card" style="border-left: 5px solid #f59e0b;">', unsafe_allow_html=True)
            st.write(f"**模型识别极性**：{r_impl['label']}")
            st.write(f"**置信度**：{r_impl['score']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.info("💡 **观察提示**：请观察模型能否准确听懂隐式陈述背后的“负面”或“正面”情绪。许多小模型在隐式情绪上容易将它判定为 'Neutral (中性)' 因为它看起来是一句陈述句。这是 NLP 的一大难点！")


# ================================
# 模块 3：舆情挖掘与可视化仪表盘
# ================================
with tab3:
    st.header("舆情挖掘与可视化仪表盘 (Opinion Mining)")
    st.markdown("将单句分析扩展到大规模语料库中，可以为产品改进、危机预警提供宏观洞察。")
    
    # 模拟生成的商业评论数据
    mock_reviews = [
        "电池续航很棒，一天充一次足够了！",
        "买回来就有划痕，品控太差了。",
        "物流还挺快的，顺丰快递第二天就到了。",
        "性价比确实高，千元机做到这样很不错了。",
        "系统经常卡顿死机，体验很不好，退货了。",
        "玩游戏一小时之后后盖有些发热。",
        "外观设计很好看，手感轻薄。",
        "客服态度傲慢，半天不回消息！",
        "拍照效果一般，夜景有很多噪点。",
        "送给长辈的，屏幕大声音响，老人很喜欢。",
        "指纹解锁偶尔不灵敏，需要多按几次。",
        "刚过保修期主板就烧了，真的服了。",
        "原装耳机音质不错，低音很震撼。",
        "总体来说在这个价位还算中规中矩吧。",
        "非常漂亮的一款手机，颜色很高级。"
    ]
    
    if st.button("📊 生成测试舆情数据并批量分析"):
        with st.spinner("正在对数据库进行批量情感分析，请稍候..."):
            # 批量推理
            batch_results = classifier(mock_reviews)
            
            # 转入 pandas DataFrame 进行汇总分析
            df = pd.DataFrame({
                "用户评论": mock_reviews,
                "所属情感极性": [r['label'].capitalize() for r in batch_results],
                "置信度": [r['score'] for r in batch_results]
            })
            
            # 统计数量
            counts = df['所属情感极性'].value_counts().reset_index()
            counts.columns = ['情感极性', '数量']
            
            color_map = {
                'Positive': '#10b981',  # 绿
                'Negative': '#ef4444',  # 红
                'Neutral': '#6b7280'    # 灰
            }
            
            st.subheader("整体口碑数据监控看板")
            
            col_data, col_chart = st.columns([1.5, 1.5])
            
            with col_data:
                st.write("**原始数据挖掘结果**：")
                st.dataframe(df, use_container_width=True)
                
            with col_chart:
                # 绘制饼图
                fig_pie = go.Figure(data=[go.Pie(
                    labels=counts['情感极性'], 
                    values=counts['数量'],
                    hole=.4, # 甜甜圈环形图
                    marker=dict(colors=[color_map.get(lbl, '#333333') for lbl in counts['情感极性']]),
                    textinfo='label+percent'
                )])
                fig_pie.update_layout(
                    title_text="全网商品口碑比例分布分析",
                    annotations=[dict(text='意见分布', x=0.5, y=0.5, font_size=16, showarrow=False)],
                    height=350,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            # 提供管理层视角的商业洞察
            st.success("""
            🎯 **AI 商业洞察**：
            从大规模语义挖掘结果可知，本次抽取数据中存在一定的红区（Negative比例）。结合原始数据，用户的核心痛点集中在“死机卡顿”、“客服傲慢”与“主板品控”。建议产品部门优先排查系统固件问题，并对客服服务体系进行告警。
            """)