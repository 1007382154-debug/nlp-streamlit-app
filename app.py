import streamlit as st

st.set_page_config(
    page_title="项目导航首页",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 NLP项目 - 首页导航")
st.markdown("---")
st.markdown("欢迎访问项目演示平台。以下为您部署的各个子应用模块（A1-A9）的入口：")

st.info("💡 **提示**: 如果您的 A1-A9 已经分别在 Streamlit 部署，请将下方代码中的 `#` 替换为它们各自的实际部署链接网址。")

col1, col2, col3 = st.columns(3)

# 可以在这里补充各个模块的真实描述和部署链接
apps = [
    {"id": "A1", "title": "A1 模块", "desc": "A1 应用的功能简述", "url": "https://nlp-app-app-fec9wt3drznobbcglm77wk.streamlit.app/"},
    {"id": "A2", "title": "A2 模块", "desc": "A2 应用的功能简述", "url": "https://nlp-a2-vibecoding.streamlit.app/"},
    {"id": "A3", "title": "A3 模块", "desc": "A3 应用的功能简述", "url": "https://nlp-a3-vibecoding.streamlit.app/"},
    {"id": "A4", "title": "A4 模块", "desc": "A4 应用的功能简述", "url": "https://nlp-a4-vibecoding.streamlit.app/"},
    {"id": "A5", "title": "A5 模块", "desc": "A5 应用的功能简述", "url": "https://nlp-a5-vibecoding.streamlit.app/"},
    {"id": "A6", "title": "A6 模块", "desc": "A6 应用的功能简述", "url": "https://nlp-a6-vibecoding.streamlit.app/"},
    {"id": "A7", "title": "A7 模块", "desc": "A7 应用的功能简述", "url": "https://nlp-a7-vibecoding.streamlit.app/"},
    {"id": "A8", "title": "A8 模块", "desc": "A8 应用的功能简述", "url": "https://nlp-a8-vibecoding.streamlit.app/"},
    {"id": "A9", "title": "A9 模块", "desc": "A9 应用的功能简述", "url": "https://nlp-a9-vibecoding.streamlit.app/"},
]

for i, app in enumerate(apps):
    if i % 3 == 0:
        col = col1
    elif i % 3 == 1:
        col = col2
    else:
        col = col3
        
    with col:
        st.markdown(f"""
        ### {app['title']}
        <span style="color:gray">{app['desc']}</span>
        
        [👉 **点击进入 {app['id']} 应用**]({app['url']})
        """, unsafe_allow_html=True)
        st.markdown("---")
