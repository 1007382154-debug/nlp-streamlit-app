import os
import sys

# 必须在导入任何带有 protobuf 或 streamlit 的包之前设置这些环境变量，并且强制移除以前导入过的 pb2
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
if 'google.protobuf' in sys.modules:
    del sys.modules['google.protobuf']

# --- 核心修复：绕过 Windows 中文用户名导致的 SentencePiece C++ 底层路径加载崩溃 ---
# 因为您的电脑用户名为 "王旭东"，SentencePiece 底层 C++ 代码在读取包含中文路径的文件时会报 OSErrer，
# 因此我们强行将 HuggingFace 缓存和 NLTK 下载目录挂载到一个合法的纯英文目录下。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if os.name == 'nt':  # Windows 系统下
    os.environ["HF_HOME"] = "D:\\nlp_models_cache\\huggingface"
    os.environ["NLTK_DATA"] = "D:\\nlp_models_cache\\nltk_data"
else:  # Linux (Streamlit Cloud 云端环境) 下
    os.environ["HF_HOME"] = "/tmp/huggingface"
    os.environ["NLTK_DATA"] = "/tmp/nltk_data"

import streamlit as st
import spacy
import benepar
import nltk
import svgling
from spacy import displacy
from nltk.tree import Tree
import os

# 必须是在脚本第一句 Streamlit API 调用之前执行
st.set_page_config(page_title="句法分析树状工具", layout="wide")

# ---------- 核心模型加载逻辑 ----------
@st.cache_resource(show_spinner=True)
def init_nlp_models():
    """
    负责检查、下载并加载 SpaCy 与 Benepar 模型。
    使用 st.cache_resource 确保页面刷新并不会重复加载模型，优化体验。
    """
    # 1. 自动检查/下载 Spacy 英文核心模型 (en_core_web_sm)
    try:
        spacy.util.get_package_path('en_core_web_sm')
    except (IOError, ImportError, ModuleNotFoundError, OSError):
        st.warning("🔄 首次运行系统检测到缺失 en_core_web_sm 模型，正在为您自动下载，请稍候...")
        from spacy.cli import download as spacy_download
        spacy_download('en_core_web_sm')
        
    # 2. 自动检查/下载 NLTK 组件
    nltk_data_path = os.environ.get("NLTK_DATA")
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)
        
    # 3. 自动检查/下载 Benepar 成分分析模型
    try:
        nltk.data.find('models/benepar_en3')
    except LookupError:
        st.warning("🔄 检测到缺失 benepar_en3 模型，正在为您自动下载至 D:\\nlp_models_cache 下 (100M+)，请稍候...")
        benepar.download('benepar_en3', download_dir=nltk_data_path)

    # 加载已有的 SpaCy 语言模型
    nlp = spacy.load('en_core_web_sm')
    
    # 极客配置：将 benepar 集成到 SpaCy 的 pipeline 里面
    # 这样一句话经过 nlp() 后，除了原本 SpaCy 的依存句法外，还会附加成分分析的特征
    if "benepar" not in nlp.pipe_names:
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        
    return nlp

# ---------- 主页面 UI 设计 ----------
st.title("🌲 NLP：依存与成分句法分析可视化")
st.markdown("解析经典的英语结构歧义（如介词短语依附限制歧义）非常有意思，默认的句子就是一个著名的自然语言处理难题挑战案例。")

# 页面顶部的句子输入框
user_input = st.text_input(
    "请输入英语句子进行快速解析：",
    value="The boy saw the man with the telescope."
)

if user_input:
    # 获取 NLP 实例
    nlp = init_nlp_models()
    
    # NLP管道处理
    doc = nlp(user_input)
    
    # 使用 st.tabs 将空间划分为 依存关系 和 成分结构 两个独立空间，杜绝拥挤感
    tab_dep, tab_con = st.tabs(["📏 依存关系 (Dependency Parsing)", "🌳 成分结构 (Constituency Parsing)"])
    
    # ---- 标签页 1：依存句法树 (SpaCy) ----
    with tab_dep:
        st.subheader("通过 SpaCy 提取与 DisplaCy 生成的依存树")
        # style="dep" 表示 Dependency Parsing
        # 设置 distance 使单词彼此距离放大，防止拥挤
        html_dep = displacy.render(doc, style="dep", jupyter=False, options={"distance": 130})
        # 解析出的 HTML 和 SVG 图片会由 st.components 原生展示
        st.components.v1.html(html_dep, height=500, scrolling=True)

    # ---- 标签页 2：成分句法树 (Benepar + NLTK + SVG) ----
    with tab_con:
        st.subheader("通过 Berkeley Neural Parser 生成的成分树")
        
        # doc 中可能包含多个句子，为了保险我们针对每个句子进行迭代渲染
        for idx, sent in enumerate(doc.sents):
            st.markdown(f"**第 {idx + 1} 句:** `{sent.text}`")
            
            # 由于前面的 pipeline 配置，这里的句子实例 sent 可以直接调用._.parse_string 取出成分分析的结果(LISP括号风格式)
            parse_string = sent._.parse_string
            
            # 将其转换成 NLTK Tree 对象以便进一步处理
            tree_obj = Tree.fromstring(parse_string)
            
            # 将树绘制为 svgling 对象，进而能够以精美的 SVG 格式导出
            svg_drawer = svgling.draw_tree(tree_obj)
            svg_content = svg_drawer._repr_svg_()
            
            # 使用 html 包裹将其显示到 st 网页上
            st.components.v1.html(svg_content, height=450, scrolling=True)
            
            # 同时保留文本层级结构的视图，隐藏在折叠组件内
            with st.expander("📝 点击查看纯文本缩进层级树 (代码形式)"):
                st.code(tree_obj.pformat(margin=60), language="text")
