import streamlit as st
import nltk
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import html
import torch
from transformers import BertTokenizer, BertModel
import spacy
from spacy import displacy
import pandas as pd
import torch.nn.functional as F

# 下载 NLTK 数据 (初次运行需要)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

# 页面配置
st.set_page_config(page_title="NLP Semantic Analysis System", layout="wide", page_icon="🤖")

# 自定义 CSS 美化全局样式
st.markdown("""
    <style>
    /* 主标题与副标题美化 */
    .main-title {
        font-size: 2.8rem;
        color: #1E3A8A;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-title {
        text-align: center;
        color: #4B5563;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    /* 优化按钮，使其更动感 */
    .stButton > button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        color: white;
    }
    /* Tab 标签页优化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.15rem;
        font-weight: 600;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 交互式智能语义分析系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">深入解析语言深层结构：WSD 词义消歧 与 SRL 角色提取引擎</div>', unsafe_allow_html=True)
st.divider()

# 缓存加载模型，避免每次刷新重新加载
@st.cache_resource
def load_models():
    # BERT 模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    
    # SpaCy 模型
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
    return tokenizer, model, nlp

tokenizer, bert_model, nlp = load_models()

if "wsd_report" not in st.session_state:
    st.session_state.wsd_report = None
if "srl_report" not in st.session_state:
    st.session_state.srl_report = None

# 获取 BERT 上下文词向量的辅助函数
def get_bert_embedding(sentence, target_word, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取最后一层的隐状态
    hidden_states = outputs.last_hidden_state[0] # [seq_len, hidden_size]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # 查找目标词的索引 (这里做简单的精确匹配或子串匹配)
    target_idx = -1
    for i, token in enumerate(tokens):
        if target_word.lower() in token.lower():
            target_idx = i
            break
            
    if target_idx != -1:
        return hidden_states[target_idx]
    else:
        return None


def get_most_frequent_synset(word):
    synsets = wn.synsets(word)
    if synsets:
        return synsets[0]
    return None


def extract_srl_method_a(doc):
    result = {
        "A0 (施事者)": "-",
        "Predicate (谓词)": "-",
        "A1 (受事者)": "-",
        "AM-LOC (地点)": "-",
        "AM-TMP (时间)": "-",
    }

    for token in doc:
        if token.pos_ == "VERB" or token.dep_ == "ROOT":
            if result["Predicate (谓词)"] == "-":
                result["Predicate (谓词)"] = token.text
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        result["A0 (施事者)"] = "".join([w.text_with_ws for w in child.subtree]).strip()
                    elif child.dep_ in ["dobj", "obj", "pobj"]:
                        result["A1 (受事者)"] = "".join([w.text_with_ws for w in child.subtree]).strip()

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"]:
            result["AM-LOC (地点)"] = ent.text
        elif ent.label_ in ["DATE", "TIME"]:
            result["AM-TMP (时间)"] = ent.text

    return result


def extract_srl_method_b(doc):
    result = {
        "A0 (施事者)": "-",
        "Predicate (谓词)": "-",
        "A1 (受事者)": "-",
        "AM-LOC (地点)": "-",
        "AM-TMP (时间)": "-",
    }

    predicate = None
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"]:
            predicate = token
            break
    if predicate is None:
        for token in doc:
            if token.pos_ == "VERB":
                predicate = token
                break

    if predicate is not None:
        result["Predicate (谓词)"] = predicate.lemma_

        subj_candidates = [
            tok for tok in doc if tok.dep_ in ["nsubj", "nsubjpass", "csubj"] and tok.head == predicate
        ]
        if subj_candidates:
            subj = subj_candidates[0]
            result["A0 (施事者)"] = "".join([w.text_with_ws for w in subj.subtree]).strip()

        obj_candidates = [
            tok for tok in doc if tok.dep_ in ["dobj", "obj", "attr", "oprd"] and tok.head == predicate
        ]
        if obj_candidates:
            obj = obj_candidates[0]
            result["A1 (受事者)"] = "".join([w.text_with_ws for w in obj.subtree]).strip()

        loc_preps = {"in", "on", "at", "near", "by", "under", "over", "inside", "outside"}
        time_preps = {"during", "before", "after", "since", "until", "within"}

        for child in predicate.children:
            if child.dep_ == "prep":
                prep_phrase = "".join([w.text_with_ws for w in child.subtree]).strip()
                if child.text.lower() in loc_preps and result["AM-LOC (地点)"] == "-":
                    result["AM-LOC (地点)"] = prep_phrase
                if child.text.lower() in time_preps and result["AM-TMP (时间)"] == "-":
                    result["AM-TMP (时间)"] = prep_phrase

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"] and result["AM-LOC (地点)"] == "-":
            result["AM-LOC (地点)"] = ent.text
        elif ent.label_ in ["DATE", "TIME"] and result["AM-TMP (时间)"] == "-":
            result["AM-TMP (时间)"] = ent.text

    return result


def build_full_report_html(wsd_report, srl_report):
    def render_coverage_bars(items):
        if not items:
            return "<p>无覆盖统计数据</p>"
        max_value = max([row.get("已识别角色数", 0) for row in items] + [1])
        blocks = []
        for row in items:
            label = html.escape(str(row.get("方法", "未知方法")))
            value = int(row.get("已识别角色数", 0))
            width = int((value / max_value) * 100)
            blocks.append(
                f"""
                <div class=\"bar-item\">
                    <div class=\"bar-meta\"><span>{label}</span><span>{value}</span></div>
                    <div class=\"bar-track\"><div class=\"bar-fill\" style=\"width:{width}%\"></div></div>
                </div>
                """
            )
        return "".join(blocks)

    if wsd_report:
        lesk_synset = html.escape(wsd_report.get("lesk_synset") or "未找到")
        lesk_definition = html.escape(wsd_report.get("lesk_definition") or "未找到")
        mfs_synset = html.escape(wsd_report.get("mfs_synset") or "未找到")
        mfs_definition = html.escape(wsd_report.get("mfs_definition") or "未找到")
        if wsd_report.get("cos_sim") is not None:
            sim_text = f"{wsd_report['cos_sim']:.5f}"
        else:
            sim_text = "未计算"
        wsd_section = f"""
        <section class=\"module-card\">
            <h2>模块 1：词义消歧 (WSD)</h2>
            <p class=\"section-tip\">多方法对比：Lesk vs MFS vs BERT</p>
            <div class=\"input-panel\">
                <p><strong>句子 1：</strong> {html.escape(wsd_report.get('sent1', ''))}</p>
                <p><strong>目标词：</strong> {html.escape(wsd_report.get('target_word', ''))}</p>
                <p><strong>句子 2：</strong> {html.escape(wsd_report.get('sent2', ''))}</p>
            </div>
            <div class=\"metric-row\">
                <div class=\"metric-box\"><span class=\"metric-title\">BERT 余弦相似度</span><span class=\"metric-value\">{sim_text}</span></div>
            </div>
            <table>
                <tr><th>方法</th><th>Synset</th><th>Definition</th></tr>
                <tr><td>Lesk</td><td>{lesk_synset}</td><td>{lesk_definition}</td></tr>
                <tr><td>MFS</td><td>{mfs_synset}</td><td>{mfs_definition}</td></tr>
            </table>
        </section>
        """
    else:
        wsd_section = """
        <section class="module-card">
            <h2>模块 1：词义消歧 (WSD)</h2>
            <p>未检测到 WSD 分析结果，请先在页面中运行“模块 1”。</p>
        </section>
        """

    if srl_report:
        coverage_bars = render_coverage_bars(srl_report.get("coverage_rows"))
        srl_section = f"""
        <section class=\"module-card\">
            <h2>模块 2：语义角色标注 (SRL)</h2>
            <p class=\"section-tip\">多方法角色抽取 + 可视化</p>
            <div class=\"input-panel\"><p><strong>分析句子：</strong> {html.escape(srl_report.get('sentence', ''))}</p></div>
            <h3>方法对比结果</h3>
            {srl_report.get('compare_table_html', '')}
            <h3>角色覆盖统计（网页同款条形图风格）</h3>
            <div class=\"bar-wrap\">{coverage_bars}</div>
            {srl_report.get('coverage_table_html', '')}
            <h3>角色识别矩阵</h3>
            {srl_report.get('matrix_table_html', '')}
            <h3>依存句法树</h3>
            <div class=\"tree-container\">{srl_report.get('tree_html', '')}</div>
        </section>
        """
    else:
        srl_section = """
        <section class="module-card">
            <h2>模块 2：语义角色标注 (SRL)</h2>
            <p>未检测到 SRL 分析结果，请先在页面中运行“模块 2”。</p>
        </section>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset=\"utf-8\">
        <title>NLP 语义分析综合报告</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
                margin: 0;
                background: linear-gradient(180deg, #f7fbff 0%, #eef4ff 100%);
                color: #1f2937;
                line-height: 1.6;
            }}
            .page {{ max-width: 1200px; margin: 0 auto; padding: 28px 28px 40px; }}
            .header {{ text-align: center; margin-bottom: 18px; }}
            .main-title {{ font-size: 2.5rem; color: #1E3A8A; font-weight: 800; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.08); }}
            .sub-title {{ margin: 8px 0 0; color: #4B5563; font-size: 1.05rem; font-weight: 500; }}
            .hint {{ color: #64748b; text-align: center; margin-top: 8px; }}
            .module-card {{
                background: #ffffff;
                border: 1px solid #dbe7ff;
                border-radius: 14px;
                padding: 20px;
                margin-top: 18px;
                box-shadow: 0 10px 24px rgba(30, 58, 138, 0.08);
            }}
            h2 {{ color: #0068c9; margin: 0 0 8px; }}
            h3 {{ color: #334155; margin-top: 18px; }}
            .section-tip {{ color: #64748b; margin-top: 0; }}
            .input-panel {{ background: #f8fbff; border: 1px solid #dbeafe; border-radius: 10px; padding: 10px 12px; }}
            .metric-row {{ margin: 12px 0; }}
            .metric-box {{ display: inline-block; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 10px 14px; }}
            .metric-title {{ display: block; color: #1e40af; font-size: 0.9rem; }}
            .metric-value {{ display: block; color: #1d4ed8; font-size: 1.5rem; font-weight: 800; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; background: #fff; }}
            th, td {{ border: 1px solid #d1d5db; padding: 10px 12px; text-align: left; vertical-align: top; }}
            th {{ background-color: #f3f7ff; font-weight: 700; color: #334155; }}
            .bar-wrap {{ margin: 10px 0 6px; }}
            .bar-item {{ margin-bottom: 10px; }}
            .bar-meta {{ display: flex; justify-content: space-between; font-size: 0.92rem; color: #334155; margin-bottom: 4px; }}
            .bar-track {{ width: 100%; height: 12px; background: #e5edff; border-radius: 999px; overflow: hidden; }}
            .bar-fill {{ height: 100%; background: linear-gradient(90deg, #3B82F6, #1D4ED8); }}
            .tree-container {{ overflow-x: auto; border: 1px solid #dbeafe; padding: 18px; border-radius: 8px; background: #fff; }}
        </style>
    </head>
    <body>
        <div class="page">
            <div class="header">
                <h1 class="main-title">交互式智能语义分析系统</h1>
                <p class="sub-title">深入解析语言深层结构：WSD 词义消歧 与 SRL 角色提取引擎</p>
                <p class="hint">静态导出版：视觉风格与网页端保持同步；交互能力以在线网页为准。</p>
            </div>
            {wsd_section}
            {srl_section}
        </div>
    </body>
    </html>
    """

# 创建两个标签页
tab1, tab2 = st.tabs(["模块 1：词义消歧 (WSD)", "模块 2：语义角色标注 (SRL)"])

with tab1:
    st.markdown("### 🧩 词义消歧 (WSD) 算法对比实验")
    st.info("测试场景：对比 **Lesk**、**Most Frequent Sense (MFS)** 与 **BERT上下文向量** 在复杂语义中的表现差异。", icon="💡")
    
    with st.container():
        st.markdown("#### 📥 数据输入面板")
        col1, col2 = st.columns(2)
        with col1:
            sent1 = st.text_input("📝 句子 1：供Lesk与BERT分析 (需包含多义词)", value="I went to the bank to deposit my money.")
            target_word = st.text_input("🎯 指定目标多义词", value="bank", help="请输入该词的原形")
        with col2:
            sent2 = st.text_input("📝 句子 2：仅供BERT对比不同上下文语义", value="I sat by the river bank.")
            
    st.markdown("<br>", unsafe_allow_html=True)
    wsd_btn = st.button("🚀 开始 WSD 智能化分析", use_container_width=True)
    
    if wsd_btn:
        st.divider()
        lesk_synset_name = None
        lesk_synset_definition = None
        mfs_synset_name = None
        mfs_synset_definition = None
        cos_sim = None
        tokens1 = word_tokenize(sent1)
        lesk_synset = lesk(tokens1, target_word)
        mfs_synset = get_most_frequent_synset(target_word)
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.markdown("### 🏛️ 方法 1: 传统计算语义 (Lesk 算法)")
            st.caption("基于 WordNet 字典释义的最大重叠度进行推理")
            if lesk_synset:
                lesk_synset_name = lesk_synset.name()
                lesk_synset_definition = lesk_synset.definition()
                st.success(f"📌 **预测词典语义 (Synset):** `{lesk_synset.name()}`")
                st.info(f"📖 **标准英文释义 (Definition):** \n\n _{lesk_synset.definition()}_")
            else:
                st.warning("⚠️ Lesk 算法未在知识库中找到匹配的词汇。")

        with res_col2:
            st.markdown("### 📚 方法 2: MFS 基线")
            st.caption("直接选择 WordNet 中最常见词义，作为传统强基线")
            if mfs_synset:
                mfs_synset_name = mfs_synset.name()
                mfs_synset_definition = mfs_synset.definition()
                st.success(f"📌 **MFS词义 (Synset):** `{mfs_synset.name()}`")
                st.info(f"📖 **标准英文释义 (Definition):** \n\n _{mfs_synset.definition()}_")
            else:
                st.warning("⚠️ MFS 未找到可用词义。")
                
        with res_col3:
            st.markdown("### 🧬 方法 3: 深度学习 (BERT 向量)")
            st.caption("动态计算动态语境下隐藏层特征向量表示")
            
            with st.spinner("⏳ Transformer 计算中..."):
                vec1 = get_bert_embedding(sent1, target_word, tokenizer, bert_model)
                vec2 = get_bert_embedding(sent2, target_word, tokenizer, bert_model)
            
            if vec1 is not None and vec2 is not None:
                # 计算余弦相似度
                cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
                st.success("✅ 成功提取目标词双重动态环境向量")
                
                # 绘制漂亮的 Metric
                st.metric(label="✨ 「上下文向量」余弦相似度 (Cosine Sim)", value=f"{cos_sim:.5f}")
                
                if cos_sim < 0.5:
                    st.error(f"**诊断结论**：相似度极低 `{cos_sim:.2f}` \n\n 🤯 BERT成功推断出『{target_word}』在两句中处于完全割裂的语义空间（多义性体现）。", icon="🚨")
                else:
                    st.info(f"**诊断结论**：相似度较高 `{cos_sim:.2f}` \n\n 🤝 BERT认为其在两句中的上下文语用功能十分相近。", icon="✅")
            else:
                st.error(f"❌ 分词后维度无法匹配目标词 『{target_word}』，请检查输入单词是否直接存在于句子内。")

        st.markdown("### 🔎 方法对比小结")
        compare_rows = [
            {
                "方法": "Lesk",
                "Synset": lesk_synset_name or "未找到",
                "Definition": lesk_synset_definition or "未找到",
            },
            {
                "方法": "MFS",
                "Synset": mfs_synset_name or "未找到",
                "Definition": mfs_synset_definition or "未找到",
            },
            {
                "方法": "BERT(跨句向量)",
                "Synset": "-",
                "Definition": f"余弦相似度: {cos_sim:.5f}" if cos_sim is not None else "余弦相似度未计算",
            },
        ]
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

        st.session_state.wsd_report = {
            "sent1": sent1,
            "target_word": target_word,
            "sent2": sent2,
            "lesk_synset": lesk_synset_name,
            "lesk_definition": lesk_synset_definition,
            "mfs_synset": mfs_synset_name,
            "mfs_definition": mfs_synset_definition,
            "cos_sim": cos_sim,
        }

with tab2:
    st.markdown("### 🕸️ 语义角色标注 (SRL) 提取与解析")
    st.info("基于 **spaCy 依存树提取启发式结构引擎**，重构句子的「施发动作」深层逻辑链条。", icon="🔍")
    
    st.markdown("#### 📥 分析目标句子")
    srl_sentence = st.text_input("📝 (支持英文实体和动作组合)", value="Apple is manufacturing new smartphones in China this year.", label_visibility="collapsed")
    
    srl_btn = st.button("🚀 运行高级 SRL 解析", use_container_width=True)
    
    if srl_btn:
        doc = nlp(srl_sentence)
        srl_data_a = extract_srl_method_a(doc)
        srl_data_b = extract_srl_method_b(doc)
        role_cols = ["A0 (施事者)", "Predicate (谓词)", "A1 (受事者)", "AM-LOC (地点)", "AM-TMP (时间)"]
        compare_rows = [
            {"方法": "方法A-基础依存规则", **srl_data_a},
            {"方法": "方法B-增强启发式规则", **srl_data_b},
        ]
                
        # 结构化展示
        st.divider()
        st.markdown("### 📊 第一层：多方法角色抽取对比")
        st.caption("同一句子由两套轻量级规则独立提取，便于比较稳定性与覆盖率")
        compare_df = pd.DataFrame(compare_rows)
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        st.markdown("### 📈 第二层：角色覆盖可视化")
        coverage_df = pd.DataFrame([
            {
                "方法": row["方法"],
                "已识别角色数": sum(1 for col in role_cols if row[col] != "-"),
            }
            for row in compare_rows
        ]).set_index("方法")
        st.bar_chart(coverage_df)

        matrix_df = pd.DataFrame([
            {"方法": row["方法"], **{col: (1 if row[col] != "-" else 0) for col in role_cols}}
            for row in compare_rows
        ]).set_index("方法")
        st.caption("下表中 1 表示该角色被识别，0 表示未识别。")
        st.dataframe(matrix_df, use_container_width=True)
        
        # 依存图辅助
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎨 第三层：底层依存句法语法树")
        st.caption("由 spaCy NER + Dependency Parser 强力驱动可视化：展现中心词对子词的约束。")
        html_out = displacy.render(doc, style="dep", page=False, options={"distance": 120, "color": "#1E3A8A", "bg": "#F9FAFB", "font": "Segoe UI"})
        
        with st.container():
            st.components.v1.html(html_out, height=450, scrolling=True)
        
        # === 添加专用的导出功能 ===
        st.divider()
        st.markdown("### 💾 文档生成引擎")
        st.caption("一键式将分析结果编译并封装为纯静态 HTML Web 页面。")
        
        # 生成带样式的独立 HTML 文档
        compare_table_html = compare_df.to_html(index=False)
        coverage_table_html = coverage_df.reset_index().to_html(index=False)
        matrix_table_html = matrix_df.reset_index().to_html(index=False)
        coverage_rows = coverage_df.reset_index().to_dict(orient="records")
        tree_html = displacy.render(doc, style="dep", page=True, options={"distance": 100})
        st.session_state.srl_report = {
            "sentence": srl_sentence,
            "compare_table_html": compare_table_html,
            "coverage_table_html": coverage_table_html,
            "matrix_table_html": matrix_table_html,
            "coverage_rows": coverage_rows,
            "tree_html": tree_html,
        }

        full_html = build_full_report_html(
            st.session_state.get("wsd_report"),
            st.session_state.get("srl_report"),
        )
        
        st.download_button(
            label="📥 下载双模块综合 HTML 报告",
            data=full_html,
            file_name="NLP_WSD_SRL_Report.html",
            mime="text/html"
        )

