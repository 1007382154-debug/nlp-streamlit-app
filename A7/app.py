import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json

st.set_page_config(page_title="信息抽取与知识图谱系统", layout="wide")

import spacy

# 缓存加载 spaCy 模型，避免每次交互重复加载
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # 如果模型未安装，则自动下载（适配部分环境）
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

# ==========================================
# 基于 spaCy 的通用数据抽取模型 (进阶版本)
# ==========================================
def extract_pipeline(text):
    doc = nlp(text)
    
    # 1. 抽取实体 (NER)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
        
    # 2. 抽取序列标注 (BIO Tagging)
    bio_tags = []
    for token in doc:
        # spaCy 的 ent_iob_ 返回 I, O, B 格式
        if token.ent_iob_ == "O":
            tag = "O"
        else:
            tag = f"{token.ent_iob_}-{token.ent_type_}"
        bio_tags.append({
            "token": token.text,
            "tag": tag
        })
        
    # 3. 简单关系抽取 (基于依存句法分析 Dependency Parsing)
    # 这是一种简化的基于规则(主谓宾)抽取方式来实现任何文本的关系抓取
    relations = []
    for token in doc:
        # 寻找动词作为关系词
        if token.pos_ == "VERB":
            subject = None
            object_ = None
            for child in token.children:
                # 寻找主语
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child.text
                # 寻找宾语
                if child.dep_ in ("dobj", "pobj", "attr"):
                    object_ = child.text
            
            # 找到成对主宾关系时，组成三元组
            if subject and object_:
                relations.append({
                    "source": subject,
                    "relation": token.lemma_.upper(),
                    "target": object_
                })
                
    # 如果基于语法没抽到关系，但存在多个实体，模拟共现关系（确保图谱有数据看）
    if not relations and len(entities) > 1:
        for i in range(len(entities) - 1):
            relations.append({
                "source": entities[i]["text"],
                "relation": "CO_OCCURRENCE",
                "target": entities[i+1]["text"]
            })

    return {
        "entities": entities,
        "relations": relations,
        "bio_tags": bio_tags
    }

# ==========================================
# UI 布局和渲染
# ==========================================
st.title("🧩 信息抽取与知识图谱系统 (A7 实验)")
st.markdown("通过 Vibe Coding 模式开发的集成 NER、关系抽取与知识图谱可视化的交互式 Web 系统。")

# 模块1: 文本输入
st.header("1. 文本输入与实体识别 (NER)")
text_input = st.text_area(
    "请输入一段包含人名、机构名的语料进行分析 (例如: Steve Jobs founded Apple / University of California, Los Angeles)", 
    "Steve Jobs founded Apple."
)

if st.button("开始抽取分析", type="primary"):
    result = extract_pipeline(text_input)
    
    st.markdown("### 识别结果")
    
    # Checkbox切换模式
    show_bio = st.checkbox("查看底层标注 (BIO模式)")
    
    if show_bio:
        st.info("💡 **观察任务**：体会课件中 B（Begin）、I（Inside）、O（Outside）是如何帮助模型确定实体边界的；思考嵌套实体识别时的挑战。")
        bio_df = pd.DataFrame(result["bio_tags"])
        st.table(bio_df)
    else:
        # 简单使用 Markdown 和颜色来模拟高亮
        highlighted_html = text_input
        color_map = {"PER": "#ff9d66", "ORG": "#5ec878", "LOC": "#5f9df6"}
        
        for ent in result["entities"]:
            color = color_map.get(ent["label"], "#8ea2b9")
            tag_html = f"<span style='background-color:{color}; color:white; padding:2px 6px; border-radius:4px; margin: 0 4px;'>{ent['text']} ({ent['label']})</span>"
            highlighted_html = highlighted_html.replace(ent['text'], tag_html)
            
        st.markdown(f"<div style='line-height:2.5; font-size:18px; padding:10px; border:1px solid #ddd; border-radius:8px;'>{highlighted_html}</div>", unsafe_allow_html=True)

    # 模块2: 实体关系抽取
    st.header("2. 实体关系抽取 (Relation Extraction)")
    st.info("💡 **观察任务**：理解关系抽取本质上是在图结构中为两个实体节点之间预测是否存在特定语义类型的边。")
    rel_df = pd.DataFrame(result["relations"])
    
    if not rel_df.empty:
        rel_df.columns = ["主体 (Subject)", "关系词 (Predicate)", "客体 (Object)"]
        st.table(rel_df)
    else:
        st.warning("未抽取到关系数据")

    # 模块3: 知识图谱交互可视化
    st.header("3. 知识图谱交互可视化 (Knowledge Graph)")
    st.info("💡 **观察任务**：观察生成的知识图谱是如何将“线性文本”瞬间转化为“网状结构数据”的。")
    
    # 准备 Echarts 图数据
    nodes_map = {}
    for ent in result["entities"]:
        nodes_map[ent["text"]] = {
            "id": ent["text"], 
            "name": ent["text"], 
            "category": ent["label"],
            "symbolSize": 50
        }
    
    links = []
    for rel in result["relations"]:
        links.append({"source": rel["source"], "target": rel["target"], "relation": rel["relation"]})
        if rel["source"] not in nodes_map:
            nodes_map[rel["source"]] = {"id": rel["source"], "name": rel["source"], "category": "UNKNOWN", "symbolSize": 40}
        if rel["target"] not in nodes_map:
            nodes_map[rel["target"]] = {"id": rel["target"], "name": rel["target"], "category": "UNKNOWN", "symbolSize": 40}

    graph_data = {"nodes": list(nodes_map.values()), "links": links}
    graph_json = json.dumps(graph_data, ensure_ascii=False)
    
    # 内嵌 HTML Echarts
    echarts_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    </head>
    <body style="margin:0; padding:0;">
        <div id="main" style="width: 100%; height: 500px; border: 1px solid #eee; border-radius: 8px;"></div>
        <script>
            var chartDom = document.getElementById('main');
            var myChart = echarts.init(chartDom);
            var graphData = {graph_json};
            var categoryColors = {{ PER: '#ff9d66', ORG: '#5ec878', LOC: '#5f9df6', UNKNOWN: '#8ea2b9' }};
            
            var option = {{
                tooltip: {{
                    formatter: function(params) {{
                        if (params.dataType === 'edge') return params.data.source + ' -> ' + params.data.relation + ' -> ' + params.data.target;
                        return params.data.name + ' (' + params.data.category + ')';
                    }}
                }},
                series: [{{
                    type: 'graph',
                    layout: 'force',
                    roam: true,
                    label: {{ show: true, fontSize: 14 }},
                    edgeSymbol: ['none', 'arrow'],
                    edgeSymbolSize: [4, 10],
                    force: {{ repulsion: 1000, edgeLength: 150 }},
                    data: graphData.nodes.map(function(n) {{
                        return Object.assign({{}}, n, {{ itemStyle: {{ color: categoryColors[n.category] || categoryColors.UNKNOWN }} }});
                    }}),
                    links: graphData.links.map(function(e) {{
                        return Object.assign({{}}, e, {{ label: {{ show: true, formatter: e.relation }}, lineStyle: {{ curveness: 0.2, width: 2 }} }});
                    }})
                }}]
            }};
            myChart.setOption(option);
            window.addEventListener('resize', function() {{ myChart.resize(); }});
        </script>
    </body>
    </html>
    """
    
    components.html(echarts_html, height=520)