import streamlit as st
import jieba
import jieba.posseg as pseg
from snownlp import SnowNLP
import re
import zhconv
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def full_to_half(text):
    """全角转半角"""
    result = ""
    for uchar in text:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转换
            inside_code -= 65248
        result += chr(inside_code)
    return result

def normalize_text(text):
    """文本规范化"""
    # 1. 繁体转简体
    text = zhconv.convert(text, 'zh-cn')
    # 2. 全角转半角
    text = full_to_half(text)
    # 3. 去除特殊符号（保留基本标点、汉字、字母和数字）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；：“”‘’（）《》]', '', text)
    return text

st.set_page_config(page_title="NLP 中文文本处理与分词平台", layout="wide")

st.title("📚 NLP 中文文本处理与分词平台")
st.markdown("该平台提供文本规范化、中文分词及词频统计、词性标注、以及**分词算法对比**功能。")

# --- 主体功能区 ---
st.header("1. 长文本基础NLP处理")
user_input = st.text_area("请输入一段中文长文本：", value="中國傳統的文本需要規範化！今天是2026年， NLP技術發展非常迅速。這是一個測試句子，包含全角字符１２３，以及特殊符號@#￥%。", height=150)

if user_input:
    col1, col2, col3 = st.columns(3)
    
    # 区块 1：文本规范化
    with col1:
        st.subheader("区块1：文本规范化结果")
        normalized_text = normalize_text(user_input)
        st.success("规范化处理完成：")
        st.write(normalized_text)
        
    # 区块 2：中文分词与词频统计
    with col2:
        st.subheader("区块2：分词与词频统计")
        if normalized_text.strip():
            # 使用jieba进行分词
            words = list(jieba.cut(normalized_text))
            # 去除停用词或标点（简单处理：过滤短于2的或是纯标点）
            valid_words = [w for w in words if re.match(r'[\u4e00-\u9fa5a-zA-Z0-9]+', w)]
            
            st.write("**分词结果：**")
            st.info(" ".join(words))
            
            word_counts = Counter(valid_words)
            top_5 = word_counts.most_common(5)
            
            st.write("**最高频的5个词：**")
            if top_5:
                df_top5 = pd.DataFrame(top_5, columns=["词语", "频次"])
                st.dataframe(df_top5, hide_index=True)
                
                # 去掉 matplotlib，改用 streamlit 原生图表
                st.write("**Top 5 词频统计：**")
                # 设置词语为索引，以便 st.bar_chart 直接作为 X 轴
                st.bar_chart(df_top5.set_index('词语')['频次'])
            else:
                st.write("未能提取到有效词语！")
        else:
            st.warning("输入文本为空！")

    # 区块 3：词性标注
    with col3:
        st.subheader("区块3：词性标注")
        if normalized_text.strip():
            words_with_flags = pseg.cut(normalized_text)
            
            # 使用HTML进行高亮
            # 名词(n开头)->红色, 动词(v开头)->蓝色, 形容词(a开头)->绿色
            html_output = ""
            for word, flag in words_with_flags:
                if flag.startswith('n'):
                    html_output += f'<span style="color:red; font-weight:bold; margin-right:5px;" title="{flag}">{word}</span>'
                elif flag.startswith('v'):
                    html_output += f'<span style="color:blue; font-weight:bold; margin-right:5px;" title="{flag}">{word}</span>'
                elif flag.startswith('a'):
                    html_output += f'<span style="color:green; font-weight:bold; margin-right:5px;" title="{flag}">{word}</span>'
                else:
                    html_output += f'<span style="color:gray; margin-right:5px;" title="{flag}">{word}</span>'
                    
            st.markdown(html_output, unsafe_allow_html=True)
            
            st.markdown("""
            **图例：**
            - <span style="color:red; font-weight:bold;">名词 (Red)</span>
            - <span style="color:blue; font-weight:bold;">动词 (Blue)</span>
            - <span style="color:green; font-weight:bold;">形容词 (Green)</span>
            - <span style="color:gray;">其他 (Gray)</span>
            """, unsafe_allow_html=True)


st.divider()

# --- 分词算法对比区 ---
st.header("2. 歧义句分词算法对比分析")
st.markdown("输入容易产生歧义的句子，观察不同算法（Jieba的不同模式、SnowNLP）的切分结果。")

ambiguous_input = st.text_input("请输入测试句子：", value="南京市长江大桥")

if ambiguous_input:
    algorithms = ["Jieba (精确模式)", "Jieba (全模式)", "Jieba (搜索引擎模式)", "SnowNLP"]
    results = []
    
    # 1. Jieba 精确模式
    start_time = time.time()
    exact_res = list(jieba.cut(ambiguous_input, cut_all=False))
    exact_time = (time.time() - start_time) * 1000
    results.append({"算法": "Jieba (精确模式)", "分词结果": " / ".join(exact_res), "分词数量": len(exact_res), "耗时(ms)": exact_time})
    
    # 2. Jieba 全模式
    start_time = time.time()
    full_res = list(jieba.cut(ambiguous_input, cut_all=True))
    full_time = (time.time() - start_time) * 1000
    results.append({"算法": "Jieba (全模式)", "分词结果": " / ".join(full_res), "分词数量": len(full_res), "耗时(ms)": full_time})
    
    # 3. Jieba 搜索引擎模式
    start_time = time.time()
    search_res = list(jieba.cut_for_search(ambiguous_input))
    search_time = (time.time() - start_time) * 1000
    results.append({"算法": "Jieba (搜索引擎模式)", "分词结果": " / ".join(search_res), "分词数量": len(search_res), "耗时(ms)": search_time})
    
    # 4. SnowNLP
    try:
        start_time = time.time()
        s = SnowNLP(ambiguous_input)
        snow_res = s.words
        snow_time = (time.time() - start_time) * 1000
    except Exception as e:
        snow_res = ["Error"]
        snow_time = 0.0
    results.append({"算法": "SnowNLP", "分词结果": " / ".join(snow_res), "分词数量": len(snow_res), "耗时(ms)": snow_time})
    
    # 展示DataFrame
    df_results = pd.DataFrame(results)
    st.table(df_results)
    
    # 可视化分析
    st.subheader("算法表现可视化分析")
    
    # 改用 Streamlit 原生图表解决云端中文乱码问题
    st.write("**各算法分词数量对比**")
    st.bar_chart(df_results.set_index("算法")["分词数量"])
    
    st.write("**各算法分词耗时对比 (ms)**")
    st.bar_chart(df_results.set_index("算法")["耗时(ms)"])
        
    st.info("**分析结论示例：**\n\nJieba精确模式和SnowNLP旨在找出最可能的一条切分路径，往往遇到歧义时切分可能不同；而Jieba的『全模式』扫描出了所有字典中存在的词组，因此分词数量会显著多于其他算法。")
