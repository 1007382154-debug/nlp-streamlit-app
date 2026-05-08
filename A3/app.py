import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec, FastText
import gensim.downloader as api
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

st.set_page_config(page_title="NLP Text Representation Apps", layout="wide")

# Ensure NLTK resources are available
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
download_nltk_data()

# Sample Text
SAMPLE_TEXT = """
Natural language processing (NLP) is an interdisciplinary subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

Challenges in natural language processing frequently involve speech recognition, natural-language understanding, and natural-language generation. Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time not articulated as a problem separate from artificial intelligence.

The premise of symbolic NLP is well-summarized by John Searle's Chinese room experiment: Given a collection of rules (e.g., a Chinese phrasebook, with questions and matching answers), the computer emulates natural language understanding (or other NLP tasks) by applying those rules to the data it confronts.

Machine learning approaches, which include both statistical machine learning and deep learning, are a paradigm shift from symbolic NLP. Statistical machine learning uses statistical inference to automatically learn such rules through the analysis of large corpora of typical real-world examples. Deep learning attempts to model how the human brain works by using neural networks to process data.

Text mining, also known as text data mining, is the process of deriving high-quality information from text. High-quality information is typically derived through the devising of patterns and trends through means such as statistical pattern learning.

Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with many dimensions per word to a continuous vector space with a much lower dimension.

Methods to generate this mapping include neural networks, dimensionality reduction on the word co-occurrence matrix, probabilistic models, explainable knowledge base method, and explicit representation in terms of the context in which words appear.
"""

st.title("NLP 文本表示与词向量交互系统")

corpus = st.text_area("输入英文语料 (输入样本约 300词)：", value=SAMPLE_TEXT.strip(), height=200)

if corpus:
    sentences = sent_tokenize(corpus)
    # 预处理：分词小写，只保留字母
    tokenized_sentences = [[word.lower() for word in word_tokenize(sent) if word.isalpha()] for sent in sentences]
else:
    st.warning("内容为空！")
    st.stop()


tab1, tab2, tab3, tab4 = st.tabs(["1: 传统统计模型 (TF-IDF & LSA)", "2: Word2Vec (CBOW vs Skip-Gram)", "3: 预训练模型 (GloVe)", "4: 子词特征与句向量 (FastText & Sent2Vec)"])

with tab1:
    st.header("传统统计模型: TF-IDF 与 LSA")
    st.write("将文本按句子切分为文档集合。")
    
    clean_sentences = [" ".join(sent) for sent in tokenized_sentences if sent]
    if not clean_sentences:
        st.warning("语料中未找到有效句子。")
    else:
        # TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(clean_sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        # 获取最重要5个关键词 (通过平均 TF-IDF)
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        top5_idx = mean_tfidf.argsort()[-5:][::-1]
        top5_words = [feature_names[i] for i in top5_idx]
        
        st.subheader("TF-IDF 最高权重的 5 个关键词")
        st.write(", ".join(top5_words))
        
        st.subheader("LSA 降维可视化")
        if len(feature_names) >= 2:
            svd = TruncatedSVD(n_components=2, random_state=42)
            # 通过转置对词汇进行降维，或使用原始共现矩阵
            # count_vectorizer = CountVectorizer()
            # X = count_vectorizer.fit_transform(clean_sentences)
            # X_T = X.T # shape: (words, documents)
            # word_coords = svd.fit_transform(X_T)
            # 此处直接对 tfidf 的特征（即词汇在各文档的特征）转置来降维词汇
            word_coords = svd.fit_transform(tfidf_matrix.T)
            
            # 使用 Streamlit 原生 scatter_chart 解决图表中文乱码问题
            df_coords = pd.DataFrame(
                word_coords,
                columns=['x', 'y'],
                index=feature_names
            )
            # 因为数据点不多，直接把 top words 的标记加到 DataFrame 中以便提示
            # 也可以直接显示普通的散点图
            st.scatter_chart(df_coords)
            
            st.info("观察：共现频率较高的词语是否被映射到了相近的二维空间。")

with tab2:
    st.header("Word2Vec 实时训练与对比")
    
    col1, col2 = st.columns(2)
    with col1:
        arch = st.radio("选择模型架构 (Architecture)", ["CBOW (sg=0)", "Skip-Gram (sg=1)"])
        sg_param = 0 if arch.startswith("CBOW") else 1
    with col2:
        window_size = st.slider("上下文窗口大小 (Window)", min_value=2, max_value=10, value=5)
    
    # 动态训练 Word2Vec
    if tokenized_sentences:
        w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=window_size, min_count=1, sg=sg_param, workers=4)
        
        target_word = st.text_input("输入查询目标词汇", value="language")
        target_word = target_word.lower().strip()
        if target_word:
            if target_word in w2v_model.wv.key_to_index:
                similar_words = w2v_model.wv.most_similar(target_word, topn=5)
                st.write(f"与 **{target_word}** 余弦相似度最高的前 5 个词汇：")
                for w, sim in similar_words:
                    st.write(f"- {w}: {sim:.4f}")
            else:
                st.error(f"词汇 '{target_word}' 不在词表中 (OOV)。请尝试语料中有的词汇。")
                st.write("可用词汇示例：", ", ".join(list(w2v_model.wv.key_to_index.keys())[:10]))
        st.info("观察任务：切换 CBOW 和 Skip-Gram 架构，观察同一个目标词的 Top 5 相似词是否发生变化。")

with tab3:
    st.header("预训练模型与词类比 (GloVe)")
    
    @st.cache_resource(show_spinner="Downloading glove-twitter-25 model... This may take a moment.")
    def load_glove():
        return api.load("glove-twitter-25")
    
    try:
        glove_model = load_glove()
        st.success("GloVe 模型加载成功! (glove-twitter-25)")
        
        st.subheader("词类比: A - B + C = ?")
        st.write("例如课件例子: king - man + woman = queen")
        c1, c2, c3 = st.columns(3)
        word_a = c1.text_input("A (正向)", value="king")
        word_b = c2.text_input("B (负向)", value="man")
        word_c = c3.text_input("C (正向)", value="woman")
        
        if st.button("计算词类比"):
            try:
                # model.most_similar(positive=[word_a, word_c], negative=[word_b])
                result = glove_model.most_similar(positive=[word_a.lower(), word_c.lower()], negative=[word_b.lower()], topn=3)
                st.write("最接近的词：")
                for w, sim in result:
                    st.write(f"- **{w}** (相似度: {sim:.4f})")
            except KeyError as e:
                st.error(f"输入的词包含模型未登录词: {e}")
        
        st.write("尝试其他示例：巴黎(paris) - 法国(france) + 中国(china) = 北京(beijing)")
        
        st.subheader("两个单词相似度计算")
        ca, cb = st.columns(2)
        sim_word1 = ca.text_input("单词1", value="car")
        sim_word2 = cb.text_input("单词2", value="automobile")
        if st.button("计算分数"):
            try:
                score = glove_model.similarity(sim_word1.lower(), sim_word2.lower())
                st.write(f"相似度: **{score:.4f}**")
            except KeyError as e:
                st.error(f"未登录词: {e}")
                
    except Exception as e:
        st.error(f"加载模型失败: {e}")


with tab4:
    st.header("子词特征与句向量 (FastText & Sent2Vec)")
    
    if tokenized_sentences:
        ft_model = FastText(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, workers=4)
        
        st.subheader("FastText 处理 OOV (未登录词) 问题")
        oov_word = st.text_input("输入一个故意拼错的词（例如 computeer）", value="computteer")
        
        if st.button("测试 OOV (Word2Vec vs FastText)"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Word2Vec 提取情况:**")
                # 重新构建一个 w2v，确保存在
                w2v_model_tmp = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1)
                try:
                    _ = w2v_model_tmp.wv[oov_word]
                    st.success("竟然存在！")
                except KeyError:
                    st.error(f"未登录词: KeyError: '{oov_word}' not in vocabulary")
            
            with c2:
                st.write("**FastText 提取情况:**")
                try:
                    ft_vec = ft_model.wv[oov_word]
                    st.success(f"成功获取向量! 维度: {ft_vec.shape}")
                    sims = ft_model.wv.most_similar(oov_word, topn=3)
                    st.write("FastText 认为最相似的词：")
                    for w, s in sims:
                        st.write(f"- {w}: {s:.4f}")
                except Exception as e:
                    st.error(f"提取出错: {e}")
                    
        st.subheader("Sent2Vec - 句向量均值计算 (Average Pooling)")
        sent1 = st.text_area("句子 1", "Natural language processing is amazing.")
        sent2 = st.text_area("句子 2", "The computer algorithm processes natural language successfully.")
        
        if st.button("计算句子相似度"):
            words1 = [w.lower() for w in word_tokenize(sent1) if w.isalpha()]
            words2 = [w.lower() for w in word_tokenize(sent2) if w.isalpha()]
            
            if words1 and words2:
                vecs1 = [ft_model.wv[w] for w in words1]
                vecs2 = [ft_model.wv[w] for w in words2]
                
                # Average pooling
                sent_vec1 = np.mean(vecs1, axis=0)
                sent_vec2 = np.mean(vecs2, axis=0)
                
                # Cosine similarity
                cos_sim = np.dot(sent_vec1, sent_vec2) / (np.linalg.norm(sent_vec1) * np.linalg.norm(sent_vec2))
                st.write(f"句向量余弦相似度: **{cos_sim:.4f}**")
            else:
                st.warning("句子由于只包含标点或为空，无法提取有效词汇。")
