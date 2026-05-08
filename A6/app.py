import html
import math
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

import nltk
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from nltk.corpus import reuters
from nltk.tokenize import TreebankWordTokenizer
from nltk.util import ngrams
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


st.set_page_config(
    page_title="综合语言模型交互系统",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Source+Serif+4:wght@400;600&display=swap');
    :root {
        --ink: #14213d;
        --subink: #2a4d69;
        --line: #c9d8e5;
        --card: rgba(255, 255, 255, 0.88);
        --accent-1: #ffd77a;
        --accent-2: #b6e2d3;
        --accent-3: #c7d2fe;
        --accent-4: #ffc9de;
    }
    html, body, [class*="css"] {
        font-family: 'Source Serif 4', 'PingFang SC', 'Microsoft YaHei', serif;
    }
    .stApp {
        color: var(--ink);
        background:
            radial-gradient(100% 50% at 5% 0%, #fff7d6 0%, transparent 60%),
            radial-gradient(90% 60% at 95% 5%, #d9efe7 0%, transparent 65%),
            linear-gradient(180deg, #f8fbff 0%, #eef4f9 100%);
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    .hero {
        border: 1px solid rgba(20, 33, 61, 0.16);
        border-radius: 18px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        background: linear-gradient(120deg, rgba(255,255,255,0.9) 0%, rgba(228,240,255,0.78) 100%);
        box-shadow: 0 10px 30px rgba(20, 33, 61, 0.08);
        animation: fadeUp 0.6s ease-out;
    }
    .hero h1 {
        margin: 0;
        font-family: 'Space Grotesk', 'Source Serif 4', serif;
        letter-spacing: 0.4px;
        color: #0f274f;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        color: var(--subink);
    }
    [data-testid="stTabs"] {
        margin-top: 0.8rem;
        animation: fadeUp 0.75s ease-out;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-radius: 12px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.72);
        gap: 8px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        color: #314e66;
        font-family: 'Space Grotesk', 'Source Serif 4', serif;
        font-weight: 700;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #fff0bc 0%, #d9efdf 100%);
        color: #14213d;
        border: 1px solid #c7d9cb;
    }
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #b8cad8;
        font-weight: 700;
        box-shadow: 0 6px 15px rgba(20, 33, 61, 0.12);
    }
    .stTextArea textarea,
    .stTextInput input {
        border-radius: 10px;
        background: rgba(255,255,255,0.85);
    }
    .metric-card {
        border: 1px solid #ccdbe7;
        border-radius: 12px;
        padding: 10px 12px;
        background: var(--card);
        margin-bottom: 8px;
    }
    .module-note {
        color: #365772;
        font-size: 0.94rem;
        margin-bottom: 0.5rem;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 768px) {
        .hero h1 { font-size: 1.5rem; }
        .block-container { padding-top: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_report_state() -> None:
    if "report_state" not in st.session_state:
        st.session_state["report_state"] = {
            "module1": {},
            "module2": {},
            "module3": {},
            "module4": {},
        }


@st.cache_resource(show_spinner=False)
def ensure_reuters_corpus() -> bool:
    try:
        nltk.data.find("corpora/reuters")
        return True
    except LookupError:
        try:
            nltk.download("reuters", quiet=True)
            nltk.data.find("corpora/reuters")
            return True
        except Exception:  # noqa: BLE001
            return False


@st.cache_data(show_spinner=False)
def load_reuters_text(category: str, max_docs: int = 10, max_chars: int = 8000) -> str:
    file_ids = reuters.fileids(category)
    docs = [reuters.raw(fid) for fid in file_ids[:max_docs]]
    text = " ".join(docs)
    return text[:max_chars]


TOKENIZER = TreebankWordTokenizer()


def tokenize_words(text: str) -> List[str]:
    return [tok.lower() for tok in TOKENIZER.tokenize(text) if tok.strip()]


def build_ngram_stats(corpus_text: str, n: int) -> Dict:
    tokens = tokenize_words(corpus_text)
    padded = ["<s>"] * (n - 1) + tokens + ["</s>"]

    ngram_counter = Counter(ngrams(padded, n))
    context_counter = Counter(ngrams(padded, n - 1))
    word_freq = Counter(tokens)

    return {
        "tokens": tokens,
        "ngram_counter": ngram_counter,
        "context_counter": context_counter,
        "word_freq": word_freq,
        "vocab": sorted(set(tokens)),
    }


def sentence_joint_probability(sentence: str, stats: Dict, n: int, smoothing: bool) -> Dict:
    sent_tokens = tokenize_words(sentence)
    padded = ["<s>"] * (n - 1) + sent_tokens + ["</s>"]

    rows = []
    log_prob = 0.0
    zero_event = False
    vocab_size = max(1, len(stats["vocab"]) + 1)

    for gram in ngrams(padded, n):
        context = gram[:-1]
        gram_count = stats["ngram_counter"].get(gram, 0)
        context_count = stats["context_counter"].get(context, 0)

        if smoothing:
            prob = (gram_count + 1.0) / (context_count + vocab_size)
        else:
            prob = (gram_count / context_count) if context_count > 0 else 0.0

        if prob == 0.0:
            zero_event = True
            log_prob = float("-inf")
        elif log_prob != float("-inf"):
            log_prob += math.log(prob)

        rows.append(
            {
                "n-gram": " ".join(gram),
                "count": gram_count,
                "context_count": context_count,
                "probability": prob,
                "seen": "Yes" if gram_count > 0 else "No",
            }
        )

    if zero_event:
        joint = 0.0
    else:
        joint = math.exp(log_prob) if log_prob > -745 else 0.0

    return {
        "rows": rows,
        "joint": joint,
        "log_prob": log_prob,
        "zero_event": zero_event,
    }


class CharRNNLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, rnn_type: str = "RNN"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        logits = self.fc(out)
        return logits, hidden


def build_char_dataset(text: str, seq_len: int = 25, max_samples: int = 2500) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    chars = sorted(set(text))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}

    encoded = [char2idx[ch] for ch in text]
    stride = max(1, seq_len // 2)

    xs = []
    ys = []
    for i in range(0, len(encoded) - seq_len - 1, stride):
        xs.append(encoded[i : i + seq_len])
        ys.append(encoded[i + 1 : i + seq_len + 1])
        if len(xs) >= max_samples:
            break

    x_tensor = torch.tensor(xs, dtype=torch.long)
    y_tensor = torch.tensor(ys, dtype=torch.long)
    return x_tensor, y_tensor, char2idx, idx2char


def train_char_rnn(
    text: str,
    hidden_size: int,
    epochs: int,
    lr: float,
    rnn_type: str,
) -> Dict:
    if len(set(text)) < 3:
        raise ValueError("训练语料字符种类太少，请输入更丰富的文本。")

    seq_len = min(35, max(12, len(text) // 8))
    x_tensor, y_tensor, char2idx, idx2char = build_char_dataset(text, seq_len=seq_len)

    if len(x_tensor) < 2:
        raise ValueError("语料长度不足，至少需要约 60 个字符。")

    vocab_size = len(char2idx)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CharRNNLM(vocab_size, hidden_size, rnn_type=rnn_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    progress = st.progress(0)
    status = st.empty()
    chart = st.empty()

    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for bx, by in loader:
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits.reshape(-1, vocab_size), by.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        losses.append(avg_loss)

        chart.line_chart(pd.DataFrame({"Loss": losses}))
        progress.progress((epoch + 1) / epochs)
        status.info(f"训练中: Epoch {epoch + 1}/{epochs}, Loss={avg_loss:.4f}")

    status.success("训练完成")

    return {
        "model": model,
        "char2idx": char2idx,
        "idx2char": idx2char,
        "losses": losses,
        "rnn_type": rnn_type,
    }


def generate_text(bundle: Dict, seed: str, gen_len: int = 50, temperature: float = 0.9) -> str:
    model: CharRNNLM = bundle["model"]
    char2idx = bundle["char2idx"]
    idx2char = bundle["idx2char"]

    if not seed:
        seed = next(iter(char2idx.keys()))

    model.eval()
    hidden = None
    output_chars = list(seed)

    with torch.no_grad():
        default_idx = next(iter(idx2char.keys()))

        for ch in seed[:-1]:
            idx = char2idx.get(ch, default_idx)
            x = torch.tensor([[idx]], dtype=torch.long)
            _, hidden = model(x, hidden)

        current_idx = char2idx.get(seed[-1], default_idx)

        for _ in range(gen_len):
            x = torch.tensor([[current_idx]], dtype=torch.long)
            logits, hidden = model(x, hidden)
            logits = logits[:, -1, :] / max(0.1, temperature)
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            output_chars.append(idx2char[next_idx])
            current_idx = next_idx

    return "".join(output_chars)


@st.cache_resource(show_spinner=False)
def load_bert_fill_mask():
    return pipeline("fill-mask", model="bert-base-uncased")


@st.cache_resource(show_spinner=False)
def load_gpt2_assets():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return tokenizer, model, text_generator


def gpt2_continue(prompt: str, text_generator, max_words: int = 20) -> Tuple[str, str]:
    outputs = text_generator(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=1,
    )

    full_text = outputs[0]["generated_text"]
    continuation = full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text.strip()
    cont_words = continuation.split()
    trimmed = " ".join(cont_words[:max_words])
    return trimmed, full_text


def calc_ppl_for_sentences(sentences: List[str], tokenizer, model) -> pd.DataFrame:
    rows = []
    for text in sentences:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask", None),
                labels=encoded["input_ids"],
            )
            loss = float(outputs.loss.item())

        ppl = math.exp(loss) if loss < 50 else float("inf")
        rows.append({"Sentence": text, "CrossEntropyLoss": loss, "PPL": ppl})

    return pd.DataFrame(rows)


def html_table_from_rows(rows: List[Dict]) -> str:
    if not rows:
        return '<p class="empty">暂无结果</p>'
    frame = pd.DataFrame(rows)
    return frame.to_html(index=False, escape=True, classes="tbl")


def build_static_html(report_state: Dict) -> str:
    m1 = report_state.get("module1", {})
    m2 = report_state.get("module2", {})
    m3 = report_state.get("module3", {})
    m4 = report_state.get("module4", {})

    m1_top_freq = html_table_from_rows(m1.get("top_freq_rows", []))
    m1_prob_table = html_table_from_rows(m1.get("prob_rows", []))
    m3_bert_table = html_table_from_rows(m3.get("bert_rows", []))
    m4_ppl_table = html_table_from_rows(m4.get("ppl_rows", []))

    return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>综合语言模型交互系统 - 静态报告</title>
  <style>
    :root {{
      --ink: #14213d;
      --sub: #2a4d69;
      --line: #c9d8e5;
      --card: rgba(255,255,255,0.94);
      --bg1: #fff8de;
      --bg2: #d9efe7;
      --bg3: #eef4f9;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: 'Source Serif 4', 'PingFang SC', 'Microsoft YaHei', serif;
      background:
        radial-gradient(90% 40% at 5% 0%, var(--bg1) 0%, transparent 60%),
        radial-gradient(90% 50% at 95% 5%, var(--bg2) 0%, transparent 65%),
        linear-gradient(180deg, #f8fbff 0%, var(--bg3) 100%);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 24px auto;
      padding: 0 14px 28px;
    }}
    .hero {{
      border: 1px solid rgba(20,33,61,0.16);
      border-radius: 16px;
      padding: 16px;
      background: linear-gradient(120deg, #ffffff 0%, #e4f0ff 100%);
      box-shadow: 0 10px 26px rgba(20,33,61,0.08);
    }}
    .hero h1 {{ margin: 0; font-size: 28px; }}
    .hero p {{ margin: 8px 0 0 0; color: var(--sub); }}
    .grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: 1fr;
      margin-top: 14px;
    }}
    .mod {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--card);
      padding: 14px;
      box-shadow: 0 5px 15px rgba(20,33,61,0.06);
    }}
    .mod h2 {{ margin-top: 0; font-size: 20px; }}
    .meta {{ color: var(--sub); margin: 4px 0; }}
    .tbl {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 14px;
    }}
    .tbl th, .tbl td {{
      border: 1px solid #d8e1ea;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    .tbl th {{
      background: #edf4fb;
    }}
    .quote {{
      border-left: 4px solid #cbd9e8;
      padding: 8px 10px;
      background: #f7fbff;
      white-space: pre-wrap;
    }}
    .empty {{ color: #5a7286; font-style: italic; }}
    @media (min-width: 960px) {{
      .grid {{ grid-template-columns: 1fr 1fr; }}
      .span2 {{ grid-column: span 2; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>综合语言模型交互系统 - 静态报告</h1>
      <p>导出时间: {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
      <p>包含模块: n 元统计模型、RNN 训练、BERT vs GPT-2 对比、GPT-2 困惑度评估</p>
    </section>

    <div class="grid">
      <section class="mod">
        <h2>模块 1: n 元语言模型与平滑</h2>
        <p class="meta">n 值: {html.escape(str(m1.get('n', 'N/A')))} | 语料 Token 数: {html.escape(str(m1.get('token_count', 'N/A')))} | 词表大小: {html.escape(str(m1.get('vocab_size', 'N/A')))}</p>
        <p class="meta">测试句子: {html.escape(m1.get('sentence', ''))}</p>
        <p class="meta">未平滑联合概率: {html.escape(str(m1.get('unsmoothed_joint', 'N/A')))}</p>
        <p class="meta">加一平滑联合概率: {html.escape(str(m1.get('smoothed_joint', 'N/A')))}</p>
        <h3>Top 词频</h3>
        {m1_top_freq}
        <h3>逐 n-gram 概率明细</h3>
        {m1_prob_table}
      </section>

      <section class="mod">
        <h2>模块 2: 从零训练 RNN 语言模型</h2>
        <p class="meta">模型: {html.escape(str(m2.get('rnn_type', 'N/A')))} | Hidden Size: {html.escape(str(m2.get('hidden_size', 'N/A')))} | Epochs: {html.escape(str(m2.get('epochs', 'N/A')))} | LR: {html.escape(str(m2.get('lr', 'N/A')))}</p>
        <p class="meta">最终 Loss: {html.escape(str(m2.get('final_loss', 'N/A')))}</p>
        <h3>训练语料预览</h3>
        <div class="quote">{html.escape(m2.get('train_text_preview', '暂无'))}</div>
        <h3>生成结果</h3>
        <p class="meta">Seed: {html.escape(m2.get('seed', ''))}</p>
        <div class="quote">{html.escape(m2.get('generated_text', '暂无生成结果'))}</div>
      </section>

      <section class="mod">
        <h2>模块 3: BERT 与 GPT-2 机制对比</h2>
        <h3>BERT Masked LM</h3>
        <p class="meta">输入: {html.escape(m3.get('mask_input', ''))}</p>
        {m3_bert_table}
        <h3>GPT-2 Causal LM</h3>
        <p class="meta">Prompt: {html.escape(m3.get('gpt_prompt', ''))}</p>
        <div class="quote">{html.escape(m3.get('gpt_continuation', '暂无生成结果'))}</div>
      </section>

      <section class="mod">
        <h2>模块 4: GPT-2 困惑度 (PPL) 评价</h2>
        <p class="meta">说明: PPL 越小，模型对句子的建模越好。</p>
        {m4_ppl_table}
      </section>

      <section class="mod span2">
        <h2>结论观察提示</h2>
        <p class="meta">1) 在模块 1 输入未见过的 n-gram，未平滑概率会归零；开启加一平滑后概率恢复为非零。</p>
        <p class="meta">2) 在模块 2 输入重复规律文本，调高 Epoch 通常会加速模式学习，生成文本更接近原序列。</p>
        <p class="meta">3) 模块 3 中 BERT 利用左右文填空，GPT-2 仅从左到右续写。</p>
        <p class="meta">4) 模块 4 中语法通顺句通常 PPL 更低，乱码句 PPL 更高。</p>
      </section>
    </div>
  </div>
</body>
</html>
"""


def render_module_1() -> Dict:
    st.subheader("模块 1: n 元语言模型与数据平滑")
    st.markdown("<div class='module-note'>基于 nltk 构建 Bigram/Trigram 统计语言模型，并观察零概率与加一平滑效果。</div>", unsafe_allow_html=True)

    source = st.radio("语料来源", ["手动输入", "NLTK Reuters 示例"], horizontal=True)

    default_manual = (
        "Language models estimate the probability of a word sequence. "
        "Data sparsity causes unseen n-grams to have zero probability in pure count-based models."
    )

    corpus_text = default_manual
    if source == "NLTK Reuters 示例":
        if ensure_reuters_corpus():
            categories = sorted(reuters.categories())
            category = st.selectbox("选择 Reuters 类别", categories, index=0)
            corpus_text = load_reuters_text(category)
        else:
            st.warning("未能下载 Reuters 语料，已切换为手动语料。")

    corpus_text = st.text_area("语料文本", value=corpus_text, height=180)
    n = st.selectbox("n-gram 阶数", [2, 3], index=1)

    if not corpus_text.strip():
        st.error("请先输入语料文本。")
        return {}

    stats = build_ngram_stats(corpus_text, n)
    token_count = len(stats["tokens"])
    vocab_size = len(stats["vocab"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='metric-card'><b>Token 数:</b> {token_count}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><b>词表大小:</b> {vocab_size}</div>", unsafe_allow_html=True)

    top_freq = stats["word_freq"].most_common(15)
    top_freq_rows = [{"word": w, "freq": f} for w, f in top_freq]
    st.dataframe(pd.DataFrame(top_freq_rows), use_container_width=True)

    sentence = st.text_input("输入待评估句子", value="The model predicts unseen events gracefully")
    use_smoothing = st.checkbox("对主结果启用加一平滑 (Laplace)", value=False)

    prob_rows = []
    unsmoothed_joint = None
    smoothed_joint = None

    if sentence.strip():
        unsmoothed = sentence_joint_probability(sentence, stats, n, smoothing=False)
        smoothed = sentence_joint_probability(sentence, stats, n, smoothing=True)

        unsmoothed_joint = unsmoothed["joint"]
        smoothed_joint = smoothed["joint"]

        selected = smoothed if use_smoothing else unsmoothed
        prob_rows = selected["rows"]

        st.markdown("### 概率结果")
        st.write(f"未平滑联合概率: `{unsmoothed['joint']:.6e}`")
        st.write(f"加一平滑联合概率: `{smoothed['joint']:.6e}`")
        st.write(f"当前主结果（{'平滑' if use_smoothing else '未平滑'}）: `{selected['joint']:.6e}`")

        if unsmoothed["zero_event"]:
            st.warning("检测到未见过的 n-gram。未平滑时联合概率会归零，体现数据稀疏问题。")

        st.dataframe(pd.DataFrame(prob_rows), use_container_width=True)

    snapshot = {
        "n": n,
        "token_count": token_count,
        "vocab_size": vocab_size,
        "sentence": sentence,
        "unsmoothed_joint": f"{unsmoothed_joint:.6e}" if unsmoothed_joint is not None else "N/A",
        "smoothed_joint": f"{smoothed_joint:.6e}" if smoothed_joint is not None else "N/A",
        "top_freq_rows": top_freq_rows,
        "prob_rows": prob_rows,
    }

    st.session_state["report_state"]["module1"] = snapshot
    return snapshot


def render_module_2() -> Dict:
    st.subheader("模块 2: 从零训练 RNN 语言模型")
    st.markdown("<div class='module-note'>使用 PyTorch 训练字符级 RNN/LSTM，实现 next-character 自回归预测并可视化 Loss 曲线。</div>", unsafe_allow_html=True)

    train_text = st.text_area(
        "输入训练语料",
        value="hello world hello world hello world hello world ",
        height=160,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hidden_size = st.slider("Hidden Size", 16, 128, 64, step=8)
    with col2:
        epochs = st.slider("Epochs", 10, 200, 60, step=10)
    with col3:
        lr = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
    with col4:
        rnn_type = st.selectbox("结构", ["RNN", "LSTM"], index=0)

    if st.button("开始训练", key="train_rnn"):
        try:
            result = train_char_rnn(
                text=train_text,
                hidden_size=hidden_size,
                epochs=epochs,
                lr=lr,
                rnn_type=rnn_type,
            )
            st.session_state["rnn_bundle"] = result
            st.session_state["rnn_meta"] = {
                "hidden_size": hidden_size,
                "epochs": epochs,
                "lr": lr,
                "rnn_type": rnn_type,
                "train_text_preview": train_text[:400],
            }
            st.success("模型训练完成，可在下方输入 Seed 生成文本。")
        except Exception as exc:  # noqa: BLE001
            st.error(f"训练失败: {exc}")

    generated_text = ""
    seed = ""
    if "rnn_bundle" in st.session_state:
        st.markdown("### 文本生成")
        c1, c2, c3 = st.columns(3)
        with c1:
            seed = st.text_input("Seed", value="h")
        with c2:
            gen_len = st.slider("生成长度", 20, 200, 50, step=10)
        with c3:
            temperature = st.slider("Temperature", 0.2, 1.5, 0.9, step=0.1)

        if st.button("生成文本", key="gen_rnn"):
            generated_text = generate_text(
                st.session_state["rnn_bundle"],
                seed=seed,
                gen_len=gen_len,
                temperature=temperature,
            )
            st.session_state["rnn_generated_text"] = generated_text

        if "rnn_generated_text" in st.session_state:
            st.code(st.session_state["rnn_generated_text"], language="text")

    meta = st.session_state.get("rnn_meta", {})
    bundle = st.session_state.get("rnn_bundle", {})
    final_loss = bundle.get("losses", [None])[-1] if bundle else None

    snapshot = {
        "rnn_type": meta.get("rnn_type", "N/A"),
        "hidden_size": meta.get("hidden_size", "N/A"),
        "epochs": meta.get("epochs", "N/A"),
        "lr": meta.get("lr", "N/A"),
        "final_loss": f"{final_loss:.6f}" if isinstance(final_loss, float) else "N/A",
        "train_text_preview": meta.get("train_text_preview", ""),
        "seed": seed,
        "generated_text": st.session_state.get("rnn_generated_text", generated_text),
    }

    st.session_state["report_state"]["module2"] = snapshot
    return snapshot


def render_module_3() -> Dict:
    st.subheader("模块 3: 预训练架构对比 (Masked LM vs Causal LM)")
    st.markdown("<div class='module-note'>BERT 使用 [MASK] 进行双向上下文填空；GPT-2 按从左到右的方式自回归续写。</div>", unsafe_allow_html=True)

    left, right = st.columns(2)

    bert_rows = []
    mask_input = "The man went to the [MASK] to buy some milk."

    with left:
        st.markdown("#### BERT: Masked LM")
        mask_input = st.text_input("输入含 [MASK] 的句子", value=mask_input)

        if st.button("运行 BERT Top-5 预测", key="bert_run"):
            if "[MASK]" not in mask_input:
                st.error("请在句子中包含 [MASK] 标记。")
            else:
                try:
                    fill_mask = load_bert_fill_mask()
                    outputs = fill_mask(mask_input, top_k=5)
                    bert_rows = [
                        {
                            "rank": i + 1,
                            "token": item["token_str"].strip(),
                            "probability": item["score"],
                        }
                        for i, item in enumerate(outputs)
                    ]
                    st.session_state["bert_rows"] = bert_rows
                except Exception as exc:  # noqa: BLE001
                    st.error(f"BERT 推理失败: {exc}")

        if "bert_rows" in st.session_state:
            bert_rows = st.session_state["bert_rows"]
            st.dataframe(pd.DataFrame(bert_rows), use_container_width=True)

    gpt_prompt = "In a future classroom, language models"
    with right:
        st.markdown("#### GPT-2: Causal LM")
        gpt_prompt = st.text_area("输入 Prompt", value=gpt_prompt, height=120)

        if st.button("运行 GPT-2 续写(20词)", key="gpt_run"):
            if not gpt_prompt.strip():
                st.error("请输入非空 Prompt。")
            else:
                try:
                    _, _, text_generator = load_gpt2_assets()
                    continuation, _ = gpt2_continue(gpt_prompt, text_generator, max_words=20)
                    st.session_state["gpt_continuation"] = continuation
                except Exception as exc:  # noqa: BLE001
                    st.error(f"GPT-2 续写失败: {exc}")

        if "gpt_continuation" in st.session_state:
            st.write("续写结果（后续 20 词）:")
            st.code(st.session_state["gpt_continuation"], language="text")

    snapshot = {
        "mask_input": mask_input,
        "bert_rows": st.session_state.get("bert_rows", bert_rows),
        "gpt_prompt": gpt_prompt,
        "gpt_continuation": st.session_state.get("gpt_continuation", ""),
    }

    st.session_state["report_state"]["module3"] = snapshot
    return snapshot


def render_module_4() -> Dict:
    st.subheader("模块 4: 语言模型评价 (GPT-2 困惑度 PPL)")
    st.markdown("<div class='module-note'>输入多行测试句子，逐句计算交叉熵损失和困惑度。PPL 越小，模型建模效果越好。</div>", unsafe_allow_html=True)

    default_eval_text = (
        "The weather is pleasant and the meeting starts at nine.\n"
        "milk random sky keyboard quickly and unless banana."
    )
    eval_text = st.text_area("输入待评估句子（每行一句）", value=default_eval_text, height=160)

    ppl_rows = []
    if st.button("计算 PPL", key="calc_ppl"):
        sentences = [line.strip() for line in eval_text.splitlines() if line.strip()]
        if not sentences:
            st.warning("请至少输入一句测试文本。")
        else:
            try:
                tokenizer, model, _ = load_gpt2_assets()
                df = calc_ppl_for_sentences(sentences, tokenizer, model)
                st.session_state["ppl_df"] = df
            except Exception as exc:  # noqa: BLE001
                st.error(f"PPL 计算失败: {exc}")

    if "ppl_df" in st.session_state:
        df = st.session_state["ppl_df"].copy()
        st.dataframe(df, use_container_width=True)
        ppl_rows = df.to_dict(orient="records")

    snapshot = {
        "ppl_rows": ppl_rows,
    }

    st.session_state["report_state"]["module4"] = snapshot
    return snapshot


init_report_state()

st.markdown(
    """
    <div class="hero">
      <h1>综合交互式语言模型 Web 系统</h1>
      <p>集成统计 n 元模型、从零 RNN 训练、BERT/GPT-2 机制对比与 GPT-2 困惑度评价，并支持四模块静态 HTML 报告下载。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "模块1: n 元模型",
        "模块2: RNN 训练",
        "模块3: BERT vs GPT-2",
        "模块4: PPL 评价",
    ]
)

with tab1:
    render_module_1()

with tab2:
    render_module_2()

with tab3:
    render_module_3()

with tab4:
    render_module_4()

st.markdown("---")
st.markdown("### 下载静态 HTML（含四个模块）")
report_html = build_static_html(st.session_state["report_state"])

st.download_button(
    label="下载静态 HTML 报告",
    data=report_html.encode("utf-8"),
    file_name=f"language_model_lab_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
    mime="text/html",
    use_container_width=True,
)
