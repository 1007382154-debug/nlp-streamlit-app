import html
import json
import os
import re
from pathlib import Path

import streamlit as st
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(SCRIPT_DIR / ".hf_cache"))

MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"
LOCAL_MODEL_DIR = Path("models") / MODEL_NAME.replace("/", "--")
MODEL_SOURCE = os.getenv("MT_MODEL_PATH") or (
    str(LOCAL_MODEL_DIR) if (LOCAL_MODEL_DIR / "source.spm").is_file() else MODEL_NAME
)

RULE_DICTIONARY = {
    "i": "我",
    "you": "你",
    "he": "他",
    "she": "她",
    "it": "它",
    "we": "我们",
    "they": "他们",
    "am": "是",
    "is": "是",
    "are": "是",
    "was": "是",
    "were": "是",
    "be": "是",
    "have": "有",
    "has": "有",
    "had": "有",
    "do": "做",
    "does": "做",
    "did": "做",
    "not": "不",
    "like": "喜欢",
    "likes": "喜欢",
    "love": "爱",
    "eat": "吃",
    "drink": "喝",
    "go": "去",
    "come": "来",
    "see": "看见",
    "think": "认为",
    "know": "知道",
    "want": "想要",
    "need": "需要",
    "read": "阅读",
    "reads": "阅读",
    "can": "能",
    "will": "将",
    "would": "会",
    "should": "应该",
    "the": "这个",
    "a": "一个",
    "an": "一个",
    "this": "这",
    "that": "那",
    "these": "这些",
    "those": "那些",
    "my": "我的",
    "your": "你的",
    "his": "他的",
    "her": "她的",
    "our": "我们的",
    "their": "他们的",
    "cat": "猫",
    "cats": "猫",
    "dog": "狗",
    "dogs": "狗",
    "rain": "雨",
    "rains": "下雨",
    "book": "书",
    "books": "书",
    "student": "学生",
    "students": "学生",
    "teacher": "老师",
    "school": "学校",
    "city": "城市",
    "country": "国家",
    "language": "语言",
    "translation": "翻译",
    "machine": "机器",
    "computer": "计算机",
    "important": "重要的",
    "beautiful": "美丽的",
    "good": "好的",
    "bad": "坏的",
    "big": "大的",
    "small": "小的",
    "fast": "快的",
    "slow": "慢的",
    "today": "今天",
    "tomorrow": "明天",
    "yesterday": "昨天",
    "because": "因为",
    "and": "和",
    "or": "或者",
    "but": "但是",
    "if": "如果",
    "when": "当",
    "who": "谁",
    "which": "哪个",
    "where": "哪里",
    "in": "在",
    "on": "在",
    "at": "在",
    "to": "到",
    "from": "从",
    "with": "和",
    "for": "为了",
    "of": "的",
}


APP_CSS = """
<style>
:root {
  --ink: #172033;
  --muted: #667085;
  --line: #d9e2ec;
  --soft: #f5f8fb;
  --accent: #1664d9;
  --accent-2: #139a74;
  --warn: #a15c00;
}
.stApp {
  background:
    radial-gradient(circle at 10% 0%, rgba(22, 100, 217, 0.12), transparent 28rem),
    linear-gradient(180deg, #f7fbff 0%, #ffffff 32rem);
  color: var(--ink);
}
.block-container {
  max-width: 1180px;
  padding-top: 2.2rem;
  padding-bottom: 3rem;
}
.hero {
  padding: 1.6rem 1.7rem;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.86);
  box-shadow: 0 16px 44px rgba(20, 43, 74, 0.08);
}
.hero h1 {
  margin: 0 0 .45rem 0;
  font-size: 2.05rem;
  line-height: 1.22;
  letter-spacing: 0;
}
.hero p {
  margin: 0;
  color: var(--muted);
  font-size: 1rem;
}
.module-note {
  padding: .9rem 1rem;
  border-left: 4px solid var(--accent);
  background: #eef5ff;
  border-radius: 6px;
  color: #24456f;
}
.compare-title {
  color: var(--muted);
  font-size: .92rem;
  margin-bottom: .35rem;
}
div[data-testid="stTabs"] button {
  font-weight: 700;
}
div[data-testid="stTextArea"] textarea {
  border-radius: 8px;
}
.stButton > button,
.stDownloadButton > button {
  border-radius: 8px;
  font-weight: 700;
}
.result-box {
  min-height: 88px;
  padding: 1rem;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  line-height: 1.7;
  white-space: pre-wrap;
}
.small-muted {
  color: var(--muted);
  font-size: .92rem;
}
</style>
"""


@st.cache_resource(show_spinner=False)
def load_translation_pipeline():
    from transformers import pipeline

    return pipeline("translation", model=MODEL_SOURCE)


def nmt_translate(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    translator = load_translation_pipeline()
    result = translator(cleaned, max_length=256)
    return result[0]["translation_text"]


def rule_based_translate(text: str) -> str:
    tokens = re.findall(r"[A-Za-z']+|[0-9]+|[^\w\s]", text)
    translated_tokens = []

    for token in tokens:
        key = token.lower()
        if re.fullmatch(r"[^\w\s]", token):
            translated_tokens.append(token)
        else:
            translated_tokens.append(RULE_DICTIONARY.get(key, token))

    return (
        " ".join(translated_tokens)
        .replace(" ,", "，")
        .replace(" .", "。")
        .replace(" ?", "？")
        .replace(" !", "！")
    )


def tokenize_for_bleu(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    if " " in cleaned:
        return [token for token in re.split(r"\s+", cleaned) if token]

    return [char for char in cleaned if not char.isspace()]


def calculate_bleu(reference: str, candidate: str) -> float:
    reference_tokens = tokenize_for_bleu(reference)
    candidate_tokens = tokenize_for_bleu(candidate)

    if not reference_tokens or not candidate_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)


def model_error_hint(exc: Exception) -> str:
    message = str(exc)
    if "huggingface-hub" in message or "require_version" in message:
        return (
            "当前 conda 环境中的 Hugging Face 依赖版本不匹配。请在 `%ENV_NAME%` 环境中执行：\n\n"
            "`python -m pip install -U -r requirements.txt`"
        )
    if "Connection" in message or "resolve" in message or "timed out" in message or "not the path to a directory" in message:
        return (
            "首次运行需要从 Hugging Face 下载模型 `Helsinki-NLP/opus-mt-en-zh`。"
            "请确认当前环境可以联网访问 Hugging Face，或提前把模型缓存到本机。"
        )
    return "请展开错误详情查看底层异常，并确认依赖已安装在正在运行 Streamlit 的 conda 环境中。"


def translate_with_feedback(text: str, spinner_text: str) -> str:
    if not text.strip():
        st.info("请输入英文句子后再生成译文。")
        return ""

    try:
        with st.spinner(spinner_text):
            return nmt_translate(text)
    except Exception as exc:
        st.error("神经翻译模型加载失败。")
        st.info(model_error_hint(exc))
        with st.expander("错误详情"):
            st.code(str(exc))
        return ""


def render_result(label: str, value: str, muted: str = ""):
    st.markdown(f"<div class='compare-title'>{html.escape(label)}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='result-box'>{html.escape(value or '暂无结果')}</div>",
        unsafe_allow_html=True,
    )
    if muted:
        st.markdown(f"<div class='small-muted'>{html.escape(muted)}</div>", unsafe_allow_html=True)


def build_static_html() -> str:
    nmt_source = st.session_state.get("nmt_source", "It rains cats and dogs.")
    nmt_result = st.session_state.get("latest_nmt", "") or "示例译文：雨下得很大。"
    compare_source = st.session_state.get(
        "compare_source", "The student who likes machine translation reads a good book."
    )
    rule_result = rule_based_translate(compare_source)
    compare_nmt = st.session_state.get("compare_nmt", "") or "示例译文：喜欢机器翻译的学生读了一本好书。"
    bleu_source = st.session_state.get("bleu_source", "It rains cats and dogs.")
    reference = st.session_state.get("bleu_reference", "雨下得很大。")
    candidate = st.session_state.get("bleu_candidate", nmt_result)
    dictionary_json = json.dumps(RULE_DICTIONARY, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>机器翻译机制对比与评测平台</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #667085;
      --line: #d9e2ec;
      --soft: #f5f8fb;
      --accent: #1664d9;
      --accent-2: #139a74;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f7fbff 0%, #ffffff 420px);
    }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 34px 20px 56px; }}
    .hero {{
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,.9);
      box-shadow: 0 16px 44px rgba(20,43,74,.08);
    }}
    h1 {{ margin: 0 0 10px; font-size: 32px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 16px; font-size: 22px; letter-spacing: 0; }}
    p {{ color: var(--muted); line-height: 1.75; }}
    section {{
      margin-top: 24px;
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    label {{ display: block; margin-bottom: 8px; font-weight: 700; }}
    textarea {{
      width: 100%;
      min-height: 92px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      resize: vertical;
      font: inherit;
      line-height: 1.6;
    }}
    button {{
      margin-top: 12px;
      border: 0;
      border-radius: 8px;
      padding: 10px 16px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }}
    .result {{
      min-height: 82px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      line-height: 1.7;
      white-space: pre-wrap;
    }}
    .score {{
      display: inline-flex;
      align-items: baseline;
      gap: 8px;
      margin-top: 12px;
      padding: 12px 16px;
      border-radius: 8px;
      background: #ecfdf5;
      color: #075f48;
      font-weight: 800;
    }}
    .note {{
      padding: 12px 14px;
      border-left: 4px solid var(--accent);
      border-radius: 6px;
      background: #eef5ff;
      color: #24456f;
    }}
    @media (max-width: 760px) {{
      .grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 26px; }}
      section {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="hero">
      <h1>机器翻译机制对比与评测平台</h1>
      <p>静态导出版保留三个教学模块的前端风格。NMT 模块展示导出时的译文结果；规则直译与 BLEU 模块可在浏览器中直接交互。</p>
    </div>

    <section>
      <h2>模块 1：神经机器翻译引擎</h2>
      <label>英文原文</label>
      <textarea readonly>{html.escape(nmt_source)}</textarea>
      <label>NMT 中文译文</label>
      <div class="result">{html.escape(nmt_result)}</div>
      <p class="note">静态 HTML 不连接后端模型；这里展示的是导出页面时保存的 NMT 结果或教学示例。</p>
    </section>

    <section>
      <h2>模块 2：基于规则的直译 vs 神经网络意译</h2>
      <label for="ruleInput">英文句子</label>
      <textarea id="ruleInput">{html.escape(compare_source)}</textarea>
      <button onclick="runRule()">生成规则直译</button>
      <div class="grid" style="margin-top:18px;">
        <div>
          <label>基于规则的逐词直译</label>
          <div id="ruleOutput" class="result">{html.escape(rule_result)}</div>
        </div>
        <div>
          <label>神经机器翻译结果</label>
          <div class="result">{html.escape(compare_nmt)}</div>
        </div>
      </div>
    </section>

    <section>
      <h2>模块 3：BLEU 自动评测</h2>
      <label>待翻译英文原文</label>
      <textarea>{html.escape(bleu_source)}</textarea>
      <div class="grid">
        <div>
          <label for="reference">Reference 标准译文</label>
          <textarea id="reference">{html.escape(reference)}</textarea>
        </div>
        <div>
          <label for="candidate">Candidate 候选译文</label>
          <textarea id="candidate">{html.escape(candidate)}</textarea>
        </div>
      </div>
      <button onclick="runBleu()">计算 BLEU 分数</button>
      <div id="bleuScore" class="score">BLEU Score: --</div>
      <p>BLEU 依赖 n-gram 表面匹配。语序错乱会降低高阶 n-gram 命中；语义相近但同义改写也可能得分偏低。</p>
    </section>
  </main>

  <script>
    const dictionary = {dictionary_json};

    function ruleTranslate(text) {{
      const tokens = text.match(/[A-Za-z']+|[0-9]+|[^\\w\\s]/g) || [];
      return tokens.map(token => {{
        if (/^[^\\w\\s]$/.test(token)) return token;
        return dictionary[token.toLowerCase()] || token;
      }}).join(' ')
        .replaceAll(' ,', '，')
        .replaceAll(' .', '。')
        .replaceAll(' ?', '？')
        .replaceAll(' !', '！');
    }}

    function tokenize(text) {{
      const trimmed = text.trim();
      if (!trimmed) return [];
      if (trimmed.includes(' ')) return trimmed.split(/\\s+/).filter(Boolean);
      return Array.from(trimmed).filter(ch => !/\\s/.test(ch));
    }}

    function ngrams(tokens, n) {{
      const counts = new Map();
      for (let i = 0; i <= tokens.length - n; i++) {{
        const key = tokens.slice(i, i + n).join('\\u0001');
        counts.set(key, (counts.get(key) || 0) + 1);
      }}
      return counts;
    }}

    function bleu(reference, candidate) {{
      const ref = tokenize(reference);
      const cand = tokenize(candidate);
      if (!ref.length || !cand.length) return 0;
      const precisions = [];
      for (let n = 1; n <= 4; n++) {{
        const refCounts = ngrams(ref, n);
        const candCounts = ngrams(cand, n);
        let clipped = 0;
        let total = 0;
        candCounts.forEach((count, key) => {{
          clipped += Math.min(count, refCounts.get(key) || 0);
          total += count;
        }});
        precisions.push(total ? (clipped + 0.1) / (total + 0.1) : 0.1);
      }}
      const bp = cand.length > ref.length ? 1 : Math.exp(1 - ref.length / cand.length);
      const geo = Math.exp(precisions.reduce((sum, p) => sum + Math.log(p), 0) / 4);
      return bp * geo;
    }}

    function runRule() {{
      document.getElementById('ruleOutput').textContent = ruleTranslate(document.getElementById('ruleInput').value);
    }}

    function runBleu() {{
      const score = bleu(document.getElementById('reference').value, document.getElementById('candidate').value);
      document.getElementById('bleuScore').textContent = `BLEU Score: ${{score.toFixed(4)}}`;
    }}
  </script>
</body>
</html>
"""


st.set_page_config(
    page_title="机器翻译机制对比与评测平台",
    page_icon="🌐",
    layout="wide",
)
st.markdown(APP_CSS, unsafe_allow_html=True)

for key, default in {
    "latest_nmt": "",
    "compare_rule": "",
    "compare_nmt": "",
    "last_bleu": None,
    "last_bleu_explanation": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

top_left, top_right = st.columns([0.72, 0.28], vertical_alignment="center")
with top_left:
    st.markdown(
        """
        <div class="hero">
          <h1>机器翻译机制对比与评测平台</h1>
          <p>通过神经机器翻译、规则直译和 BLEU 自动评测，观察机器翻译系统的架构差异与评价边界。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_right:
    st.download_button(
        "下载静态 HTML 页面",
        data=build_static_html(),
        file_name="machine_translation_platform.html",
        mime="text/html",
        use_container_width=True,
    )
    st.caption("静态页包含三个模块，规则直译与 BLEU 支持离线交互。")

st.info(f"当前为真实 NMT 模式：`transformers.pipeline('translation', model='{MODEL_SOURCE}')`。首次运行会下载 Hugging Face 模型。")

tab_nmt, tab_compare, tab_bleu = st.tabs(
    ["模块 1：NMT Engine", "模块 2：规则直译 vs NMT", "模块 3：BLEU 自动评测"]
)

with tab_nmt:
    st.subheader("神经机器翻译引擎")
    st.caption(f"当前模型：`{MODEL_SOURCE}`，加载方式：`transformers.pipeline('translation', model=MODEL_SOURCE)`")
    source_text = st.text_area(
        "输入英文句子",
        value="It rains cats and dogs.",
        height=120,
        key="nmt_source",
    )

    col_action, col_hint = st.columns([0.32, 0.68], vertical_alignment="center")
    with col_action:
        if st.button("生成 NMT 译文", type="primary", key="nmt_button", use_container_width=True):
            st.session_state.latest_nmt = translate_with_feedback(
                source_text,
                "正在加载模型并生成神经机器翻译结果...",
            )
    with col_hint:
        st.markdown(
            "<div class='module-note'>观察建议：尝试输入英语俚语、长句、从句或一词多义表达，观察 NMT 是否能利用上下文生成自然译文。</div>",
            unsafe_allow_html=True,
        )

    render_result("NMT 中文译文", st.session_state.latest_nmt)

with tab_compare:
    st.subheader("基于规则的逐词直译与神经网络意译对比")
    compare_text = st.text_area(
        "输入英文句子",
        value="The student who likes machine translation reads a good book.",
        height=120,
        key="compare_source",
    )

    if st.button("生成对比结果", type="primary", key="compare_button", use_container_width=True):
        st.session_state.compare_rule = rule_based_translate(compare_text)
        st.session_state.compare_nmt = translate_with_feedback(
            compare_text,
            "正在加载模型并生成神经机器翻译结果...",
        )
        if st.session_state.compare_nmt:
            st.session_state.latest_nmt = st.session_state.compare_nmt

    left, right = st.columns(2)
    with left:
        render_result(
            "基于规则的逐词直译",
            st.session_state.compare_rule,
            "只按词典逐词替换，不处理语序、搭配和上下文。",
        )
    with right:
        render_result(
            "神经机器翻译",
            st.session_state.compare_nmt,
            "依赖上下文建模，通常能更好处理习语、从句和自然表达。",
        )

    st.markdown(
        "<div class='module-note'>思考方向：逐词直译几乎不处理语序、搭配、上下文和多义词，因此遇到定语从句、习语和长距离依赖时会明显失真。</div>",
        unsafe_allow_html=True,
    )

with tab_bleu:
    st.subheader("机器翻译质量自动评测：BLEU Score")
    bleu_source = st.text_area(
        "1. 待翻译英文原文",
        value="It rains cats and dogs.",
        height=96,
        key="bleu_source",
    )
    reference = st.text_area(
        "2. 标准中文参考译文 Reference",
        value="雨下得很大。",
        height=96,
        key="bleu_reference",
    )

    if st.button("调用 NMT 生成候选译文", key="bleu_generate", use_container_width=True):
        generated = translate_with_feedback(
            bleu_source,
            "正在加载模型并生成候选译文...",
        )
        if generated:
            st.session_state.latest_nmt = generated
            st.session_state.bleu_candidate = generated

    candidate_default = st.session_state.latest_nmt or "下着倾盆大雨。"
    candidate = st.text_area(
        "3. 机器生成候选译文 Candidate",
        value=candidate_default,
        height=96,
        key="bleu_candidate",
    )

    if st.button("计算 BLEU 分数", type="primary", key="bleu_button", use_container_width=True):
        bleu = calculate_bleu(reference, candidate)
        st.session_state.last_bleu = bleu

        if bleu >= 0.7:
            explanation = "候选译文与参考译文的 n-gram 重合度较高，表面词序和用词较接近。"
        elif bleu >= 0.3:
            explanation = "候选译文与参考译文有一定重合，但可能存在词序、用词或表达差异。"
        else:
            explanation = "候选译文与参考译文的 n-gram 重合较少，不一定代表语义完全错误，可能是同义改写导致分数偏低。"
        st.session_state.last_bleu_explanation = explanation

    score_col, note_col = st.columns([0.32, 0.68], vertical_alignment="center")
    with score_col:
        if st.session_state.last_bleu is None:
            st.metric("BLEU Score", "--")
        else:
            st.metric("BLEU Score", f"{st.session_state.last_bleu:.4f}")
    with note_col:
        st.markdown(
            f"<div class='module-note'>{html.escape(st.session_state.last_bleu_explanation or 'BLEU 基于候选译文与参考译文之间的 n-gram 匹配，并带有长度惩罚。')}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "BLEU 适合快速比较大量机器译文，但对中文同义词替换、合理改写和语义等价表达不够敏感，因此不能完全替代人工评价。"
    )
