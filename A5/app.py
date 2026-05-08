import html
import importlib
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple

import requests
import spacy
import streamlit as st


st.set_page_config(
    page_title="Discourse Interactive Lab",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Sans:wght@400;600;700&display=swap');
    :root {
        --ink: #0f2f3a;
        --sub-ink: #355663;
        --line: #d7e6ea;
        --panel: rgba(255, 255, 255, 0.86);
        --mint: #d7f2dd;
        --sky: #d6ecff;
        --amber: #ffe7ba;
        --rose: #ffd7cf;
    }
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(85% 45% at 10% 0%, #dff5f2 0%, transparent 60%),
            radial-gradient(75% 55% at 100% 10%, #fff0d1 0%, transparent 65%),
            linear-gradient(180deg, #f4fafb 0%, #eef5f7 100%);
        color: var(--ink);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-shell {
        position: relative;
        padding: 1.3rem 1.2rem 1rem 1.2rem;
        border-radius: 18px;
        border: 1px solid rgba(17, 50, 77, 0.12);
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.88) 0%, rgba(233, 247, 255, 0.72) 100%);
        box-shadow: 0 10px 30px rgba(17, 50, 77, 0.08);
        backdrop-filter: blur(6px);
        animation: fadeUp 0.6s ease-out;
    }
    .app-title {
        font-family: 'Manrope', 'IBM Plex Sans', sans-serif;
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: 0.3px;
        color: #11324d;
        margin-bottom: 0.3rem;
    }
    .app-subtitle {
        color: #3a6170;
        margin-bottom: 1rem;
        font-size: 1.02rem;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .hero-badges span {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid #c8dfe8;
        background: #f8fcff;
        color: #244a59;
        font-size: 0.82rem;
        font-weight: 600;
    }
    [data-testid="stTabs"] {
        animation: fadeUp 0.75s ease-out;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.72);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #325261;
        font-weight: 700;
        padding: 8px 12px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #def2ff 0%, #f0ffe7 100%);
        color: #0f2f3a;
        border: 1px solid #c7dfeb;
    }
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #b8d2dc;
        font-weight: 700;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 5px 14px rgba(17, 50, 77, 0.12);
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(17, 50, 77, 0.18);
    }
    .stTextArea textarea,
    .stSelectbox [data-baseweb="select"] {
        border-radius: 10px;
    }
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.82);
    }
    .stAlert {
        border-radius: 12px;
    }
    .edu-grid {
        display: grid;
        gap: 10px;
    }
    .edu-card {
        border: 1px solid #cfe0e8;
        border-radius: 12px;
        padding: 10px 12px;
        background: var(--panel);
        line-height: 1.7;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.03);
        animation: fadeUp 0.5s ease-out;
    }
    .edu-card strong {
        color: #26495d;
    }
    .boundary-token {
        background: #ffcc66;
        border-radius: 5px;
        padding: 0 4px;
        font-weight: 700;
    }
    .tag {
        display: inline-block;
        margin: 2px 5px 2px 0;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #173f51;
        background: #dff0ff;
        border: 1px solid #c2deef;
    }
    .conn {
        font-weight: 800;
        border-radius: 5px;
        padding: 0 4px;
    }
    .cat-temporal { background: #ffe6be; color: #6a4300; }
    .cat-contingency { background: #d9f3dd; color: #1c5e2b; }
    .cat-comparison { background: #ffd7d0; color: #942f23; }
    .cat-expansion { background: #d9f0ea; color: #174e45; }
    .arg-box {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
        line-height: 1.7;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    .arg1 { background: #e8f5ff; border: 1px solid #9dcef2; }
    .arg2 { background: #fff3d9; border: 1px solid #ffd58a; }
    .cluster-chip {
        display: inline-block;
        margin: 3px 6px 3px 0;
        padding: 2px 8px;
        border-radius: 999px;
        background: #eceff1;
        font-size: 0.82rem;
    }
    .coref-view {
        border: 1px solid #cedee6;
        border-radius: 12px;
        padding: 12px;
        background: var(--panel);
        line-height: 1.8;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    .section-note {
        color: #4b6a79;
        font-size: 0.93rem;
        margin-bottom: 0.6rem;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 768px) {
        .app-title {
            font-size: 1.65rem;
            line-height: 1.35;
        }
        .hero-shell {
            padding: 1rem 0.9rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


NEURALEDUSEG_BASE = "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst"
DEFAULT_SAMPLE = {
    "DEV": ["wsj_0603.out"],
    "TRAINING": [
        "wsj_0601.out",
        "wsj_0604.out",
        "wsj_0605.out",
        "wsj_0606.out",
        "wsj_1101.out",
        "wsj_1102.out",
        "wsj_1103.out",
        "wsj_1104.out",
    ],
    "TEST": ["wsj_1105.out", "wsj_1106.out"],
}

FALLBACK_OUT = (
    "THE FINANCIAL ACCOUNTING STANDARDS BOARD'S coming rule on disclosure involving "
    "financial instruments will be effective for financial statements with fiscal years "
    "ending after June 15, 1990. The date was misstated in Friday's edition. "
    "(See: \"FASB Plans Rule on Financial Risk of Instruments\" -- WSJ Oct. 27, 1989)"
)
FALLBACK_EDUS = [
    "THE FINANCIAL ACCOUNTING STANDARDS BOARD'S coming rule on disclosure",
    "involving financial instruments",
    "will be effective for financial statements with fiscal years",
    "ending after June 15, 1990.",
    "The date was misstated in Friday's edition.",
    '(See: "FASB Plans Rule on Financial Risk of Instruments"',
    "-- WSJ Oct. 27, 1989)",
]

CONNECTIVE_TO_CLASS = {
    "when": "TEMPORAL",
    "after": "TEMPORAL",
    "before": "TEMPORAL",
    "while": "TEMPORAL",
    "because": "CONTINGENCY",
    "since": "CONTINGENCY",
    "so": "CONTINGENCY",
    "therefore": "CONTINGENCY",
    "thus": "CONTINGENCY",
    "although": "COMPARISON",
    "though": "COMPARISON",
    "but": "COMPARISON",
    "however": "COMPARISON",
    "whereas": "COMPARISON",
    "and": "EXPANSION",
    "or": "EXPANSION",
    "also": "EXPANSION",
    "moreover": "EXPANSION",
    "in addition": "EXPANSION",
}

CAT_CLASS = {
    "TEMPORAL": "cat-temporal",
    "CONTINGENCY": "cat-contingency",
    "COMPARISON": "cat-comparison",
    "EXPANSION": "cat-expansion",
}

SCONJ_WORDS = {
    "because",
    "since",
    "although",
    "though",
    "if",
    "unless",
    "while",
    "when",
    "after",
    "before",
    "whereas",
}

COREf_COLORS = [
    "#ffd8a8",
    "#d8f5a2",
    "#b2f2bb",
    "#a5d8ff",
    "#ffd6a5",
    "#ffc9c9",
    "#ffe066",
    "#c0eb75",
]

DEFAULT_COREF_TEXT = (
    "Barack Obama was born in Hawaii and became the 44th president of the United States. "
    "He studied at Columbia University and later at Harvard Law School. "
    "After graduation, Obama worked as a civil rights attorney in Chicago. "
    "His memoir was published years later, and it helped him gain national attention."
)


@st.cache_resource(show_spinner=False)
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 discourse-lab"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_neuraleduseg_sample(split: str, file_name: str) -> Dict[str, str]:
    out_url = f"{NEURALEDUSEG_BASE}/{split}/{file_name}"
    edus_url = f"{NEURALEDUSEG_BASE}/{split}/{file_name}.edus"
    pre_url = f"{NEURALEDUSEG_BASE}/{split}/{file_name}.preprecessed"

    errors: List[str] = []
    out_text = ""
    edus_text = ""
    pre_text = ""

    try:
        out_text = fetch_text(out_url)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"拉取原文失败: {exc}")

    try:
        edus_text = fetch_text(edus_url)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"拉取 .edus 失败: {exc}")

    try:
        pre_text = fetch_text(pre_url)
    except Exception:
        pre_text = ""

    if not out_text:
        out_text = FALLBACK_OUT
    if not edus_text and not pre_text:
        edus_text = "\n".join(FALLBACK_EDUS)

    return {
        "out_text": out_text,
        "edus_text": edus_text,
        "pre_text": pre_text,
        "out_url": out_url,
        "edus_url": edus_url,
        "pre_url": pre_url,
        "errors": "\n".join(errors),
    }


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_gt_edus(edus_text: str, pre_text: str) -> List[str]:
    preprocessed_edus: List[str] = []
    if pre_text:
        for line in pre_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_text = normalize_spaces(str(obj.get("raw_text", "")))
            if raw_text:
                preprocessed_edus.append(raw_text)

    if preprocessed_edus:
        return preprocessed_edus

    edus = []
    for line in edus_text.splitlines():
        clean = normalize_spaces(line)
        if clean:
            edus.append(clean)
    return edus


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|--|[^\w\s]", text)


def token_key(token: str) -> str:
    return token.lower().strip()


def find_subsequence_start(tokens: List[str], sub_tokens: List[str], start: int) -> int:
    if not sub_tokens:
        return -1
    limit = len(tokens) - len(sub_tokens) + 1
    for idx in range(max(start, 0), max(limit, 0)):
        match = True
        for shift, item in enumerate(sub_tokens):
            if token_key(tokens[idx + shift]) != token_key(item):
                match = False
                break
        if match:
            return idx
    return -1


def align_gt_boundaries(raw_text: str, gt_edus: List[str]) -> Tuple[List[List[str]], List[str]]:
    full_tokens = simple_tokenize(raw_text)
    pointer = 0
    segmented: List[List[str]] = []
    boundary_tokens: List[str] = []

    for edu in gt_edus:
        edu_tokens = simple_tokenize(edu)
        if not edu_tokens:
            continue

        start = find_subsequence_start(full_tokens, edu_tokens, pointer)
        if start == -1:
            segmented.append(edu_tokens)
            boundary_tokens.append(edu_tokens[-1])
            continue

        end = start + len(edu_tokens)
        chunk = full_tokens[start:end]
        segmented.append(chunk)
        boundary_tokens.append(chunk[-1])
        pointer = end

    if not segmented:
        segmented = [simple_tokenize(line) for line in gt_edus if line.strip()]
        boundary_tokens = [seg[-1] for seg in segmented if seg]

    return segmented, boundary_tokens


def segment_by_rules(text: str, nlp) -> Tuple[List[List[str]], List[str]]:
    doc = nlp(text)
    tokens = [tok.text for tok in doc]
    boundaries = set()

    for i, tok in enumerate(doc):
        t = tok.text

        if t in {".", "?", "!", ";", ":"}:
            boundaries.add(i)

        if t in {",", "--", "-"} and i < len(doc) - 1:
            nxt = doc[i + 1]
            if nxt.pos_ in {"SCONJ", "PRON", "PROPN", "DET", "NOUN"}:
                boundaries.add(i)

        if tok.pos_ == "SCONJ" and tok.lower_ in SCONJ_WORDS and i > 0:
            boundaries.add(i - 1)

        if tok.dep_ in {"advcl", "ccomp", "xcomp", "relcl", "acl"} and tok.pos_ in {"VERB", "AUX"} and i > 0:
            boundaries.add(i - 1)

    if tokens:
        boundaries.add(len(tokens) - 1)

    segments: List[List[str]] = []
    start = 0
    ordered = sorted(idx for idx in boundaries if idx >= 0)
    for b_idx in ordered:
        if b_idx < start:
            continue
        chunk = tokens[start : b_idx + 1]
        if chunk:
            segments.append(chunk)
        start = b_idx + 1

    if start < len(tokens):
        tail = tokens[start:]
        if tail:
            segments.append(tail)

    boundary_tokens = [seg[-1] for seg in segments if seg]
    return segments, boundary_tokens


def tokens_to_html(tokens: List[str], highlight_index: int = -1) -> str:
    no_space_before = {".", ",", ";", ":", "?", "!", ")", "]", "}", "%", "'s"}
    no_space_after = {"(", "[", "{", "\""}

    pieces: List[str] = []
    for idx, token in enumerate(tokens):
        safe = html.escape(token)
        if idx == highlight_index:
            safe = f'<span class="boundary-token">{safe}</span>'

        if idx > 0 and token not in no_space_before and tokens[idx - 1] not in no_space_after:
            pieces.append(" ")
        pieces.append(safe)

    return "".join(pieces)


def render_edu_cards(segmented_tokens: List[List[str]]) -> str:
    cards: List[str] = ["<div class='edu-grid'>"]
    for i, seg in enumerate(segmented_tokens, start=1):
        body = tokens_to_html(seg, highlight_index=len(seg) - 1)
        cards.append(f"<div class='edu-card'><strong>EDU {i}</strong><br>{body}</div>")
    cards.append("</div>")
    return "".join(cards)


def find_connectives(sentence: str) -> List[Dict[str, str]]:
    matches: List[Dict[str, str]] = []
    occupied: List[Tuple[int, int]] = []

    phrases = sorted(CONNECTIVE_TO_CLASS.keys(), key=len, reverse=True)
    for phrase in phrases:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", flags=re.IGNORECASE)
        for mt in pattern.finditer(sentence):
            s, e = mt.start(), mt.end()
            overlap = any((s < oe and e > os) for os, oe in occupied)
            if overlap:
                continue
            occupied.append((s, e))
            matches.append(
                {
                    "text": sentence[s:e],
                    "category": CONNECTIVE_TO_CLASS[phrase],
                    "start": s,
                    "end": e,
                }
            )

    matches.sort(key=lambda x: x["start"])
    return matches


def highlight_connectives(sentence: str, matches: List[Dict[str, str]]) -> str:
    if not matches:
        return html.escape(sentence)

    parts: List[str] = []
    cursor = 0
    for item in matches:
        s = int(item["start"])
        e = int(item["end"])
        cat = str(item["category"])
        cat_css = CAT_CLASS.get(cat, "cat-expansion")
        parts.append(html.escape(sentence[cursor:s]))
        highlighted = html.escape(sentence[s:e])
        parts.append(
            f'<span class="conn {cat_css}">{highlighted} [{cat}]</span>'
        )
        cursor = e
    parts.append(html.escape(sentence[cursor:]))
    return "".join(parts)


def split_arguments(sentence: str, matches: List[Dict[str, str]]) -> Tuple[str, str, str]:
    if not matches:
        return sentence.strip(), "", ""

    first = matches[0]
    s = int(first["start"])
    e = int(first["end"])
    conn = sentence[s:e]

    if s <= 2:
        after = sentence[e:].strip()
        comma = after.find(",")
        if comma != -1:
            arg2 = after[:comma].strip()
            arg1 = after[comma + 1 :].strip()
            return arg1, arg2, conn

    arg1 = sentence[:s].strip(" ,;-")
    arg2 = sentence[e:].strip(" ,;-")
    return arg1, arg2, conn


@st.cache_resource(show_spinner=False)
def load_fastcoref_model():
    from fastcoref import FCoref

    try:
        return FCoref(device="cpu")
    except AttributeError as exc:
        # 兼容 fastcoref 与较新 transformers 的字段差异
        if "all_tied_weights_keys" not in str(exc):
            raise

        _patch_fastcoref_model_class()
        return FCoref(device="cpu")


def _patch_fastcoref_model_class() -> None:
    module_candidates = [
        "fastcoref.modeling",
        "fastcoref.modeling_fcoref",
        "fastcoref.coref_models.modeling",
    ]

    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        model_cls = getattr(module, "FCorefModel", None)
        if model_cls is None:
            continue

        if not hasattr(model_cls, "all_tied_weights_keys"):
            setattr(model_cls, "all_tied_weights_keys", [])
        if not hasattr(model_cls, "_tied_weights_keys"):
            setattr(model_cls, "_tied_weights_keys", [])


def build_coref_html(text: str, char_clusters: List[List[Tuple[int, int]]]) -> str:
    mentions = []
    for cid, spans in enumerate(char_clusters):
        for start, end in spans:
            if 0 <= start < end <= len(text):
                mentions.append({"cid": cid, "start": start, "end": end})

    mentions.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    filtered = []
    current_end = -1
    for item in mentions:
        if item["start"] < current_end:
            continue
        filtered.append(item)
        current_end = item["end"]

    result: List[str] = []
    cursor = 0
    for m in filtered:
        result.append(html.escape(text[cursor : m["start"]]))
        color = COREf_COLORS[m["cid"] % len(COREf_COLORS)]
        mention = html.escape(text[m["start"] : m["end"]])
        result.append(
            f'<span style="background:{color};border-radius:4px;padding:0 3px;">{mention}</span>'
        )
        cursor = m["end"]
    result.append(html.escape(text[cursor:]))

    return "".join(result)


def run_coref(text: str):
    model = load_fastcoref_model()
    pred = model.predict(texts=[text])[0]
    char_clusters = pred.get_clusters(as_strings=False)
    str_clusters = pred.get_clusters()
    html_view = build_coref_html(text, char_clusters)
    return str_clusters, html_view


def build_static_html_report(m1: Dict, m2: Dict, m3: Dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    m1_block = '<section class="report-panel"><h2>标签页 1: 话语分割 (EDU)</h2><p>模块 1 未生成结果。</p></section>'
    if m1:
        m1_block = f"""
        <section class="report-panel">
        <h2>标签页 1: 话语分割 (EDU)</h2>
        <div class="meta-row">
            <span class="badge">数据源: {html.escape(m1.get('source', ''))}</span>
        </div>
        <p><strong>原文:</strong> {html.escape(m1.get('raw_text', ''))}</p>
        <div class="report-grid">
            <div><h3>规则基线</h3>{m1.get('baseline_html', '')}</div>
            <div><h3>NeuralEDUSeg 真实标注</h3>{m1.get('gt_html', '')}</div>
        </div>
        </section>
        """

    m2_block = '<section class="report-panel"><h2>标签页 2: 浅层篇章关系</h2><p>模块 2 未生成结果。</p></section>'
    if m2:
        m2_block = f"""
        <section class="report-panel">
        <h2>标签页 2: 浅层篇章关系</h2>
        <p><strong>输入句子:</strong> {html.escape(m2.get('sentence', ''))}</p>
        <p><strong>连接词标注:</strong></p>
        <div class="coref-view">{m2.get('highlighted_sentence', '')}</div>
        <div style="height:8px"></div>
        <div class="arg-box arg1"><strong>Arg1</strong><br>{html.escape(m2.get('arg1', ''))}</div>
        <div class="arg-box arg2"><strong>Arg2</strong><br>{html.escape(m2.get('arg2', ''))}</div>
        </section>
        """

    m3_block = '<section class="report-panel"><h2>标签页 3: 指代消解</h2><p>模块 3 未生成结果。</p></section>'
    if m3:
        cluster_rows = []
        for i, clu in enumerate(m3.get("clusters", []), start=1):
            cluster_rows.append(f"<li><span class=\"badge\">Cluster {i}</span> {html.escape(str(clu))}</li>")
        cluster_html = "<ul>" + "".join(cluster_rows) + "</ul>" if cluster_rows else "<p>未识别到簇。</p>"

        m3_block = f"""
        <section class="report-panel">
        <h2>标签页 3: 指代消解</h2>
        <p><strong>输入文本:</strong></p>
        <div class="report-plain">{html.escape(m3.get('text', ''))}</div>
        <p><strong>高亮结果:</strong></p>
        <div class="coref-view">{m3.get('highlighted_html', '')}</div>
        <p><strong>簇列表:</strong></p>
        {cluster_html}
        </section>
        """

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Discourse Interactive Lab Report</title>
      <style>
                @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Sans:wght@400;600;700&display=swap');
                :root {{
                        --ink: #123340;
                        --sub-ink: #3d5f6b;
                        --line: #d5e5ec;
                        --panel: rgba(255, 255, 255, 0.9);
                }}
                * {{ box-sizing: border-box; }}
                body {{
                        font-family: 'IBM Plex Sans', 'PingFang SC', 'Microsoft YaHei', sans-serif;
                        margin: 0;
                        color: var(--ink);
                        background:
                                radial-gradient(75% 38% at 5% 0%, #dff5f2 0%, transparent 60%),
                                radial-gradient(78% 55% at 100% 10%, #fff0d1 0%, transparent 65%),
                                linear-gradient(180deg, #f4fafb 0%, #edf4f6 100%);
                }}
                .report-shell {{
                        max-width: 1180px;
                        margin: 24px auto;
                        padding: 0 14px 30px 14px;
                }}
                .report-hero {{
                        border: 1px solid rgba(17, 50, 77, 0.12);
                        border-radius: 16px;
                        background: linear-gradient(120deg, rgba(255, 255, 255, 0.92) 0%, rgba(233, 247, 255, 0.72) 100%);
                        padding: 18px 16px;
                        box-shadow: 0 10px 26px rgba(17, 50, 77, 0.08);
                        margin-bottom: 14px;
                }}
                h1 {{
                        margin: 0 0 8px 0;
                        font-family: 'Manrope', 'IBM Plex Sans', sans-serif;
                        font-size: 2rem;
                        letter-spacing: 0.2px;
                        color: #11324d;
                }}
                .time-line {{ color: var(--sub-ink); margin: 0; }}
                .report-panel {{
                        border: 1px solid var(--line);
                        border-radius: 14px;
                        padding: 14px;
                        margin-top: 14px;
                        background: var(--panel);
                        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.04);
                }}
                h2 {{ margin: 0 0 10px 0; color: #234b5e; }}
                h3 {{ color: #2a5367; }}
                .meta-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
                .badge {{
                        display: inline-block;
                        padding: 4px 9px;
                        border-radius: 999px;
                        font-size: 0.82rem;
                        font-weight: 700;
                        background: #dff0ff;
                        border: 1px solid #c2deef;
                        color: #173f51;
                }}
                .report-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
                .edu-card {{ border:1px solid #cfe0e8;border-radius:12px;padding:10px 12px;background:#f8fcff;line-height:1.7;margin-bottom:8px; }}
                .boundary-token {{ background:#ffcc66;border-radius:5px;padding:0 4px;font-weight:700; }}
                .conn {{ font-weight:700;border-radius:5px;padding:0 4px; }}
                .cat-temporal {{ background:#ffe6be;color:#6a4300; }}
                .cat-contingency {{ background:#d9f3dd;color:#1c5e2b; }}
                .cat-comparison {{ background:#ffd7d0;color:#942f23; }}
                .cat-expansion {{ background:#d9f0ea;color:#174e45; }}
                .arg-box {{ border-radius: 12px; padding: 12px; margin-bottom: 8px; }}
                .arg1 {{ background: #e8f5ff; border: 1px solid #9dcef2; }}
                .arg2 {{ background: #fff3d9; border: 1px solid #ffd58a; }}
                .coref-view {{ border: 1px solid #cedee6; border-radius: 12px; padding: 12px; background: #ffffff; line-height: 1.85; }}
                .report-plain {{ border: 1px dashed #c5d8df; border-radius: 10px; background: #f9fdff; padding: 10px; line-height: 1.8; }}
                ul {{ margin-top: 8px; }}
                li {{ margin: 8px 0; }}
                @media (max-width: 900px) {{
                        .report-grid {{ grid-template-columns: 1fr; }}
                        h1 {{ font-size: 1.58rem; line-height: 1.35; }}
                        .report-shell {{ margin: 14px auto; }}
                }}
      </style>
    </head>
    <body>
            <div class="report-shell">
                <div class="report-hero">
                    <h1>综合篇章分析系统导出报告</h1>
                    <p class="time-line">生成时间: {now}</p>
                </div>
                {m1_block}
                {m2_block}
                {m3_block}
            </div>
    </body>
    </html>
    """


st.markdown(
    """
    <div class="hero-shell">
        <div class="app-title">综合篇章分析交互系统</div>
        <div class="app-subtitle">模块包含: EDU 话语分割对比、浅层篇章关系提取、指代消解可视化</div>
        <div class="hero-badges">
            <span>NeuralEDUSeg 数据对比</span>
            <span>PDTB 显式关系</span>
            <span>fastcoref 指代簇</span>
            <span>静态 HTML 导出</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

nlp = load_spacy_model()

if "m1_state" not in st.session_state:
    st.session_state.m1_state = {"split": "DEV", "file": "wsj_0603.out"}
if "m1_result" not in st.session_state:
    st.session_state.m1_result = {}
if "m2_result" not in st.session_state:
    st.session_state.m2_result = {}
if "m3_result" not in st.session_state:
    st.session_state.m3_result = {}


tab1, tab2, tab3 = st.tabs(
    [
        "模块 1: EDU 话语分割",
        "模块 2: 显式关系提取",
        "模块 3: 指代消解",
    ]
)


with tab1:
    st.markdown("### 模块 1: 规则基线 vs NeuralEDUSeg 真实数据")
    st.markdown(
        '<div class="section-note">从 NeuralEDUSeg 样本抓取原文与真实 EDU 标注，并与 spaCy 规则基线进行对比。</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([1, 1, 0.8])
    with col_a:
        split = st.selectbox("数据划分", list(DEFAULT_SAMPLE.keys()), index=0)
    with col_b:
        file_options = DEFAULT_SAMPLE[split]
        default_idx = 0 if "wsj_0603.out" not in file_options else file_options.index("wsj_0603.out")
        file_name = st.selectbox("样本文件", file_options, index=default_idx)
    with col_c:
        load_btn = st.button("抓取并分析", use_container_width=True)

    if load_btn or not st.session_state.m1_result:
        st.session_state.m1_state = {"split": split, "file": file_name}
        data = fetch_neuraleduseg_sample(split=split, file_name=file_name)
        raw_text = normalize_spaces(data["out_text"])
        gt_edus = parse_gt_edus(data["edus_text"], data["pre_text"])

        baseline_segments, baseline_boundaries = segment_by_rules(raw_text, nlp)
        gt_segments, gt_boundaries = align_gt_boundaries(raw_text, gt_edus)

        baseline_html = render_edu_cards(baseline_segments)
        gt_html = render_edu_cards(gt_segments)

        st.session_state.m1_result = {
            "source": f"{split}/{file_name}",
            "raw_text": raw_text,
            "baseline_segments": baseline_segments,
            "gt_segments": gt_segments,
            "baseline_boundaries": baseline_boundaries,
            "gt_boundaries": gt_boundaries,
            "baseline_html": baseline_html,
            "gt_html": gt_html,
            "data_urls": [data["out_url"], data["edus_url"], data["pre_url"]],
            "errors": data["errors"],
        }

    m1 = st.session_state.m1_result
    if m1.get("errors"):
        st.warning(m1["errors"])

    st.text_area("原始文本", value=m1.get("raw_text", ""), height=120)
    st.caption("数据来源:")
    for item_url in m1.get("data_urls", []):
        st.write(item_url)

    left, right = st.columns(2)

    with left:
        st.markdown("#### 规则基线切分结果")
        st.markdown(m1.get("baseline_html", ""), unsafe_allow_html=True)
        chips = " ".join(
            [f"<span class='tag'>{html.escape(x)}</span>" for x in m1.get("baseline_boundaries", [])]
        )
        st.markdown(f"**Boundary Tokens**<br>{chips}", unsafe_allow_html=True)

    with right:
        st.markdown("#### NeuralEDUSeg 数据集真实标注")
        st.markdown(m1.get("gt_html", ""), unsafe_allow_html=True)
        chips = " ".join([f"<span class='tag'>{html.escape(x)}</span>" for x in m1.get("gt_boundaries", [])])
        st.markdown(f"**Boundary Tokens**<br>{chips}", unsafe_allow_html=True)


with tab2:
    st.markdown("### 模块 2: 浅层篇章分析与显式关系提取")
    st.markdown(
        '<div class="section-note">识别显式连接词并映射到 PDTB 顶级语义类别，同时进行简易 Arg1/Arg2 切分。</div>',
        unsafe_allow_html=True,
    )

    default_sentence = (
        "Third-quarter sales in Europe were exceptionally strong, boosted by promotional programs "
        "and new products - although weaker foreign currencies reduced the company's earnings."
    )
    sentence = st.text_area("输入英文句子", value=default_sentence, height=120)

    matches = find_connectives(sentence)
    highlighted = highlight_connectives(sentence, matches)
    arg1, arg2, trigger = split_arguments(sentence, matches)

    st.markdown("#### 连接词高亮")
    st.markdown(f"<div class='coref-view'>{highlighted}</div>", unsafe_allow_html=True)

    if matches:
        info = " ".join(
            [
                f"<span class='tag'>{html.escape(m['text'])} -> {html.escape(str(m['category']))}</span>"
                for m in matches
            ]
        )
        st.markdown(info, unsafe_allow_html=True)
    else:
        st.info("未匹配到预定义显式连接词。")

    st.markdown("#### Arg1 / Arg2 简易提取")
    st.markdown(f"<div class='arg-box arg1'><strong>Arg1</strong><br>{html.escape(arg1 or '(空)')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='arg-box arg2'><strong>Arg2</strong><br>{html.escape(arg2 or '(空)')}</div>", unsafe_allow_html=True)

    if trigger:
        st.caption(f"切分连接词: {trigger}")

    st.session_state.m2_result = {
        "sentence": sentence,
        "matches": matches,
        "highlighted_sentence": highlighted,
        "arg1": arg1,
        "arg2": arg2,
        "trigger": trigger,
    }


with tab3:
    st.markdown("### 模块 3: 指代消解 (Coreference Resolution)")
    st.markdown(
        '<div class="section-note">调用 fastcoref E2E 神经模型，提取指代簇并在原文中按簇上色高亮。</div>',
        unsafe_allow_html=True,
    )

    coref_text = st.text_area("输入文本", value=DEFAULT_COREF_TEXT, height=180)
    run_btn = st.button("运行指代消解", type="primary")

    if run_btn:
        try:
            with st.spinner("fastcoref 模型推理中..."):
                clusters, highlighted_html = run_coref(coref_text)

            st.session_state.m3_result = {
                "text": coref_text,
                "clusters": clusters,
                "highlighted_html": highlighted_html,
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            st.session_state.m3_result = {
                "text": coref_text,
                "clusters": [],
                "highlighted_html": html.escape(coref_text),
                "error": str(exc),
            }

    m3 = st.session_state.m3_result
    if m3:
        if m3.get("error"):
            extra_tip = ""
            if "all_tied_weights_keys" in str(m3["error"]):
                extra_tip = (
                    "\n检测到 fastcoref 与 transformers 版本兼容问题，"
                    "请执行: pip install -U fastcoref torch \"transformers<4.46\""
                )
            st.error(
                "fastcoref 运行失败。请先安装依赖并保证网络可下载模型。"
                f"\n错误信息: {m3['error']}"
                f"{extra_tip}"
            )

        st.markdown("#### 指代簇高亮")
        st.markdown(f"<div class='coref-view'>{m3.get('highlighted_html', '')}</div>", unsafe_allow_html=True)

        st.markdown("#### Coreference Clusters")
        clusters = m3.get("clusters", [])
        if clusters:
            for i, clu in enumerate(clusters, start=1):
                st.write(f"Cluster {i}: {clu}")
        else:
            st.info("当前未提取到指代簇。")


st.divider()
st.markdown("### 导出")
st.caption("下载当前页面状态对应的静态 HTML 报告，包含三个标签页的结果信息。")

html_report = build_static_html_report(
    m1=st.session_state.m1_result,
    m2=st.session_state.m2_result,
    m3=st.session_state.m3_result,
)

st.download_button(
    label="下载静态 HTML 报告",
    data=html_report,
    file_name="discourse_interactive_report.html",
    mime="text/html",
    use_container_width=True,
)
