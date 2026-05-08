import io
import json
import os
import sys
import zipfile
from datetime import datetime
import html

from flask import Flask, jsonify, render_template, request, send_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from extractor import extract_pipeline

app = Flask(__name__)
TEMPLATE_HTML_DIR = os.path.join(BASE_DIR, "template_html")


def _html_table(headers, rows):
        if not rows:
                return '<p class="empty">暂无数据</p>'

        head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        body_rows = []
        for row in rows:
                cols = "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
                body_rows.append(f"<tr>{cols}</tr>")

        return (
                f"<table class='tbl'><thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(body_rows)}</tbody></table>"
        )


def _build_graph_data(result):
        node_map = {}
        for ent in result.get("entities", []):
                key = ent.get("text")
                if not key:
                        continue
                if key not in node_map:
                        node_map[key] = {
                                "id": key,
                                "name": key,
                                "category": ent.get("label", "UNKNOWN"),
                                "symbolSize": 58 if ent.get("label") == "ORG" else 52,
                        }

        links = []
        for rel in result.get("relations", []):
                source = rel.get("source")
                target = rel.get("target")
                relation = rel.get("relation", "RELATED_TO")
                if not source or not target:
                        continue
                links.append({"source": source, "target": target, "relation": relation})
                if source not in node_map:
                        node_map[source] = {"id": source, "name": source, "category": "UNKNOWN", "symbolSize": 48}
                if target not in node_map:
                        node_map[target] = {"id": target, "name": target, "category": "UNKNOWN", "symbolSize": 48}

        return {"nodes": list(node_map.values()), "links": links}


def build_static_html_report(text, result):
        entity_rows = [
                [e.get("text", ""), e.get("label", ""), e.get("start", ""), e.get("end", "")]
                for e in result.get("entities", [])
        ]
        relation_rows = [
                [r.get("source", ""), r.get("relation", ""), r.get("target", "")]
                for r in result.get("relations", [])
        ]

        bio_rows_raw = result.get("bio_tags", [])
        bio_rows = [[b.get("token", ""), b.get("tag", "")] for b in bio_rows_raw[:400]]
        bio_note = "" if len(bio_rows_raw) <= 400 else f"（已截断显示前 400 条，完整共 {len(bio_rows_raw)} 条）"

        graph_data = _build_graph_data(result)
        graph_json = json.dumps(graph_data, ensure_ascii=False)

        entity_table = _html_table(["Entity", "Type", "Start", "End"], entity_rows)
        relation_table = _html_table(["Subject", "Predicate", "Object"], relation_rows)
        bio_table = _html_table(["Token", "BIO Tag"], bio_rows)

        return f"""
<!doctype html>
<html lang=\"zh-CN\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>信息抽取静态 HTML 报告</title>
    <style>
        :root {{
            --ink: #16263a;
            --sub: #4f6277;
            --line: #d8e0ea;
            --card: #ffffff;
        }}
        body {{
            margin: 0;
            color: var(--ink);
            font-family: 'Noto Sans SC', 'Segoe UI', sans-serif;
            background:
                radial-gradient(90% 45% at 0% 0%, #edf4ff 0%, transparent 55%),
                radial-gradient(90% 55% at 100% 10%, #eaf9ef 0%, transparent 60%),
                linear-gradient(180deg, #f9fcff 0%, #f5fff8 100%);
        }}
        .wrap {{ max-width: 1180px; margin: 18px auto 28px; padding: 0 14px; }}
        .hero {{
            border: 1px solid rgba(20,33,61,0.14);
            border-radius: 16px;
            padding: 14px;
            background: linear-gradient(120deg, #ffffff 0%, #e7f1ff 100%);
            box-shadow: 0 10px 30px rgba(20,33,61,0.08);
        }}
        .hero h1 {{ margin: 0; font-size: 28px; }}
        .hero p {{ margin: 8px 0 0; color: var(--sub); }}
        .grid {{ display: grid; gap: 14px; margin-top: 14px; grid-template-columns: 1fr; }}
        .card {{
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--card);
            padding: 14px;
            box-shadow: 0 6px 18px rgba(20,33,61,0.08);
        }}
        .card h2 {{ margin-top: 0; }}
        .quote {{
            white-space: pre-wrap;
            line-height: 1.7;
            background: #f7fbff;
            border: 1px solid #dce8f5;
            border-left: 4px solid #7fb0ef;
            border-radius: 10px;
            padding: 10px;
        }}
        .tbl {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14px; }}
        .tbl th, .tbl td {{ border: 1px solid #dfe7f1; padding: 8px; text-align: left; }}
        .tbl th {{ background: #f2f7ff; }}
        .empty {{ color: #5e7287; font-style: italic; }}
        #graph {{ height: 430px; border: 1px solid #dce7f3; border-radius: 10px; }}
        .meta {{ color: var(--sub); margin: 4px 0 0; }}
        @media (min-width: 960px) {{
            .grid {{ grid-template-columns: 1fr 1fr; }}
            .span2 {{ grid-column: span 2; }}
        }}
    </style>
</head>
<body>
    <div class=\"wrap\">
        <section class=\"hero\">
            <h1>信息抽取静态 HTML 报告</h1>
            <p>导出时间: {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
            <p>包含模块: NER/BIO、关系抽取、知识图谱可视化</p>
        </section>

        <div class=\"grid\">
            <section class=\"card span2\">
                <h2>输入文本</h2>
                <div class=\"quote\">{html.escape(text)}</div>
            </section>

            <section class=\"card\">
                <h2>实体识别结果</h2>
                <p class=\"meta\">实体数量: {len(entity_rows)}</p>
                {entity_table}
            </section>

            <section class=\"card\">
                <h2>关系抽取结果</h2>
                <p class=\"meta\">关系数量: {len(relation_rows)}</p>
                {relation_table}
            </section>

            <section class=\"card span2\">
                <h2>BIO 标注序列</h2>
                <p class=\"meta\">{html.escape(bio_note)}</p>
                {bio_table}
            </section>

            <section class=\"card span2\">
                <h2>知识图谱</h2>
                <div id=\"graph\"></div>
            </section>
        </div>
    </div>

    <script src=\"https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js\"></script>
    <script>
        const graphData = {graph_json};
        const chart = echarts.init(document.getElementById('graph'));
        const categoryColors = {{ PER: '#ff9d66', ORG: '#5ec878', LOC: '#5f9df6', UNKNOWN: '#8ea2b9' }};
        chart.setOption({{
            tooltip: {{
                trigger: 'item',
                formatter: function(params) {{
                    if (params.dataType === 'edge') {{
                        return params.data.source + ' - ' + params.data.relation + ' - ' + params.data.target;
                    }}
                    return params.data.name + ' (' + params.data.category + ')';
                }}
            }},
            series: [{{
                type: 'graph',
                layout: 'force',
                roam: true,
                draggable: true,
                edgeSymbol: ['none', 'arrow'],
                edgeSymbolSize: [4, 10],
                data: graphData.nodes.map(function(n) {{
                    return Object.assign({{}}, n, {{
                        itemStyle: {{ color: categoryColors[n.category] || '#8ea2b9' }},
                        label: {{ show: true, color: '#1d2f46', fontWeight: 700 }}
                    }});
                }}),
                links: graphData.links.map(function(e) {{
                    return Object.assign({{}}, e, {{
                        lineStyle: {{ color: '#6a7c96', width: 1.8, curveness: 0.14 }},
                        label: {{ show: true, formatter: e.relation, color: '#2f4a6c' }}
                    }});
                }}),
                force: {{ repulsion: 420, edgeLength: [110, 220], gravity: 0.04 }}
            }}]
        }});
        window.addEventListener('resize', function() {{ chart.resize(); }});
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/extract", methods=["POST"])
def api_extract():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "请输入文本后再抽取。"}), 400

    result = extract_pipeline(text)
    return jsonify(result)


@app.route("/download/templates", methods=["GET"])
def download_templates():
    template_files = [
        "template_ner_bio.html",
        "template_relation_panel.html",
        "template_kg_visualization.html",
    ]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_name in template_files:
            file_path = os.path.join(TEMPLATE_HTML_DIR, file_name)
            if os.path.exists(file_path):
                zf.write(file_path, arcname=file_name)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name="kg_web_templates.zip",
        mimetype="application/zip",
    )


@app.route("/download/report", methods=["POST"])
def download_report():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "请输入文本后再下载静态 HTML 报告。"}), 400

    result = extract_pipeline(text)
    report_html = build_static_html_report(text, result)
    file_name = f"kg_extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    return send_file(
        io.BytesIO(report_html.encode("utf-8")),
        as_attachment=True,
        download_name=file_name,
        mimetype="text/html",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
