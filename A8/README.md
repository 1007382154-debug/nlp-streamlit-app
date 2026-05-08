# 机器翻译机制对比与评测平台

项目名：`A8`

这是一个 Streamlit 教学应用，用于对比神经机器翻译、基于规则的逐词直译，并用 BLEU Score 观察机器翻译自动评测的特点。

## 功能模块

1. `模块 1：NMT Engine`
   - 输入英文句子。
   - 使用 Hugging Face `transformers.pipeline` 加载 `Helsinki-NLP/opus-mt-en-zh`。
   - 输出中文神经机器翻译结果。
   - 首次加载模型时显示 Streamlit Spinner。

2. `模块 2：规则直译 vs NMT`
   - 使用内置英汉词典进行空格/标点级的逐词直译。
   - 与 NMT 译文并排展示。
   - 适合观察习语、定语从句、一词多义和语序差异。

3. `模块 3：BLEU 自动评测`
   - 输入英文原文、中文参考译文和候选译文。
   - 使用 `nltk.translate.bleu_score` 计算 BLEU。
   - 展示分数并解释分数含义。

## Conda 环境运行

按项目约定，环境名使用 `%ENV_NAME%` 占位符。注意：`%ENV_NAME%` 只是占位符，实际运行时需要替换成真实 conda 环境名。

如果你已经看到终端前缀类似 `(%ENV_NAME%) PS ...`，说明环境已经激活，此时直接运行：

```powershell
python -m pip install -U -r requirements.txt
python download_model.py
streamlit run app.py
```

CMD：

```bat
conda activate 你的环境名
python -m pip install -U -r requirements.txt
python download_model.py
streamlit run app.py
```

PowerShell：

```powershell
$env:ENV_NAME="你的真实环境名"
conda run -n $env:ENV_NAME python -m pip install -U -r requirements.txt
conda run -n $env:ENV_NAME python download_model.py
conda run -n $env:ENV_NAME streamlit run app.py
```

## 下载 Hugging Face 模型

应用会真实调用：

```python
transformers.pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
```

如果下载很慢，可以在启动前设置镜像端点：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python download_model.py
streamlit run app.py
```

如果出现 `Unable to load vocabulary from file`，通常是不完整下载缓存导致的。删除 Hugging Face 缓存中 `models--Helsinki-NLP--opus-mt-en-zh` 对应目录后，再运行 `python download_model.py` 重新下载。

也可以使用本地模型目录：

```powershell
$env:MT_MODEL_PATH="D:\path\to\opus-mt-en-zh"
conda run -n $env:ENV_NAME streamlit run app.py
```
