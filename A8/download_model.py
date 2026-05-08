import os
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
os.environ.setdefault("HF_HOME", str(SCRIPT_DIR / ".hf_cache"))

from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = os.getenv("MT_MODEL_PATH", "Helsinki-NLP/opus-mt-en-zh")
LOCAL_MODEL_DIR = Path("models") / MODEL_NAME.replace("/", "--")
REQUIRED_FILES = {
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "source.spm",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
}


def clean_local_model_dir():
    if LOCAL_MODEL_DIR.exists():
        print(f"Removing corrupted/incomplete local model dir: {LOCAL_MODEL_DIR}")
        shutil.rmtree(LOCAL_MODEL_DIR)


def missing_required_files(model_dir: Path) -> list[str]:
    return sorted(
        file_name
        for file_name in REQUIRED_FILES
        if not (model_dir / file_name).is_file() or (model_dir / file_name).stat().st_size == 0
    )


def download_snapshot(force_download: bool = False):
    print(f"Downloading Hugging Face snapshot: {MODEL_NAME}")
    print(f"Using Hugging Face endpoint: {HF_ENDPOINT}")
    print(f"Local model dir: {LOCAL_MODEL_DIR}")
    return snapshot_download(
        repo_id=MODEL_NAME,
        force_download=force_download,
        local_dir=LOCAL_MODEL_DIR,
        local_files_only=False,
        allow_patterns=sorted(REQUIRED_FILES),
    )


def main():
    try:
        local_path = download_snapshot()
        print(f"Snapshot ready: {local_path}")
        missing = missing_required_files(Path(local_path))
        if missing:
            raise OSError(f"Missing required model files: {', '.join(missing)}")

        load_path = str(LOCAL_MODEL_DIR)
        print(f"Loading tokenizer: {load_path}")
        AutoTokenizer.from_pretrained(load_path)

        print(f"Loading model: {load_path}")
        AutoModelForSeq2SeqLM.from_pretrained(load_path)
    except OSError as exc:
        message = str(exc)
        if "Unable to load vocabulary" not in message and "source.spm" not in message:
            raise

        clean_local_model_dir()
        local_path = download_snapshot(force_download=True)
        print(f"Snapshot ready after retry: {local_path}")
        missing = missing_required_files(Path(local_path))
        if missing:
            raise OSError(f"Missing required model files after retry: {', '.join(missing)}")

        load_path = str(LOCAL_MODEL_DIR)
        print(f"Loading tokenizer after retry: {load_path}")
        AutoTokenizer.from_pretrained(load_path)

        print(f"Loading model after retry: {load_path}")
        AutoModelForSeq2SeqLM.from_pretrained(load_path)

    print(f"Model is ready in the local project dir: {SCRIPT_DIR / LOCAL_MODEL_DIR}")


if __name__ == "__main__":
    main()
