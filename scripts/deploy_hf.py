"""One-shot deploy to a Hugging Face Space (Docker SDK).

Uploads the serving files (app/, autoanalyst/, samples/, Dockerfile,
requirements-serve.txt) plus README_HF.md (as the Space's README.md) via the
huggingface_hub API — no git-LFS, which is flaky on Windows.

Usage (Anaconda Prompt, Nenv active):
    python scripts/deploy_hf.py --token hf_xxxxxxxx
    # or set HF_TOKEN in the environment and omit --token

IMPORTANT: after the first deploy, add your Groq key as a Space *secret* named
GROQ_API_KEY (Space → Settings → Variables and secrets). Without it the demo
shows "offline".
"""
import argparse
import os
import sys
from pathlib import Path

# this user's network stalls on HF's Xet CDN — force the classic upload path
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Abdullahkousa2/autoanalyst",
                    help="HF Space id (user/space-name)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF write token (or set HF_TOKEN env var)")
    args = ap.parse_args()
    if not args.token:
        sys.exit("provide a write token via --token or the HF_TOKEN env var")

    from huggingface_hub import HfApi
    api = HfApi()

    # create the Docker Space if it doesn't exist yet (safe to re-run)
    api.create_repo(args.repo, repo_type="space", space_sdk="docker",
                    token=args.token, exist_ok=True)
    print(f"-> Space ready: {args.repo}")

    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=args.repo,
        repo_type="space",
        token=args.token,
        allow_patterns=["app/**", "autoanalyst/**", "samples/**",
                        "Dockerfile", "requirements-serve.txt"],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
    )
    # README with HF frontmatter -> the Space's README.md
    api.upload_file(
        path_or_fileobj=str(ROOT / "README_HF.md"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="space",
        token=args.token,
    )
    print(f"-> deployed to https://huggingface.co/spaces/{args.repo}")
    print("   reminder: set the GROQ_API_KEY secret in the Space settings.")


if __name__ == "__main__":
    main()
