import argparse
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model files")
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="Model name to use",
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default="cd6881a82d62252f5a84593c61acf290f15d89e3",
        help="Model revision",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./saved_model/",
        help="Path to save the model files",
    )

    args = parser.parse_args()

    snapshot_download(
        repo_id=args.model_name,
        local_dir=args.model_path,
        revision=args.model_revision,
    )

    print(f"Model files downloaded to {args.model_path}")
