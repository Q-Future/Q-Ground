from huggingface_hub import snapshot_download

local_path = snapshot_download(repo_id="chaofengc/QGround-100K", repo_type="dataset", local_dir="./dataset", local_dir_use_symlinks=False)



