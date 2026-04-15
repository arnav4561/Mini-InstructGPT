import zipfile
import os

def zip_folder(folder_name):
    with zipfile.ZipFile(f"{folder_name}.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_name):
            for file in files:
                file_path = os.path.join(root, file)
                # Use forward slashes for archive name
                arcname = file_path.replace("\\", "/")
                zf.write(file_path, arcname)
    print(f"Created {folder_name}.zip")

zip_folder("sft_model_v2")
zip_folder("reward_model_v2")
zip_folder("ppo_model")