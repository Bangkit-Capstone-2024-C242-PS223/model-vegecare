import os
import zipfile

zip_path = "/content/drive/MyDrive/vegecare_dataset.zip"
extract_path = "/content/"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Extracted to:", extract_path)
