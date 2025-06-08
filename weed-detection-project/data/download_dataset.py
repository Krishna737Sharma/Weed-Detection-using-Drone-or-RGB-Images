import os
import requests
import zipfile

def download_cwfid_dataset():
    os.makedirs('data', exist_ok=True)
    url = "https://github.com/cwfid/dataset/archive/refs/tags/v1.0.zip"
    output = "data/cwfid.zip"
    
    if not os.path.exists(output):
        print("Downloading CWFID dataset...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('data/')
        print("Dataset downloaded and extracted")
    else:
        print("Dataset already exists")

if __name__ == "__main__":
    download_cwfid_dataset()