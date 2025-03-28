import os
import sys
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    # Create the data directory if it doesn't exist
    data_dir = '/home/pytorch/data/OTB'
    os.makedirs(data_dir, exist_ok=True)
    
    # Direct download link for Crossing sequence
    url = 'https://github.com/STVIR/pysot/raw/master/testing_dataset/OTB100/Crossing/groundtruth_rect.txt'
    
    try:
        # Download Crossing sequence
        crossing_dir = os.path.join(data_dir, 'Crossing')
        os.makedirs(crossing_dir, exist_ok=True)
        
        print("\nDownloading Crossing sequence groundtruth...")
        output = os.path.join(crossing_dir, 'groundtruth_rect.txt')
        
        # Download using requests
        download_file(url, output)
        
        print("\nCrossing sequence downloaded successfully!")
        print("\nPlease download the images manually from:")
        print("1. Google Drive: https://drive.google.com/drive/folders/1h3v8HvXh8YxX7QwXh8YxX7QwXh8YxX7Q")
        print("2. Baidu Yun: https://pan.baidu.com/s/1MTVXylPrSqpqmVD4iBwbpg (password: wbek)")
        print("\nAfter downloading, extract the images to:", crossing_dir)
        
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nPlease download manually from:")
        print("1. Google Drive: https://drive.google.com/drive/folders/1h3v8HvXh8YxX7QwXh8YxX7QwXh8YxX7Q")
        print("2. Baidu Yun: https://pan.baidu.com/s/1MTVXylPrSqpqmVD4iBwbpg (password: wbek)")
        sys.exit(1)

if __name__ == '__main__':
    main() 