import requests
from bs4 import BeautifulSoup
import os

def collect_images(query, num_images, data_path):
    url = f"https://www.google.com/search?hl=en&tbm=isch&q={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img', limit=num_images)
    os.makedirs(data_path, exist_ok=True)
    for i, img in enumerate(img_tags):
        img_url = img['src']
        if not img_url.startswith('http'):
            continue
        img_data = requests.get(img_url).content
        with open(os.path.join(data_path, f'{query}_{i}.jpg'), 'wb') as handler:
            handler.write(img_data)
if __name__ == '__main__':
  #for ex
  collect_images('chicken meme', 100, "/content/drive/MyDrive/animals_v2/animals/train/chicken")
