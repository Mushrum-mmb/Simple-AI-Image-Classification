import requests
from bs4 import BeautifulSoup
import os
import time

def collect_images(query, num_images, data_path):
    url_template = "https://www.google.com/search?hl=en&tbm=isch&q={query}&start={start}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Create directory for storing images
    os.makedirs(data_path, exist_ok=True)

    images_collected = 0
    start = 0

    while images_collected < num_images:
        url = url_template.format(query=query.replace(' ', '+'), start=start)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        img_tags = soup.find_all('img')

        for img in img_tags:
            img_url = img['src']
            if not img_url.startswith('http'):
                continue
            try:
                img_data = requests.get(img_url).content
                with open(os.path.join(data_path, f'{query}_{images_collected}.jpg'), 'wb') as handler:
                    handler.write(img_data)
                images_collected += 1
                print(f'Downloaded {images_collected}/{num_images} images.')
                
                if images_collected >= num_images:
                    break
            except Exception as e:
                print(f"Could not download {img_url}: {e}")

        start += 20  # Increment the start index for the next set of results
        time.sleep(1)  # Respectful delay to avoid overwhelming the server

if __name__ == '__main__':
    collect_images('dog meme', 400, "/content")
