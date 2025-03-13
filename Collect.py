"""
In this script, we will download and prepare our private datasets, which depend on various topics and different cases for each class.
For example, my target classes are dog, cat, capybara, mouse, parrot, and pufferfish.
However, each class has many relevant variations, such as 'dog in real life,' 'cute dog,' 'dog meme,' and 'puppy.'
To facilitate this, we will create an automation process for collecting images from Google Images based on specified search queries.

First imports:
- requests: For making HTTP requests to fetch web pages.
- BeautifulSoup: From the bs4 library, used to parse HTML and extract data.
- os: For interacting with the operating system (e.g., creating directories).
- time: To introduce delays between requests.
"""

import requests
from bs4 import BeautifulSoup
import os
import time
# Defines a function collect_images that takes three parameters
def collect_images(query, num_images, dataset_path):
    # Make a URL template
    url_template = "https://www.google.com/search?tbm=isch&q={query}&start={start}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    # Initialize the counter
    img_collected = 0
    start = 0
    # Create a loop until all images have been collected
    while img_collected < num_images:
        # Construct the full URL and replace all ' ' with '+'
        url = url_template.format(query=query.replace(' ', '+'), start=start)
        # Make request, parse HTML and find image tags
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all('img')
        # Loop through img_tags and checks if img_url starts with http
        for img in img_tags:
            img_url = img.get('src')
            if not img_url or not img_url.startswith('http'):
                continue  # skip
            # Try-Except for downloading
            try:
                # Downloads the image data
                img_data = requests.get(img_url).content
                # Saving the image, open file in write-binary mode
                with open(os.path.join(dataset_path, f"{query}_{img_collected}.jpg"), 'wb') as img_path:
                    img_path.write(img_data)
                    img_collected += 1
                    print(f"Downloaded {img_collected}/{num_images} {query} images in {dataset_path}")
                    # Break if collecting enough
                    if img_collected == num_images:
                        break
            except Exception as e:
                # Print out the error
                print(f"Could not download {img_url} due to {e}")
        # Increment the starting index to fetch the next set of results
        start += 20
        # Pause the execution for 1 second to avoid overwhelming the server
        time.sleep(1)

if __name__ == '__main__':
    """
    In this section, I will start downloading images into the test and train folders.
    I prefer to set it to 300 images per subcategory and take 20% of the total images for validation.
    For example, I will download 1,800 dog images, which will include 300 images each of the following subcategories:
    300 'dog in real life,' 300 'cute dogs,' 300 'puppy,' 300 'dog funny and memes,' 300 'dog cartoons,' and 300 'dog sketches.'
    The validation folder will contain a random selection of 20% of the total images, which equals 360 images.
    """
    
    # Initialize the number of train and validation images, main categories, and dataset path
    num_trains = 300
    num_vals = 60
    categories = ['dog', 'cat', 'capybara', 'hamster', 'parrot', 'pufferfish']
    subcategories = ['real life', 'cute', 'baby', 'funny and meme', 'cartoon', 'sketch']
    train_path = "/content/drive/MyDrive/AnimalDataset/train"  # My path looks like this because I use Google Colab
    val_path = "/content/drive/MyDrive/AnimalDataset/val"

    # Create directories for each category
    for cate in categories:
        os.makedirs(os.path.join(train_path, cate), exist_ok=True)
        os.makedirs(os.path.join(val_path, cate), exist_ok=True)
        
    # Create a loop to download images in the train folder
    for cate in categories:
        for subcate in subcategories:
            collect_images(f"{subcate} {cate}", num_trains, os.path.join(train_path, cate))
    
    # Validation
    for cate in categories:
        for subcate in subcategories:
            collect_images(f"{subcate} {cate}", num_vals, os.path.join(val_path, cate))

    print("Downloading successfull! ^V^")
