import requests
import os

def save_images(session, image_urls, label_template, media_dir):
    os.makedirs(media_dir, exist_ok=True)
    paths = []

    for i, url in enumerate(image_urls, 1):
        try:
            response = session.get(url, timeout = 20)
            response.raise_for_status()

            filename = label_template.replace("NUM", str(i))
            filepath = os.path.join(media_dir, filename)

            with open(filepath, 'wb') as f:
                f.write(response.content)
            paths.append(filepath)

        except Exception as e:
            print(f"[!] Could not download image: {url} - {e}")
            paths.append("Download failed")
    
    return paths
