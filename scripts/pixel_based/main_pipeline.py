"""
main.py

Main script to orchestrate image-based emotion recognition using three different models:
- BReG-NeXt (TensorFlow)
- AffectNet AlexNet (PyTorch)
- SongFan EMO6Classifier (PyTorch)

This pipeline loads example images, detects faces, and runs inference across all models.
"""

from data_loader import DataLoader
from affectnet_model import load_affectnet_model
from songfan_model import load_songfan_model
from bregnext_model_loader import load_bregnext_model

import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage import transform

# Setup models
sess, new_input, predictions = load_bregnext_model()
affectnet_model, affectnet_transform = load_affectnet_model()
songfan_model, songfan_transform = load_songfan_model()

# Load example images
example_images = [
    '../../multimedia_sentiment/media/NYT_11_2020/article_51_004.jpg',
    '../../multimedia_sentiment/media/NYT_11_2020/article_166_006.jpg',
    '../../multimedia_sentiment/media/NYT_11_2020/article_74_002.jpg',
    '../../multimedia_sentiment/media/NYT_11_2020/article_185_001.jpg',
    '../../multimedia_sentiment/media/NYT_11_2020/article_60_013.jpg'
]

dl = DataLoader()

for i, img_path in enumerate(example_images):
    new_img = dl.get_new_image(i, img_path=img_path)
    img_faces = dl.find_faces(i, new_img)

    # BReG-NeXt predictions
    feed_dict = {new_input: img_faces}
    pred_prob = sess.run(predictions, feed_dict)
    labels_breg = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
    results_breg = pd.DataFrame(pred_prob, columns=labels_breg)
    print("\nBReG-NeXt:")
    print(results_breg, results_breg.idxmax(axis=1))

    # AffectNet predictions
    img_faces_tensor = torch.stack([affectnet_transform(f) for f in img_faces])
    pred_label, pred_val = affectnet_model(img_faces_tensor)
    labels_affect = labels_breg + ['None', 'Uncertain', 'No-Face']
    results_affect = pd.DataFrame(pred_label.detach().numpy(), columns=labels_affect)
    print("\nAffectNet:")
    print(results_affect, results_affect.idxmax(axis=1), pred_val)

    # SongFan predictions
    songfan_input = songfan_transform(new_img)
    pred, attention_map, salience_map = songfan_model(songfan_input[0][None], songfan_input[1][None])
    labels_songfan = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
    results_songfan = pd.DataFrame(pred[:, 2:].detach().numpy(), columns=labels_songfan)
    print("\nSongFan:")
    print(results_songfan, results_songfan.idxmax(axis=1), pred[:, :2])

    amap = transform.resize(attention_map.detach().numpy()[0, 0], (600, 800))
    plt.imsave(f'examples/fig{i}_attention.png', amap)

    smap = transform.resize(salience_map.detach().numpy()[0, 0], (600, 800))
    plt.imsave(f'examples/fig{i}_salience.png', smap)

    plt.close('all')
