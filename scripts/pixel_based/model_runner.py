import os
import torch
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from skimage import transform

from transforms import ANRescale, ANToTensor, SFNormalize, SFRescale, SFToTensor
from AffectNet import EmotionClassifier
from SongFan import EMO6Classifier

class ModelRunner:
    """
    Handles loading and inference for BReG-NeXt (TensorFlow), AffectNet (PyTorch),
    and SongFan (PyTorch) emotion classification models.
    """

    def __init__(self, bregnext_ckpt, affectnet_ckpt, songfan_ckpt, device='cpu', example_dir='examples'):
        """
        Load all models and set up transforms.

        Args:
            bregnext_ckpt (str): Path to BReG-NeXt checkpoint directory.
            affectnet_ckpt (str): Path to AffectNet checkpoint file.
            songfan_ckpt (str): Path to SongFan checkpoint file.
            device (str): 'cpu' or 'cuda'.
            example_dir (str): Directory for saving example outputs.
        """
        self.device = device
        self.example_dir = example_dir
        os.makedirs(self.example_dir, exist_ok=True)

        # Load BReG-NeXt TensorFlow model
        self.sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.import_meta_graph(os.path.join(bregnext_ckpt, 'checkpoints-4300.meta'))
        saver.restore(self.sess, tf.compat.v1.train.latest_checkpoint(bregnext_ckpt))
        graph = tf.compat.v1.get_default_graph()
        self.bregnext_input = graph.get_operation_by_name('image_batch_placeholder').outputs[0]
        self.bregnext_predictions = graph.get_operation_by_name('FullyConnected/BiasAdd').outputs[0]

        # Load AffectNet PyTorch model
        self.affectnet = EmotionClassifier()
        self.affectnet.load(affectnet_ckpt)
        self.affectnet.eval()
        self.affectnet_transform = torch.nn.Sequential(
            ANRescale(),
            ANToTensor()
        )

        # Load SongFan PyTorch model
        self.songfan = EMO6Classifier(pretrained=True, path_to_checkpoint=songfan_ckpt)
        self.songfan.eval()
        self.songfan_transform = torch.nn.Sequential(
            SFNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            SFRescale(),
            SFToTensor()
        )

    def run_bregnext(self, face_imgs):
        """
        Run BReG-NeXt inference on cropped face images.

        Args:
            face_imgs (np.ndarray): Batch of cropped face images.

        Returns:
            pd.DataFrame: Prediction probabilities for emotion classes.
        """
        feed_dict = {self.bregnext_input: face_imgs}
        pred_prob = self.sess.run(self.bregnext_predictions, feed_dict)
        labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
        return pd.DataFrame(pred_prob, columns=labels)

    def run_affectnet(self, face_imgs):
        """
        Run AffectNet inference on cropped face images.

        Args:
            face_imgs (np.ndarray): Batch of cropped face images.

        Returns:
            pd.DataFrame: Prediction scores for emotion classes.
        """
        transformed_imgs = torch.stack([self.affectnet_transform(f) for f in face_imgs])
        pred_label, pred_val = self.affectnet(transformed_imgs)
        labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain', 'No-Face']
        return pd.DataFrame(pred_label.detach().numpy(), columns=labels), pred_val

    def run_songfan(self, img):
        """
        Run SongFan inference on a full image.

        Args:
            img (np.ndarray): Full RGB image.

        Returns:
            pd.DataFrame: Prediction scores for emotions.
            np.ndarray: Attention map.
            np.ndarray: Salience map.
        """
        transformed_img = self.songfan_transform(img)
        pred, attention_map, salience_map = self.songfan(transformed_img[0][None], transformed_img[1][None])
        labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
        results = pd.DataFrame(pred[:, 2:].detach().numpy(), columns=labels)

        amap = transform.resize(attention_map.detach().numpy()[0, 0], (600, 800))
        smap = transform.resize(s
