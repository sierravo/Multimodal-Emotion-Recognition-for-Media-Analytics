import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import transform

from bregnext_model_loader import load_bregnext_model
from emotion_models_loader import load_affectnet_model, load_songfan_model


class ModelRunner:
    """
    Handles inference for BReG-NeXt, AffectNet, and SongFan models.
    """

    def __init__(
        self,
        bregnext_ckpt,
        affectnet_ckpt,
        songfan_ckpt,
        example_dir="examples",
    ):
        self.example_dir = example_dir
        os.makedirs(self.example_dir, exist_ok=True)

        # Load models
        self.sess, self.bregnext_input, self.bregnext_predictions = load_bregnext_model(
            bregnext_ckpt
        )
        self.affectnet_model, self.affectnet_transform = load_affectnet_model(
            affectnet_ckpt
        )
        self.songfan_model, self.songfan_transform = load_songfan_model(
            songfan_ckpt
        )

    def run_bregnext(self, face_imgs):
        labels = [
            "Neutral", "Happy", "Sad", "Surprise",
            "Fear", "Disgust", "Anger", "Contempt"
        ]
        feed_dict = {self.bregnext_input: face_imgs}
        pred_prob = self.sess.run(self.bregnext_predictions, feed_dict)
        return pd.DataFrame(pred_prob, columns=labels)

    def run_affectnet(self, face_imgs):
        labels = [
            "Neutral", "Happy", "Sad", "Surprise",
            "Fear", "Disgust", "Anger", "Contempt",
            "None", "Uncertain", "No-Face"
        ]

        transformed_imgs = [self.affectnet_transform(face_img) for face_img in face_imgs]
        pred_label, pred_val = self.affectnet_model(transformed_imgs)
        results = pd.DataFrame(pred_label.detach().numpy(), columns=labels)
        return results, pred_val

    def run_songfan(self, new_img, i):
        songfan_input = self.songfan_transform(new_img)
        pred, attention_map, salience_map = self.songfan_model(
            songfan_input[0][None],
            songfan_input[1][None]
        )

        labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
        results = pd.DataFrame(pred[:, 2:].detach().numpy(), columns=labels)

        amap = transform.resize(attention_map.detach().numpy()[0, 0], (600, 800))
        plt.imsave(os.path.join(self.example_dir, f"fig{i}_attention.png"), amap)

        smap = transform.resize(salience_map.detach().numpy()[0, 0], (600, 800))
        plt.imsave(os.path.join(self.example_dir, f"fig{i}_salience.png"), smap)

        plt.close("all")
        return results, pred[:, :2]

    def run_all(self, new_img, img_faces, i):
        results = {}

        results["bregnext"] = self.run_bregnext(img_faces)
        results["affectnet"], results["affectnet_meta"] = self.run_affectnet(img_faces)
        results["songfan"], results["songfan_meta"] = self.run_songfan(new_img, i)

        return results