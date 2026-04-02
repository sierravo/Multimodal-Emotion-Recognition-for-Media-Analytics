import torch
from torchvision import transforms
from SongFan import EMO6Classifier
from AffectNet import EmotionClassifier
from transforms import ANRescale, ANToTensor, SFNormalize, SFRescale, SFToTensor


def load_affectnet_model(checkpoint_path):
    """
    Load AffectNet AlexNet-based model and its preprocessing pipeline.

    Args:
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        model (EmotionClassifier): Loaded AffectNet model.
        transform (torchvision.transforms.Compose): Preprocessing pipeline.
    """
    model = EmotionClassifier()
    model.load(checkpoint_path)
    model.eval()

    transform_pipeline = transforms.Compose([
        ANRescale(),
        ANToTensor(),
    ])

    return model, transform_pipeline


def load_songfan_model(checkpoint_path):
    """
    Load SongFan CNN model and its preprocessing pipeline.

    Args:
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        model (EMO6Classifier): Loaded SongFan model.
        transform (torchvision.transforms.Compose): Preprocessing pipeline.
    """
    model = EMO6Classifier(pretrained=True, path_to_checkpoint=checkpoint_path)
    model.eval()

    transform_pipeline = transforms.Compose([
        SFNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        SFRescale(),
        SFToTensor(),
    ])

    return model, transform_pipeline
