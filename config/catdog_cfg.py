import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

class CatDogDataConfig():
        N_CLASSES = 2
        IMG_SIZE = 64
        ID2DLABEL = {0: 'Cat', 1: 'Dog'}
        LABEL2ID = {"Cat": 0, "Dog": 1}
        NORMALIZE_MEAN = [0.485, 0.456, 0.406]
        NORMALIZE_STD = [0.229, 0.224, 0.225]

class ModelConfig():
        ROOT_DIR = Path(__file__).parent.parent
        MODEL_NAME = "resnet18"
        MODEL_WEIGHT = ROOT_DIR / "models" / "weights" / "catdog_weights.pt"
        DEVICE = "cpu"
