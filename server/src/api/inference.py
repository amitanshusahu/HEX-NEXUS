import torch
from torchvision import transforms
from PIL import Image

from src.model.model import SimpleCNN
from src.model.utils import dataset  # so we reuse dataset.classes

device = "cuda" if torch.cuda.is_available() else "cpu"

_model = None

def get_model():
    global _model
    if _model is None:
        print("🔹 Loading SimpleCNN model...")
        _model = SimpleCNN(num_classes=len(dataset.classes))
        _model.load_state_dict(torch.load("src/model/saved/simplecnn.pth", map_location=device))
        _model.to(device)
        _model.eval()
    return _model


# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def predict(image_path: str):
    model = get_model()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    return dataset.classes[predicted.item()]
