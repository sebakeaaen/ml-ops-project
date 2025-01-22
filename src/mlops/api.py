from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from mlops.model import resnetSimple
import torchvision.transforms as transforms
import torch.nn.functional as F

# Global variables
model = None
device = None
class_names = ["Kirmizi", "Siirt"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, device
    print("Loading model")
    model = resnetSimple()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    yield

    print("Cleaning up")
    del model, device


app = FastAPI(lifespan=lifespan)


# Define the image preprocessing pipeline
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize the image to the size expected by ResNet18
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ]
)


@app.post("/classify/")
async def classify(data: UploadFile = File(...)):
    """Classify an image using the model."""
    # Load and preprocess the image
    i_image = Image.open(data.file).convert("RGB")  # Ensure the image is in RGB mode
    i_image = transform(i_image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make predictions
    with torch.no_grad():
        outputs = model(i_image)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()

    # Get the predicted class name
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[0, predicted_class_idx].item()

    return {"predicted_class": predicted_class, "confidence": confidence}
