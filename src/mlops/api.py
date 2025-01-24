from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from mlops.model import resnetSimple
import torchvision.transforms as transforms
import torch.nn.functional as F
from prometheus_client import Counter, Histogram, Summary, make_asgi_app
from fastapi import HTTPException

# Global variables
model = None
device = None
class_names = ["Kirmizi", "Siirt"]

# Define Prometheus metrics
error_counter = Counter("prediction_error", "Number of prediction errors")
request_counter = Counter("prediction_requests", "Number of prediction requests")
request_latency = Histogram("prediction_latency_seconds", "Prediction latency in seconds")
review_summary = Summary("review_length_summary", "Review length summary")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, device
    print("Loading model")
    model = resnetSimple()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model.load_state_dict(torch.load(model_path))
    model.eval()

    yield

    print("Cleaning up")
    del model, device


app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


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
    request_counter.inc()
    with request_latency.time():
        try:
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
        except Exception as e:
            error_counter.inc()
            raise HTTPException(status_code=500, detail=str(e)) from e
