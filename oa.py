from fastapi import FastAPI, File, UploadFile
import torch
from cnn import ImageNet
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

# Load the CNN model
model = ImageNet()
model.load_state_dict(torch.load("knee_oa_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28, 28)),  # Resize to match model input
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess the image
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Get prediction
        with torch.no_grad():
            output = model(image)
            probability = output.item()  # Get the probability score
            prediction = "Progressor" if probability > 0.5 else "Non-Progressor"

        # AI recommendation
        recommendation = (
            "⚕️ Suggested next steps:\n"
            "- Assess patient pain levels\n"
            "- Review BMI and medical history\n"
            "- Consider further imaging for confirmation"
        )

        return {
            "prediction": prediction,
            "confidence": probability,
            "recommendation": recommendation
        }
    except Exception as e:
        return {"error": str(e)}

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("oa:app", host="127.0.0.1", port=8000, reload=True)
