from train import MultiModalModel  # Import the correct model
import torch
from torchvision import transforms
from PIL import Image
import io
import pandas as pd
import numpy as np

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load clinical features count (must match training setup)
num_clinical_features = 6  # Update based on your dataset (len(clinical_df.columns) - 2)

# Load the trained model
model = MultiModalModel(num_clinical_features=num_clinical_features)
checkpoint = torch.load("knee_arthritis_model.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...), age: float = 0, bmi: float = 0, sex: int = 0, knee_pain: int = 0, xray_grade: int = 0, prior_injury: int = 0):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess the image
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Convert clinical features into tensor
        clinical_data = torch.tensor([age, bmi, sex, knee_pain, xray_grade, prior_injury], dtype=torch.float32).unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            output = model(image, clinical_data)
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
