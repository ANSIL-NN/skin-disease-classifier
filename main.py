import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(title="Skin Disease Classifier")

# ── Class names (verified order) ──────────────────────────────
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

DISEASE_LABELS = {
    "akiec": "Actinic Keratoses",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}

# ── Device ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load model once at startup ────────────────────────────────
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 7)
    checkpoint = torch.load("model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

model = load_model()
print("Model loaded successfully!")

# ── Preprocessing (must match training exactly) ───────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── GET /health ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

# ── POST /predict ─────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    top_idx = probs.argmax().item()
    top_code = CLASS_NAMES[top_idx]
    confidence = round(probs[top_idx].item() * 100, 1)

    top3_indices = probs.topk(3).indices.tolist()
    top3 = [
        {
            "code": CLASS_NAMES[i],
            "disease": DISEASE_LABELS[CLASS_NAMES[i]],
            "confidence": round(probs[i].item() * 100, 1)
        }
        for i in top3_indices
    ]

    return {
        "disease": DISEASE_LABELS[top_code],
        "code": top_code,
        "confidence": confidence,
        "top3": top3,
    }