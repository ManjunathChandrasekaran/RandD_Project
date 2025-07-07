import torch
from PIL import Image
import open_clip
import os

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use the correct model name and load from Hugging Face Hub
model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

try:
    # Load model, preprocessor, and tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    model.eval()  # Set model to evaluation mode
except Exception as e:
    print(f"Error loading model, preprocessor, or tokenizer: {e}")
    exit(1)

# image path
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "XRay", "images_2.jpeg")

if not os.path.exists(image_path):
    print(f"Image file not found: {image_path}")
    exit(1)

try:
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)  # Preprocess and add batch dimension
except Exception as e:
    print(f"Error loading or processing image: {e}")
    exit(1)

# Define diagnostic labels, including "Fracture"
diagnostic_labels = [
    "Normal chest X-ray",
    "Chest X-ray showing pneumonia",
    "Chest X-ray showing cardiomegaly",
    "Chest X-ray showing pleural effusion",
    "Chest X-ray showing atelectasis",
    "X-ray showing a radius fracture",
    "X-ray showing an ulna fracture",
    "X-ray showing a wrist fracture"
]

# Validate labels
if not diagnostic_labels:
    print("Error: Diagnostic labels list is empty")
    exit(1)

# Tokenize text labels with specific prompts
try:
    text_inputs = tokenizer(diagnostic_labels).to(device)
except Exception as e:
    print(f"Error tokenizing text: {e}")
    exit(1)

# Run through the model
try:
    with torch.no_grad():
        image_features, text_features, logit_scale = model(image, text_inputs)
        logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
        print(f"Logits shape: {logits.shape}")
except Exception as e:
    print(f"Error during model inference: {e}")
    exit(1)

# Get prediction
try:
    predicted_idx = logits.argmax().item()
    predicted_label = diagnostic_labels[predicted_idx]
    confidence = logits[0, predicted_idx].item() * 100  # Convert to percentage
except Exception as e:
    print(f"Error processing prediction: {e}")
    exit(1)

# Output results
print(f"🩻 Predicted Diagnosis: {predicted_label} (Confidence: {confidence:.2f}%)")
# Print all probabilities
print("\nAll probabilities:")
for label, prob in zip(diagnostic_labels, logits[0]):
    print(f"{label}: {prob.item()*100:.2f}%")