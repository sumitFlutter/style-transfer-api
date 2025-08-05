import os
import io
from flask import Flask, request, send_file
import torch
from PIL import Image
from torchvision import transforms
from transformer_net import TransformerNet

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    return transform(image).unsqueeze(0).to(device)

def postprocess(tensor):
    image = tensor.cpu().clone().squeeze(0)
    image = image.clamp(0, 255).permute(1, 2, 0).numpy().astype('uint8')
    return Image.fromarray(image)

@app.route("/")
def home():
    return "🎨 Style Transfer API is running."

@app.route("/stylize", methods=["POST"])
def stylize():
    if "image" not in request.files:
        return "No image uploaded", 400

    style_name = request.args.get("style", "mosaic")
    model_path = f"models/{style_name}.pth"

    if not os.path.exists(model_path):
        return f"Model '{style_name}.pth' not found.", 404

    # Load model on-demand
    model = TransformerNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Preprocess image
    input_tensor = preprocess(request.files["image"].read())

    # Inference with no_grad and memory safety
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Clean up model from memory
    del model
    torch.cuda.empty_cache()

    # Postprocess and return image
    output_image = postprocess(output_tensor)
    buf = io.BytesIO()
    output_image.save(buf, format="JPEG")
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
