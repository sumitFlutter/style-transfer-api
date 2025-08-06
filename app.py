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
    try:
        # Open and convert to RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Convert to JPEG in-memory
        jpeg_buffer = io.BytesIO()
        image.save(jpeg_buffer, format="JPEG")
        jpeg_buffer.seek(0)
        image = Image.open(jpeg_buffer)

        # Apply transformation
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        return transform(image).unsqueeze(0).to(device)

    except Exception as e:
        raise ValueError(f"Image processing error: {e}")


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

    file = request.files["image"]

    # 1. Check MIME type
    if not file.mimetype.startswith("image/"):
        return "Only image files are allowed (e.g., jpg, png, webp).", 400

    # 2. Optional: check extension
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
    filename = file.filename.lower()
    if '.' not in filename or filename.rsplit('.', 1)[1] not in ALLOWED_EXTENSIONS:
        return "Unsupported file extension.", 400

    # 3. Real image content validation
    try:
        Image.open(file).verify()
        file.seek(0)  # Reset pointer
    except Exception:
        return "Uploaded file is not a valid image.", 400

    # Get style model
    style_name = request.args.get("style", "mosaic")
    model_path = f"models/{style_name}.pth"
    if not os.path.exists(model_path):
        return f"Model '{style_name}.pth' not found.", 404

    # Load model
    model = TransformerNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Preprocess
    try:
        input_tensor = preprocess(file.read())
    except ValueError as e:
        return str(e), 400

    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Clean up
    del model
    torch.cuda.empty_cache()

    # Postprocess and return
    output_image = postprocess(output_tensor)
    buf = io.BytesIO()
    output_image.save(buf, format="JPEG")
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
        
