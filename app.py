from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
import os
import random

app = Flask(__name__)

# Ensure Static Folder Exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load Stable Diffusion Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

# Function to Generate Image
def generate_image(prompt):
    image = pipe(prompt).images[0]
    filename = f"static/generated_{random.randint(1, 100000)}.png"
    image.save(filename)
    return filename

# API Route
@app.route("/generate", methods=["GET"])
def generate():
    prompt = request.args.get("prompt", "A beautiful landscape")
    image_path = generate_image(prompt)
    return jsonify({"image_url": f"https://text-to-image-api.onrender.com/{image_path}"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
