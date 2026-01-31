
import gradio as gr
from transformers import pipeline


classifier = pipeline("image-classification", model="gilladog/vit-base-eurosat")

def predict(img):
    predictions = classifier(img)
    return {p["label"]: p["score"] for p in predictions}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="EuroSAT Land Cover Classifier",
    description="Upload a satellite image to identify the land cover type using a Vision Transformer (ViT)."
)

demo.launch()
