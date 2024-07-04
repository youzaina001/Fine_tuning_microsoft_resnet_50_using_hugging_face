from transformers import AutoModelForImageClassification
from torchvision import transforms
import torch
import numpy as np
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the pre-trained model from Hugging Face
model_name = "youzaina001/cifar10-resnet50"
model = AutoModelForImageClassification.from_pretrained(model_name).to(device)

# Manually define preprocessing for Gradio
def manual_preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)


# Lets build a Gradio interface for the model
def classify_image(image):
    
    # Preprocess the image
    image = image.convert("RGB")
    inputs = manual_preprocess(image)
    inputs = {"pixel_values": inputs}

    # Predict the class of the image
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    return predicted_class

# Lets create a Gradio interface
gface = gr.Interface(
    fn=classify_image,
    inputs=gr.components.Image(type="pil", label="Input Image"),
    outputs=gr.components.Label(num_top_classes=3),
    title="ResNet-50-CIFAR-10 Image Classifier",
    description="A pre-trained ResNet-50 model fine-tuned on the CIFAR-10 dataset.",
)

# Creating a shareable link to the Gradio interface
iface = gr.Interface(classify_image, "image", "label").launch(share=True)

if __name__ == "__main__":
    iface.launch()