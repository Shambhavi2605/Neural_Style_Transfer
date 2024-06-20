import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import streamlit as st

#Load VGG19 model
vgg19 = models.vgg19(pretrained=True).features
vgg19.eval()

# Defining image size
img_size =128

# Device configuration
device = torch.device("cpu")

# Image preprocessing
loader = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
 #Function for loading images
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device)

#Function for content loss
def content_loss(content, generated):
    return torch.mean((generated - content)**2)

# Function to define gram matrix
def gram_matrix(input):
    batch_size, c, h, w = input.size()  # Get dimensions directly from input tensor
    features = input.view(batch_size, c, h * w)  # Reshape input tensor
    gram = torch.bmm(features, features.transpose(1, 2))  # Compute Gram matrix
    gram = gram / (c * h * w)  # Normalize Gram matrix
    return gram

# Function for style loss
def style_loss(style, generated):
    S = gram_matrix(style)
    C = gram_matrix(generated)
    batch_size,c,h,w= style.size()  # Adjusted unpacking for Gram matrix shape
    loss = torch.mean((S - C)**2)
    return loss


# Hyperparameters
steps = 500
alpha = 1  # content weight
beta = 1000  # style weight

# Define a function for total loss
def total_loss(content_image_features, style_image_features, generated_image_features):
    content_loss_value = 0
    style_loss_value = 0
    for content_feature, style_feature, generated_feature in zip(content_image_features, style_image_features, generated_image_features):
        content_loss_value += content_loss(content_feature, generated_feature)
        style_loss_value += style_loss(style_feature, generated_feature)
    total_loss_value = alpha * content_loss_value + beta * style_loss_value
    return total_loss_value

import streamlit as st
st.title('Neural Style Transfer with VGG19')


content_image_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])



if content_image_file is not None and style_image_file is not None:
    content_image = load_image(content_image_file)
    style_image = load_image(style_image_file)

    st.image(content_image_file, caption="Content Image", use_column_width=True)
    st.image(style_image_file, caption="Style Image", use_column_width=True)

    

    # Initialize the generated image as content image itself
    generated_image = content_image.clone().requires_grad_(True)

     # Optimizer setup
    optimizer = optim.Adam([generated_image], lr=0.003)

    # Defining a class for the VGG model
    class VGG(nn.Module):
        def __init__(self):
            super(VGG, self).__init__()
            self.req_features = ['0', '5', '10', '19', '28']
            self.model = models.vgg19(pretrained=True).features[:29]

        def forward(self, x):
            features = []
            for name, layer in self.model._modules.items():
                x = layer(x)
                if name in self.req_features:
                    features.append(x)
            return features

    # Instantiate the VGG model
    vgg = VGG().to(device).eval()

    # Run the optimization
    progress_bar = st.progress(0)
    for step in range(steps):
        content_image_features = vgg(content_image)
        style_image_features = vgg(style_image)
        generated_image_features = vgg(generated_image)

        total_loss_value = total_loss(content_image_features, style_image_features, generated_image_features)

        optimizer.zero_grad()
        total_loss_value.backward()
        optimizer.step()

        if step % 50 == 0:
            progress_bar.progress((step + 1) / steps)
            st.write(f"Step {step}/{steps}, Total Loss: {total_loss_value.item()}")

    # Save the generated image
    save_image(generated_image, "generated.png")

    # Convert the generated image tensor to a format suitable for display
    generated_image_display = generated_image.cpu().clone().squeeze(0)
    generated_image_display = transforms.ToPILImage()(generated_image_display)

    st.image(generated_image_display, caption="Generated Image", use_column_width=True)

