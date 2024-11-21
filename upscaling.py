import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained ESPCN model from torchvision
model = torch.hub.load('pytorch/vision:v0.10.0', 'espcn', pretrained=True)
model.eval()  # Set the model to evaluation mode

# Load and preprocess the image
image = Image.open('vertebrae_disc.jpg')
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension
])

image_tensor = transform(image)

# Perform super-resolution
with torch.no_grad():
    output_tensor = model(image_tensor)

# Post-process the output
output_image = output_tensor.squeeze().clamp(0, 1).numpy()
output_image = output_image.transpose(1, 2, 0)  # Convert from CHW to HWC format

# Convert to PIL image for saving or displaying
output_image_pil = Image.fromarray((output_image * 255).astype('uint8'))

# Save or display the result
output_image_pil.save('upscaled_image.jpg')
plt.imshow(output_image_pil)
plt.show()
