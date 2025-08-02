import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Image loader
def load_image(path, max_size=400):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Step 3: Gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

# Step 4: Feature extractor
def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in ['0', '5', '10', '19', '28']:  # style layers
            features[f'style_{name}'] = x
        if name == '21':  # content layer
            features['content'] = x
    return features

# Step 5: Load content & style
content = load_image("content.jpg")
style = load_image("style.jpg")

# Step 6: Load VGG19 model
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Step 7: Feature extraction
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features if 'style' in layer}

# Step 8: Target image
target = content.clone().requires_grad_(True).to(device)
optimizer = optim.Adam([target], lr=0.003)
style_weight = 1e6
content_weight = 1

# Step 9: Training loop
for step in range(1, 201):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['content'] - content_features['content'])**2)

    style_loss = 0
    for layer in style_grams:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        style_loss += torch.mean((target_gram - style_gram)**2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Total loss: {total_loss.item()}")

# Step 10: Final image conversion & save
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze(0)
    image = image.numpy().transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return image.clip(0, 1)

final_img = im_convert(target)
plt.imshow(final_img)
plt.axis('off')
plt.savefig("output.png")
plt.show()
