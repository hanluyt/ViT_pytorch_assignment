import typing
import io
import os
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets

from models.modeling import VisionTransformer, CONFIGS

# Prepare Model
config = CONFIGS["R50-ViT-B_16"]
model = VisionTransformer(config, num_classes=10, zero_head=False, img_size=224, vis=True)
checkpoint = torch.load('./attention_data/ViT.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
class_label = testset.classes


# test image
def attention(im, x, class_label, name):
    logits, att_mat = model(x.unsqueeze(0))  # att_mat: list  each head: (1, 12, 197, 197)
    att_mat = torch.stack(att_mat).squeeze(1)  # att_mat: (12, 12, 197, 197)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)  # (12, 197, 197)
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
    # joint_attention: len: 12, each shape: (197, 197)
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")
    plt.figure(figsize=(16, 16))
    plt.subplot(121)
    plt.title("Original")
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(122)
    plt.title("Attention map")
    plt.imshow(result)
    plt.axis('off')
    plt.savefig(f'{name}_attention.png')
    probs = torch.nn.Softmax(dim=-1)(logits)
    top5 = torch.argsort(probs, dim=-1, descending=True)
    print("Prediction Label ")
    for idx in top5[0, :5]:
        print(f'{probs[0, idx.item()]:.5f}:{class_label[idx.item()]}  ', end='')

if __name__ == "__main__":
    im = Image.open('dog.png')
    x = transform(im)
    attention(im, x, class_label, "dog")
    im = Image.open('cat.png')
    x = transform(im)
    attention(im, x, class_label, "cat")
    im = Image.open('bird.png')
    x = transform(im)
    attention(im, x, class_label, "bird")

