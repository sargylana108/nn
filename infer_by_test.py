import os
import sys
import json
import base64
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

# Model definition (LiteUNetWithAttention)
class AttentionGate(nn.Module):
    def __init__(self, x_channels, g_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.theta_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.out_conv = nn.Conv2d(x_channels, g_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, g):
        if x.size(2) != g.size(2) or x.size(3) != g.size(3):
            g = F.interpolate(g, size=x.size()[2:], mode="bilinear", align_corners=False)

        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        combined = F.relu(theta_x + phi_g, inplace=True)
        psi = self.sigmoid(self.psi(combined))
        return self.out_conv(x * psi)

class LiteUNetWithAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(LiteUNetWithAttention, self).__init__()
        base_model = models.resnet18(weights=None)  # No internet access, initialize without pre-trained weights

        # Extract encoder levels from ResNet18
        self.enc1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)
        self.enc2 = nn.Sequential(base_model.maxpool, base_model.layer1)
        self.enc3 = base_model.layer2
        self.enc4 = base_model.layer3
        self.enc5 = base_model.layer4

        # Bottleneck
        self.bottleneck = self._conv_block(512, 256)

        # Attention Gates
        self.att4 = AttentionGate(256, 128, 128)
        self.att3 = AttentionGate(128, 64, 64)
        self.att2 = AttentionGate(64, 64, 32)
        self.att1 = AttentionGate(64, 64, 32)

        # Decoders
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=1)
        self.decoder4 = self._conv_block(128, 128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1)
        self.decoder3 = self._conv_block(64, 64)

        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.decoder2 = self._conv_block(64, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1)
        self.decoder1 = self._conv_block(64, 64)

        # Output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        bottleneck = self.bottleneck(enc5)

        up4 = self.upconv4(bottleneck)
        att4 = self.att4(enc4, up4)
        merge4 = torch.cat([up4, att4], dim=1)
        merge4 = self.conv4(merge4)
        dec4 = self.decoder4(merge4)

        up3 = self.upconv3(dec4)
        att3 = self.att3(enc3, up3)
        merge3 = torch.cat([up3, att3], dim=1)
        merge3 = self.conv3(merge3)
        dec3 = self.decoder3(merge3)

        up2 = self.upconv2(dec3)
        att2 = self.att2(enc2, up2)
        merge2 = torch.cat([up2, att2], dim=1)
        merge2 = self.conv2(merge2)
        dec2 = self.decoder2(merge2)

        up1 = self.upconv1(dec2)
        att1 = self.att1(enc1, up1)
        merge1 = torch.cat([up1, att1], dim=1)
        merge1 = self.conv1(merge1)
        dec1 = self.decoder1(merge1)

        dec1 = F.interpolate(dec1, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.final_conv(dec1)

# Dataset and Output paths
dataset_path, output_path = sys.argv[1:]

# Device configuration
device = torch.device("cpu")

# Initialize and load weights
model = LiteUNetWithAttention(in_channels=3, out_channels=1).to(device)
weights_path = "unet4_best_updated_weights.pth"

try:
    saved_weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(saved_weights, strict=False)
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {e}")
    sys.exit(1)

model.eval()

# Inference functions
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (480, 288))
    image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    return image.unsqueeze(0)

def postprocess_mask(mask_tensor, original_shape):
    mask = torch.sigmoid(mask_tensor).squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    return cv2.resize(mask, original_shape[::-1])

# Accumulate results
results_dict = {}

for image_name in os.listdir(dataset_path):
    if image_name.lower().endswith(".jpg"):
        image_path = os.path.join(dataset_path, image_name)
        original_image = cv2.imread(image_path)
        original_shape = original_image.shape[:2]

        input_tensor = preprocess_image(image_path).to(device)

        with torch.no_grad():
            output_mask = model(input_tensor)

        binary_mask = postprocess_mask(output_mask, original_shape)

        _, encoded_img = cv2.imencode(".png", binary_mask)
        encoded_str = base64.b64encode(encoded_img).decode("utf-8")

        results_dict[image_name] = encoded_str

# Save results to JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, ensure_ascii=False)

print(f"Inference completed. Results saved to {output_path}")