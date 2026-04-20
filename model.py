import torch
import torch.nn as nn


INPUT_H = 128
INPUT_W = 128


class DrivingModel(nn.Module):
    """
    CNN that maps a semantic-segmentation camera frame to (steering, throttle, brake).

    Input:  (B, 3, INPUT_H, INPUT_W)  — annotation image, resized from 1024×1024
    Output: (B, 3)                    — [steering ∈ (-1,1), throttle ∈ (0,1), brake ∈ (0,1)]
    """

    def __init__(self):
        super().__init__()

        self.image_reader = nn.Sequential(
            # Step 1 — shrink image from 128×128 to 64×64, detect basic shapes
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            # Step 2 — shrink to 32×32, detect road edges and boundaries
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            # Step 3 — shrink to 16×16, detect larger patterns (road curve, open space)
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            # Step 4 — shrink to 8×8, high-level scene understanding
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 256 channels × 8×8 = 16384 numbers fed into the decision layers
        self.decision_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),          nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.image_reader(x)
        raw = self.decision_layers(features)

        steering = torch.tanh(raw[:, 0:1])          # (-1, 1)
        throttle = torch.sigmoid(raw[:, 1:2])        # (0, 1)
        brake    = torch.sigmoid(raw[:, 2:3])        # (0, 1)

        return torch.cat([steering, throttle, brake], dim=1)


def preprocess(annotation_image) -> torch.Tensor:
    """
    Convert a PIL annotation image (or numpy H×W×3) to a model-ready tensor.
    Resize to INPUT_H×INPUT_W and normalize to [0, 1].

    Returns: (1, 3, INPUT_H, INPUT_W) float32 tensor
    """
    import torchvision.transforms.functional as TF
    from PIL import Image
    import numpy as np

    if isinstance(annotation_image, np.ndarray):
        annotation_image = Image.fromarray(annotation_image)

    img = annotation_image.resize((INPUT_W, INPUT_H), Image.NEAREST)
    tensor = TF.to_tensor(img)           # (3, H, W), float32 in [0, 1]
    return tensor.unsqueeze(0)           # (1, 3, H, W)


if __name__ == "__main__":
    model = DrivingModel()
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # (1, 3)
    print(f"Output (steering, throttle, brake): {out}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
