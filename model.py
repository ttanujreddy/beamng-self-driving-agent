import torch
import torch.nn as nn

# Call model.train() during training and model.eval() + torch.no_grad() at inference.


class DrivingModel(nn.Module):
    
    #Feedforward regression model for BeamNG driving control.

    # Input:  (B, 6) — [distFromCenter, headingAngle, curvature, roadWidth, drivability, Speed]
    # Output: (B, 3) — [steering ∈ (-1,1), throttle ∈ (0,1), brake ∈ (0,1)]
  

    def __init__(self):
        super().__init__()

        # Two hidden layers: 6 -> 64 -> 32
        # Dropout(0.1) between them to reduce overfitting on small datasets (~500 frames/lap)
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Single output layer produces 3 raw logits before per-output activations
        self.head = nn.Linear(32, 3)

    def forward(self, x):
        x = x.float()

        # Support both batched (B, 6) and single-sample (6,) inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)

        out = self.head(self.net(x))

        # Tanh constrains steering to (-1, 1): negative = left, positive = right
        steering = torch.tanh(out[:, 0:1])

        # Sigmoid constrains throttle and brake to (0, 1)
        throttle_brake = torch.sigmoid(out[:, 1:3])

        return torch.cat([steering, throttle_brake], dim=1)
    
    def save(self, path="best_model.pth"):
        # Save model weights to disk. Safe to call mid-training.
        torch.save(self.state_dict(), path)

    def load(self, path="best_model.pth", map_location=None):
        #Load weights into this model in-place.
        if map_location is None:
            map_location = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=map_location))