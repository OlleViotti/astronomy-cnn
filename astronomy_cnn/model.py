from torch import nn


class SpectraCNN(nn.Module):
    def __init__(self, num_labels=3, dropout_rate=0.01):
        super(SpectraCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=16, kernel_size=21, stride=1, padding=10
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=15, stride=1, padding=7
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool1d(kernel_size=2),
        )
        # Global average pooling to collapse the temporal dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Output shape will be (N, 64, 1)
        # Fully connected layer to predict the labels
        self.fc = nn.Linear(64, num_labels)

    def forward(self, x):
        # Ensure input x has shape (N, L); add channel dimension to make it (N, 1, L)
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = self.global_pool(x)  # Shape becomes (N, 64, 1)
        x = x.view(x.size(0), -1)  # Flatten to shape (N, 64)
        x = self.fc(x)
        return x
