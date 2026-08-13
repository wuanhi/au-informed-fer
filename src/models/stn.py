import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformerNetwork(nn.Module):
    """
    STN used in A1.

    Flow:
        input x
        -> localization network
        -> affine parameters theta
        -> affine_grid
        -> grid_sample(original x)
        -> transformed x

    Architecture follows the STN part of EmoNeXt.
    """

    def __init__(self):
        super().__init__()

        # Localization network
        #
        # [B, 3, 224, 224]
        # -> [B, 8, 218, 218]
        # -> [B, 8, 109, 109]
        # -> [B, 10, 105, 105]
        # -> [B, 10, 52, 52]
        self.localization = nn.Sequential(
            nn.Conv2d(
                3,
                8,
                kernel_size=7,
            ),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(
                2,
                stride=2,
            ),
            nn.ReLU(True),

            nn.Conv2d(
                8,
                10,
                kernel_size=5,
            ),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(
                2,
                stride=2,
            ),
            nn.ReLU(True),
        )

        # Affine regressor
        #
        # [B, 10, 52, 52]
        # -> [B, 27040]
        # -> [B, 32]
        # -> [B, 6]
        # -> [B, 2, 3]
        self.fc_loc = nn.Sequential(
            nn.Linear(
                10 * 52 * 52,
                32,
            ),
            nn.ReLU(True),
            nn.Linear(
                32,
                3 * 2,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Match EmoNeXt initialization for Conv2d / Linear,
        then force final affine layer to identity transform.
        """

        for module in self.modules():
            if isinstance(
                module,
                (nn.Conv2d, nn.Linear),
            ):
                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

        # Final affine regression layer:
        # theta starts at
        #
        # [1 0 0]
        # [0 1 0]
        self.fc_loc[2].weight.data.zero_()

        self.fc_loc[2].bias.data.copy_(
            torch.tensor(
                [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                ],
                dtype=torch.float,
            )
        )

    def predict_theta(self, x):
        """
        Predict affine matrices.

        Returns:
            theta: [B, 2, 3]
        """

        xs = self.localization(x)

        xs = xs.view(
            -1,
            10 * 52 * 52,
        )

        theta = self.fc_loc(xs)

        theta = theta.view(
            -1,
            2,
            3,
        )

        return theta

    def transform(self, x, theta=None):
        """
        Apply affine transformation to the original input x.

        Returns:
            transformed x: same shape as x
        """

        if theta is None:
            theta = self.predict_theta(x)

        grid = F.affine_grid(
            theta,
            x.size(),
            align_corners=True,
        )

        transformed = F.grid_sample(
            x,
            grid,
            align_corners=True,
        )

        return transformed

    def forward(self, x):
        theta = self.predict_theta(x)

        transformed = self.transform(
            x,
            theta=theta,
        )

        return transformed

    def forward_with_theta(self, x):
        """
        Diagnostic forward used after training A1.

        Returns:
            transformed image
            theta
        """

        theta = self.predict_theta(x)

        transformed = self.transform(
            x,
            theta=theta,
        )

        return transformed, theta

    def get_sampling_grid(self, x):
        """
        Diagnostic helper for later A1 failure analysis.

        Used to compute out-of-bound sampling ratio.
        """

        theta = self.predict_theta(x)

        grid = F.affine_grid(
            theta,
            x.size(),
            align_corners=True,
        )

        return grid