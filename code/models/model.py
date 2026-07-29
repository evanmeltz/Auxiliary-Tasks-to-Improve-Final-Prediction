import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_ARCHITECTURES = {
    "base",
    "multiloss",
    "parallel_heads",
    "sequential_heads",
    "moe",
}


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        residual = x
        x = self.res(x)
        return F.relu(x + residual, inplace=True)


class CNNBase(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        c = base_channels
        self.block1 = CNNBlock(3, c)
        self.block2 = CNNBlock(c, c * 2)
        self.block3 = CNNBlock(c * 2, c * 4)
        self.block4 = CNNBlock(c * 4, c * 8)
        self.block5 = CNNBlock(c * 8, c * 8)
        self.block6 = CNNBlock(c * 8, c * 8)
        self.block7 = CNNBlock(c * 8, c * 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.head = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, kernel_size=3, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def extract_features(self, images):
        x = self.pool(self.block1(images))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.pool(self.block4(x))
        x = self.pool(self.block5(x))
        x = self.pool(self.block6(x))
        return self.head(self.block7(x))


def make_mlp(input_dim, hidden_dim_1, hidden_dim_2, output_dim, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim_1),
        nn.BatchNorm1d(hidden_dim_1),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim_1, hidden_dim_2),
        nn.BatchNorm1d(hidden_dim_2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim_2, output_dim),
    )


def detached_aux_features(dist_sea, country, climate, land_cover):
    return torch.cat(
        [
            dist_sea.detach().unsqueeze(1),
            F.softmax(country.detach(), dim=1),
            F.softmax(climate.detach(), dim=1),
            F.softmax(land_cover.detach(), dim=1),
        ],
        dim=1,
    )


class BaseGeoCNN(CNNBase):
    def __init__(self, base_channels=16, dropout=0.1, **_):
        super().__init__(base_channels)
        c = base_channels
        self.mlp = make_mlp(c * 8, c * 8, c * 4, 3, dropout)

    def forward(self, images):
        return self.mlp(self.extract_features(images))


class MultiLossGeoCNN(CNNBase):
    def __init__(self, base_channels=16, dropout=0.1, num_country_classes=None,
                 num_climate_classes=None, num_land_cover_classes=None, **_):
        super().__init__(base_channels)
        feature_dim = base_channels * 8
        hidden_1, hidden_2 = feature_dim, base_channels * 4
        self.xyz_mlp = make_mlp(feature_dim, hidden_1, hidden_2, 3, dropout)
        self.dist_sea_mlp = make_mlp(feature_dim, hidden_1, hidden_2, 1, dropout)
        self.country_mlp = make_mlp(feature_dim, hidden_1, hidden_2, num_country_classes, dropout)
        self.climate_mlp = make_mlp(feature_dim, hidden_1, hidden_2, num_climate_classes, dropout)
        self.land_cover_mlp = make_mlp(feature_dim, hidden_1, hidden_2, num_land_cover_classes, dropout)

    def forward(self, images):
        features = self.extract_features(images)
        return {
            "xyz": self.xyz_mlp(features),
            "dist_sea": self.dist_sea_mlp(features).squeeze(1),
            "country": self.country_mlp(features),
            "climate": self.climate_mlp(features),
            "land_cover": self.land_cover_mlp(features),
        }


class ParallelHeadsGeoCNN(MultiLossGeoCNN):
    def __init__(self, base_channels=16, dropout=0.1, num_country_classes=None,
                 num_climate_classes=None, num_land_cover_classes=None, **kwargs):
        super().__init__(base_channels, dropout, num_country_classes,
                         num_climate_classes, num_land_cover_classes, **kwargs)
        feature_dim = base_channels * 8
        coord_dim = feature_dim + 1 + num_country_classes + num_climate_classes + num_land_cover_classes
        self.xyz_mlp = make_mlp(coord_dim, coord_dim, coord_dim // 2, 3, dropout)

    def forward(self, images):
        features = self.extract_features(images)
        dist_sea = self.dist_sea_mlp(features).squeeze(1)
        country = self.country_mlp(features)
        climate = self.climate_mlp(features)
        land_cover = self.land_cover_mlp(features)
        coord_features = torch.cat(
            [features, detached_aux_features(dist_sea, country, climate, land_cover)], dim=1
        )
        return {
            "xyz": self.xyz_mlp(coord_features),
            "dist_sea": dist_sea,
            "country": country,
            "climate": climate,
            "land_cover": land_cover,
        }


class SequentialHeadsGeoCNN(CNNBase):
    def __init__(self, base_channels=16, dropout=0.1, num_country_classes=None,
                 num_climate_classes=None, num_land_cover_classes=None, **_):
        super().__init__(base_channels)
        feature_dim = base_channels * 8
        hidden_1, hidden_2 = feature_dim, base_channels * 4
        country_dim = feature_dim + num_country_classes
        aux_dim = country_dim + num_climate_classes + num_land_cover_classes
        coord_dim = aux_dim + 1
        self.country_mlp = make_mlp(feature_dim, hidden_1, hidden_2, num_country_classes, dropout)
        self.climate_mlp = make_mlp(country_dim, hidden_1, hidden_2, num_climate_classes, dropout)
        self.land_cover_mlp = make_mlp(country_dim, hidden_1, hidden_2, num_land_cover_classes, dropout)
        self.dist_sea_mlp = make_mlp(aux_dim, hidden_1, hidden_2, 1, dropout)
        self.xyz_mlp = make_mlp(coord_dim, coord_dim, coord_dim // 2, 3, dropout)

    def forward(self, images):
        features = self.extract_features(images)
        country = self.country_mlp(features)
        country_features = torch.cat([features, F.softmax(country.detach(), dim=1)], dim=1)
        climate = self.climate_mlp(country_features)
        land_cover = self.land_cover_mlp(country_features)
        dist_features = torch.cat(
            [
                features,
                F.softmax(country.detach(), dim=1),
                F.softmax(climate.detach(), dim=1),
                F.softmax(land_cover.detach(), dim=1),
            ],
            dim=1,
        )
        dist_sea = self.dist_sea_mlp(dist_features).squeeze(1)
        xyz = self.xyz_mlp(torch.cat([dist_features, dist_sea.detach().unsqueeze(1)], dim=1))
        return {
            "xyz": xyz,
            "dist_sea": dist_sea,
            "country": country,
            "climate": climate,
            "land_cover": land_cover,
        }


class ExpertMLP(nn.Module):
    def __init__(self, feature_dim, dropout):
        super().__init__()
        self.mlp = make_mlp(feature_dim, feature_dim, feature_dim, feature_dim, dropout)

    def forward(self, features):
        return self.mlp(features)


class MoEGeoCNN(MultiLossGeoCNN):
    def __init__(self, base_channels=16, dropout=0.1, num_country_classes=None,
                 num_climate_classes=None, num_land_cover_classes=None,
                 num_experts=4, gate_temperature=1.0, **kwargs):
        super().__init__(base_channels, dropout, num_country_classes,
                         num_climate_classes, num_land_cover_classes, **kwargs)
        feature_dim = base_channels * 8
        aux_dim = 1 + num_country_classes + num_climate_classes + num_land_cover_classes
        coord_dim = feature_dim + aux_dim
        self.num_experts = num_experts
        self.gate_temperature = gate_temperature
        self.experts = nn.ModuleList([ExpertMLP(feature_dim, dropout) for _ in range(num_experts)])
        self.gate_mlp = make_mlp(coord_dim, coord_dim, coord_dim // 2, num_experts, dropout)
        self.final_xyz_mlp = make_mlp(coord_dim, feature_dim, base_channels * 4, 3, dropout)
        del self.xyz_mlp

    def forward(self, images):
        features = self.extract_features(images)
        dist_sea = self.dist_sea_mlp(features).squeeze(1)
        country = self.country_mlp(features)
        climate = self.climate_mlp(features)
        land_cover = self.land_cover_mlp(features)
        aux_features = detached_aux_features(dist_sea, country, climate, land_cover)
        coord_features = torch.cat([features, aux_features], dim=1)
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
        gate_logits = self.gate_mlp(coord_features)
        gate_probs = F.softmax(gate_logits / max(float(self.gate_temperature), 1e-6), dim=1)
        mixed_features = torch.sum(expert_outputs * gate_probs.unsqueeze(-1), dim=1)
        xyz = self.final_xyz_mlp(torch.cat([mixed_features, aux_features], dim=1))
        return {
            "xyz": xyz,
            "dist_sea": dist_sea,
            "country": country,
            "climate": climate,
            "land_cover": land_cover,
            "expert_outputs": expert_outputs,
            "gate_logits": gate_logits,
            "gate_probs": gate_probs,
        }


MODEL_CLASSES = {
    "base": BaseGeoCNN,
    "multiloss": MultiLossGeoCNN,
    "parallel_heads": ParallelHeadsGeoCNN,
    "sequential_heads": SequentialHeadsGeoCNN,
    "moe": MoEGeoCNN,
}


def build_model(architecture, **kwargs):
    if architecture not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model architecture {architecture!r}. "
            f"Choose from: {', '.join(sorted(SUPPORTED_ARCHITECTURES))}."
        )
    return MODEL_CLASSES[architecture](**kwargs)


# Backward-compatible name for code that imports GeoCNN directly.
GeoCNN = BaseGeoCNN
