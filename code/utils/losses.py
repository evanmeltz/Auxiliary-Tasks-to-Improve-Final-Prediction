import torch
import torch.nn.functional as F


def latlon_to_unit_vector(latlon_deg):
    lat = torch.deg2rad(latlon_deg[:, 0])
    lon = torch.deg2rad(latlon_deg[:, 1])
    return torch.stack(
        [torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)],
        dim=1,
    )


def unit_vector_to_latlon(xyz):
    xyz = F.normalize(xyz, dim=1, eps=1e-8)
    lat = torch.rad2deg(torch.asin(xyz[:, 2].clamp(-1.0, 1.0)))
    lon = torch.rad2deg(torch.atan2(xyz[:, 1], xyz[:, 0]))
    return torch.stack([lat, lon], dim=1)


def unit_vector_loss(pred_xyz, target_latlon):
    pred_xyz = F.normalize(pred_xyz, dim=1, eps=1e-8)
    target_xyz = latlon_to_unit_vector(target_latlon)
    return (1.0 - torch.sum(pred_xyz * target_xyz, dim=1)).mean()


def haversine_distance_km(pred_xyz, target_latlon):
    pred_latlon = unit_vector_to_latlon(pred_xyz)
    pred_lat = torch.deg2rad(pred_latlon[:, 0])
    pred_lon = torch.deg2rad(pred_latlon[:, 1])
    target_lat = torch.deg2rad(target_latlon[:, 0])
    target_lon = torch.deg2rad(target_latlon[:, 1])
    dlat = target_lat - pred_lat
    dlon = target_lon - pred_lon
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(pred_lat) * torch.cos(target_lat) * torch.sin(dlon / 2) ** 2
    )
    return 6371.0088 * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))


def make_inverse_frequency_class_weights(encoded_labels, num_classes):
    labels = torch.as_tensor(encoded_labels, dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = torch.zeros(num_classes, dtype=torch.float32)
    nonzero = counts > 0
    weights[nonzero] = 1.0 / counts[nonzero]
    weights[nonzero] = weights[nonzero] / weights[nonzero].mean()
    return weights


def build_categorical_loss_weights(train_dataset, categorical_labels):
    return {
        name: make_inverse_frequency_class_weights(
            train_dataset.labels[name].values,
            len(train_dataset.label_vocab[name]),
        )
        for name in categorical_labels
    }


def auxiliary_losses(preds, batch, categorical_loss_weights):
    device = preds["xyz"].device
    return {
        "dist_sea": F.mse_loss(
            preds["dist_sea"], batch["dist_sea"].to(device, non_blocking=True)
        ),
        "country": F.cross_entropy(
            preds["country"],
            batch["country"].to(device, non_blocking=True),
            weight=categorical_loss_weights["country"].to(device),
        ),
        "climate": F.cross_entropy(
            preds["climate"],
            batch["climate"].to(device, non_blocking=True),
            weight=categorical_loss_weights["climate"].to(device),
        ),
        "land_cover": F.cross_entropy(
            preds["land_cover"],
            batch["land_cover"].to(device, non_blocking=True),
            weight=categorical_loss_weights["land_cover"].to(device),
        ),
    }


def expert_diversity_loss(expert_outputs):
    num_experts = expert_outputs.size(1)
    if num_experts <= 1:
        return expert_outputs.new_tensor(0.0)
    normalized = F.normalize(expert_outputs, dim=2, eps=1e-8)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    off_diagonal = ~torch.eye(num_experts, dtype=torch.bool, device=expert_outputs.device)
    return similarity[:, off_diagonal].pow(2).mean()


def expert_load_balance_loss(gate_probs):
    num_experts = gate_probs.size(1)
    if num_experts <= 1:
        return gate_probs.new_tensor(0.0)
    mean_probs = gate_probs.mean(dim=0)
    uniform = torch.full_like(mean_probs, 1.0 / num_experts)
    return F.mse_loss(mean_probs, uniform) * num_experts


def multitask_loss(preds, batch, target_latlon, categorical_loss_weights, loss_weights):
    coord_loss = unit_vector_loss(preds["xyz"], target_latlon)
    aux = auxiliary_losses(preds, batch, categorical_loss_weights)
    total_loss = (
        loss_weights["coord"] * coord_loss
        + loss_weights["dist_sea"] * aux["dist_sea"]
        + loss_weights["country"] * aux["country"]
        + loss_weights["climate"] * aux["climate"]
        + loss_weights["land_cover"] * aux["land_cover"]
    )
    if "expert_outputs" in preds:
        total_loss = (
            total_loss
            + loss_weights["expert_diversity"] * expert_diversity_loss(preds["expert_outputs"])
            + loss_weights["expert_load_balance"] * expert_load_balance_loss(preds["gate_probs"])
        )
    return total_loss, coord_loss
