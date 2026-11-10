"""Paper Eq. 6/20 losses with explicit class-last layout and reductions.

Lovasz gradient/sorting follows the released loss/recon_loss.py helpers and the
MIT-licensed LovaszSoftmax implementation (licenses/LovaszSoftmax-MIT.txt).
This standalone flat adapter avoids the released wrapper's 18-class shape tests.
"""
import torch
import torch.nn.functional as F


def lovasz_softmax_flat(probabilities, labels):
    if probabilities.ndim != 2 or labels.ndim != 1 or probabilities.shape[0] != labels.numel():
        raise ValueError("Expected probabilities [N,C] and labels [N]")
    if labels.numel() == 0:
        return probabilities.sum() * 0
    losses = []
    for category in range(probabilities.shape[1]):
        foreground = (labels == category).to(probabilities.dtype)
        if not bool(foreground.any()):
            continue
        errors, order = (foreground - probabilities[:, category]).abs().sort(descending=True)
        sorted_foreground = foreground[order]
        total = sorted_foreground.sum()
        intersection = total - sorted_foreground.cumsum(0)
        union = total + (1 - sorted_foreground).cumsum(0)
        gradient = 1 - intersection / union
        if gradient.numel() > 1:
            gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
        losses.append(torch.dot(errors, gradient))
    return torch.stack(losses).mean() if losses else probabilities.sum() * 0


def occupancy_loss(logits, labels, weights, mask=None):
    logits, labels = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
    valid = labels != 255
    if mask is not None:
        valid &= mask.reshape(-1).bool()
    if not bool(valid.any()):
        raise ValueError("No supervised voxels/queries in this batch")
    logits, labels = logits[valid], labels[valid]
    ce = F.cross_entropy(logits, labels)
    lovasz = lovasz_softmax_flat(logits.softmax(-1), labels)
    return weights["ce"] * ce + weights["lovasz"] * lovasz, {"ce": ce, "lovasz": lovasz}


def latent_loss(predictions, targets, weights):
    """Average each loss over voxels/channels, then equally over three planes.

    The explicit per-plane mean approximates Eq. 20's expectation over uv and
    avoids the larger XY plane implicitly receiving more weight.
    """
    l1 = torch.stack([F.l1_loss(p, t) for p, t in zip(predictions, targets)]).mean()
    l2 = torch.stack([F.mse_loss(p, t) for p, t in zip(predictions, targets)]).mean()
    return weights["latent_l1"] * l1 + weights["latent_l2"] * l2, {"latent_l1": l1, "latent_l2": l2}
