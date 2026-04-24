import torch
import torch.nn.functional as F


def infonce_loss(anchor: torch.Tensor, positive: torch.Tensor,
                 negatives: list[torch.Tensor] | None = None,
                 tau: float = 0.07) -> torch.Tensor:
    sim_ap = (anchor @ positive.T) / tau
    if negatives:
        neg_sims = [anchor @ n.T / tau for n in negatives]
        sim = torch.cat([sim_ap] + neg_sims, dim=1)
    else:
        sim = sim_ap
    labels = torch.arange(anchor.size(0), device=sim.device)
    loss_a = F.cross_entropy(sim, labels)
    loss_p = F.cross_entropy(sim_ap.T, labels)
    return 0.5 * (loss_a + loss_p)
