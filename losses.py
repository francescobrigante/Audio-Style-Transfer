import torch
import torch.nn as nn
import torch.nn.functional as F
from discriminator import Discriminator

# ---------------------------------------------
# InfoNCE loss implemented over style embeddings
# ---------------------------------------------
def infoNCE_loss(style_emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    # make unit-norm embeddings for cosine similarity
    emb = F.normalize(style_emb, p=2, dim=1)                        # (B, d)
    # similarity matrix
    sim = emb @ emb.t()                                             # (B, B)
    
    # mask self-similarities by adding a large negative value for later softmax computation
    mask = torch.eye(sim.size(0), device=sim.device).bool()
    sim = sim.masked_fill(mask, -1e9)
    
    # scale by temperature and get log-softmax
    logits = sim / temperature
    log_prob = F.log_softmax(logits, dim=1)                         # (B, B)
    
    # create label mask for positives
    labels = labels.unsqueeze(1)                                    # (B,1)
    pos_mask = labels.eq(labels.t())                                # (B, B) positive pairs, pos_mask[i,j] = 1 if i and j belong to the same class
    pos_mask = pos_mask & ~mask              # exclude self
    
    # Compute mean log-prob over positives for each anchor
    
    # sum over positives then divide by positive count
    pos_log_prob = (log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
    
    # average over batch
    loss = -pos_log_prob.mean()
    
    return loss




# ---------------------------------------------
# Margin-based contrastive loss on class embeddings
# It has been written in the general case of C classes so it can be also used for more than 2 classes
# ---------------------------------------------
def margin_loss(class_emb: torch.Tensor, margin: float = 2.0, weight: float = 1.0) -> torch.Tensor:
    # class_emb shape: (C, d)
    # pairwise distances matrix
    diff = class_emb.unsqueeze(1) - class_emb.unsqueeze(0)
    dist = torch.norm(diff, p=2, dim=2)                                 # (C, C) = (2, 2)
    
    # extract upper triangular part of the distance matrix (because it's symmetric -> we have duplicates)
    C = class_emb.size(0)
    idx = torch.triu_indices(C, C, offset=1)
    pos_distances = dist[idx[0], idx[1]]
    
    loss = F.relu(margin - pos_distances).pow(2).mean()
    return loss





# ---------------------------------------------
# Adversarial loss for the discriminator
# It uses cross-entropy loss for style and class embeddings, and uniformity loss for content embeddings
# The goal is to have high accuracy for style and class embeddings, while content embeddings should be random => no information about instrument class from content
# we use CE loss instead of BINARY-CE so it's easy to add other instrument classes
# ---------------------------------------------
def adversarial_loss(
    style_emb: torch.Tensor,
    class_emb: torch.Tensor,
    content_emb: torch.Tensor,
    discriminator: Discriminator,
    labels: torch.Tensor,
    compute_for_discriminator: bool,
    lambda_content: float = 1.0,
    lambda_class: float = 0.5,
    lambda_style: float = 1.0
):
    """
    Args:
        lambda_content: weight for the content loss
        lambda_class: weight fort class_emb loss
        lambda_style: weight for style_emb loss
    """
    
    device = style_emb.device
    
    if content_emb.dim() == 3:  
        # mean along the sequence dimension for content_emb
        content_emb = content_emb.mean(dim=1)                                           # (B, transformer_dim)
    
    # Discriminator predictions
    style_pred = discriminator(style_emb)                                               # (B, 2)
    content_pred = discriminator(content_emb)                                           # (B, 2)
    class_pred = discriminator(class_emb) if class_emb is not None else None            # (2, 2)

    # Losses
    style_loss = nn.CrossEntropyLoss()(style_pred, labels)
    content_loss = nn.CrossEntropyLoss()(content_pred, labels)
    
    discriminator_loss = lambda_style * style_loss + lambda_content * content_loss
    
    if class_pred is not None:
        # assumes the correct class labels are 0 and 1 for the two classes IN THIS ORDER <----
        class_labels = torch.tensor([0, 1], device=device) 
        class_loss = nn.CrossEntropyLoss()(class_pred, class_labels)
        discriminator_loss += lambda_class * class_loss
        

    # separate logics    
    if compute_for_discriminator:
        generator_loss = None
        
    else:
        # experiment with minimizing negative CE loss => maximizing CE loss <--------
        # minimize -entropy => maximize entropy => uniform distribution 
        content_probs = torch.softmax(content_pred, dim=-1)
        content_entropy = -torch.sum(content_probs * torch.log(content_probs + 1e-8), dim=-1).mean()
        generator_loss = - lambda_content * content_entropy 
        #generator_loss = - lambda_content * content_loss

    return discriminator_loss, generator_loss





# ---------------------------------------------
# Disentanglement loss to encourage style and content embeddings to be independent
# It can use either cross-covariance penalty or Hilbert-Schmidt Independence Criterion (HSIC) with RBF kernel
# HSIC is preferred since it captures both linear and non-linear dependencies
# 
# Complexities:
# - Cross-cov:          O(B d^2) TIME,          O(B d + d^2) MEMORY
# - HSIC:               O(B^3 + B^2 d) TIME,    O(B^2 + B d) MEMORY
# ---------------------------------------------
def disentanglement_loss(style_emb: torch.Tensor, content_emb: torch.Tensor, use_hsic: bool = True, weight=20.0) -> torch.Tensor:
    """
    Args:
        style_emb: Tensor of shape (B, d)
        content_emb: Tensor of shape (B, d)
        use_hsic: if True, use HSIC with sigma computed via median heuristic; otherwise use cross-covariance penalty
    """
    
    B, d = style_emb.shape
    device = style_emb.device

    # Center the embeddings
    S = style_emb - style_emb.mean(dim=0, keepdim=True)                         # (B, d)
    C = content_emb - content_emb.mean(dim=0, keepdim=True)                     # (B, d)
    
        
    # Debug: print norms
    # print(f"Norm of S: {torch.norm(S).item():.4f}")
    # print(f"Norm of C: {torch.norm(C).item():.4f}")

    # Cross-covariance penalty
    if not use_hsic:
        cov = (S.T @ C) / (B - 1)                                               # (d, d)
        loss = (cov ** 2).sum()
        return loss
    
    # HSIC
    else:
        # Compute sigma using median heuristic
        # concat style and content embeddings to compute pairwise distances
        X = torch.cat([style_emb, content_emb], dim=0)                          # (2B, d)
        dist = torch.cdist(X, X, p=2)                                           # (2B, 2B)
        off_diag = dist[torch.triu_indices(dist.size(0), dist.size(0), offset=1)]  # exclude diagonal
        sigma = torch.median(off_diag) if off_diag.numel() > 0 else torch.sqrt(d)  # fallback to sqrt(d) if no off-diagonal elements

        # HSIC: 1 / (B-1)^2 * trace(K H L H)
        # centering matrix H
        H = torch.eye(B, device=device) - (1.0 / B) * torch.ones((B, B), device=device)
        
        # RBF kernel: computes similarities
        def rbf(X):
            norms = (X.unsqueeze(1) - X.unsqueeze(0)).pow(2).sum(-1)
            return torch.exp(-norms / (2 * sigma ** 2))                         # (B, B)
        
        K = rbf(S)
        L = rbf(C)
        
        # print(f"Norm of K: {torch.norm(K).item():.4f}")
        # print(f"Norm of L: {torch.norm(L).item():.4f}")
        
        KH = K @ H
        LH = L @ H
        hsic = torch.trace(KH @ LH) / ((B - 1) ** 2)

        return hsic