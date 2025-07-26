import torch


def get_guidance_scale_embedding(w: torch.Tensor, embedding_dim: int = 512,
                                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert len(w.shape) == 1
    w = w * 1000.0
    device = w.device
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb).to(device)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb

def load_balancing_loss_func(router_logits: tuple, num_experts: int = 4, topk: int = 2):
    router_logits = torch.cat(router_logits, dim=0)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, topk, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    overall_loss = num_experts * torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss