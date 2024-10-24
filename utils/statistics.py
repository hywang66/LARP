import torch
import torch.distributions as dist


def kl_divergence_from_uniform(index_counts):
    """
    Compute KL divergence from a uniform distribution
    :param index_counts: tensor of counts for each index
    :return: KL divergence
    """

    total_counts = index_counts.sum()
    if total_counts == 0:
        # If no counts, return 0.0
        return 0.0
    p_probs = index_counts / total_counts
    num_categories = len(index_counts)
    q_probs = 1.0 / num_categories * torch.ones_like(index_counts)
    kl_div = dist.kl_divergence(dist.Categorical(probs=p_probs), dist.Categorical(probs=q_probs))
    return kl_div.item()


def index_usage_percentage(index_counts):
    """
    Compute the percentage of used (non-zero) indices
    :param index_counts: tensor of counts for each index
    :return: tensor of usage percentages
    """

    num_non_zero_indices = (index_counts > 0).float().sum().item()
    num_indices = len(index_counts)
    return num_non_zero_indices / num_indices

@torch.no_grad()
@torch.autocast(device_type='cuda', enabled=False)
def calculate_topk_accuracy(logits, targets, topk=(1, 5), prepend=''):
    """
    Computes the top-k accuracy for the specified values of k.

    Args:
    logits (torch.Tensor): The predicted logits (unnormalized scores) with shape (batch_size, num_tokens, num_classes).
    targets (torch.Tensor): The true labels with shape (batch_size, num_tokens).
    topk (tuple): A tuple of integers specifying the top-k accuracies to compute.

    Returns:
    dict: A dictionary with keys 'top1' and 'top5' containing the respective accuracies.
    """
    logits = logits.float()
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)

    maxk = max(topk)
    batch_size = targets.size(0)

    # Get the top maxk predictions and their indices
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # Transpose to shape (maxk, batch_size)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # Shape (maxk, batch_size)

    topk_accuracies = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        topk_accuracies[f'{prepend}top{k}_acc'] = correct_k.mul_(100.0 / batch_size).item()

    return topk_accuracies


'''
# Example usage
logits = torch.randn(10, 100)  # Example logits for a batch of 10 samples and 100 classes
targets = torch.randint(0, 100, (10,))  # Example targets for the batch

accuracies = calculate_topk_accuracy(logits, targets)
print(f"Top-1 Accuracy: {accuracies['top1']:.2f}%")
print(f"Top-5 Accuracy: {accuracies['top5']:.2f}%")

'''
