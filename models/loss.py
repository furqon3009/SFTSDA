import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from typing import Optional

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, device, temperature=0.2, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = self.device  # 'cuda' #(torch.device('cuda')
        # if features.is_cuda
        # else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_sum = mask.sum(1)
        zeros_idx = torch.where(mask_sum == 0)[0]
        mask_sum[zeros_idx] = 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)


        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = - 1 * mean_log_prob_pos
        loss = loss.mean()

        return loss
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, device, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device
    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).to(self.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()

        return loss
def EntropyLoss(input_):
    mask = input_.ge(0.0000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = - (torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

def Entropy(input: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy
    :param input: the softmax output
    :return: entropy
    """
    entropy = -input * torch.log(input + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def evidential_uncertainty(predictions, target, num_classes, device):
    predictions = predictions.to(device)
    target = target.to(device)

    # one hot encoding
    eye = torch.eye(num_classes).to(torch.float).to(device)
    labels = eye[target]
    # Calculate evidence
    evidence = F.softplus(predictions)

    # Dirichlet distribution paramter
    alpha = evidence + 1

    # Dirichlet strength
    strength = alpha.sum(dim=-1)

    # expected probability
    p = alpha / strength[:, None]

    # calculate error
    error = (labels - p) ** 2

    # calculate variance

    var = p * (1 - p) / (strength[:, None] + 1)

    # loss function
    loss = (error + var).sum(dim=-1)

    return loss.mean()

def evident_dl(predictions):
    # Calculate evidence
    evidence = F.softplus(predictions)

    # Dirichlet distribution paramter
    alpha = evidence + 1

    # Dirichlet strength
    strength = alpha.sum(dim=-1)

    # expected probability
    p = alpha / strength[:, None]

    var = p * (1 - p) / (strength[:, None] + 1)

    evident_entropy = torch.mean(EntropyLoss(p))
    evident_var = torch.mean(var)

    return p, evident_var, evident_entropy

def contrastive_loss(logits_ins, pseudo_labels, mem_labels):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()
    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2) 
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())
    loss = F.cross_entropy(logits_ins, labels_ins)

    return loss

class TEntropyLoss(nn.Module):
    """
    The Tsallis Entropy for Uncertainty Reduction

    Parameters:
        - **t** Optional(float): the temperature factor used in TEntropyLoss
        - **order** Optional(float): the order of loss function
    """

    def __init__(self,
                 t: Optional[float] = 2.0,
                 order: Optional[float] = 2.0):
        super(TEntropyLoss, self).__init__()
        self.t = t
        self.order = order

    def forward(self,
                output: torch.Tensor) -> torch.Tensor:
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)
        entropy_weight = Entropy(softmax_out).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (n_sample * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        entropy_weight = entropy_weight.repeat(1, n_class)
        tentropy = torch.pow(softmax_out, self.order) * entropy_weight
        # weight_softmax_out=softmax_out*entropy_weight
        tentropy = tentropy.sum(dim=0) / softmax_out.sum(dim=0)
        loss = -torch.sum(tentropy) / (n_class * (self.order - 1.0))
        return loss




