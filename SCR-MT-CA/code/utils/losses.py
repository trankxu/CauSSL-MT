import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

"""
The different uncertainty methods loss implementation.
Including:
    Ignore, Zeros, Ones, SelfTrained, MultiClass
"""

METHODS = ['U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', 'U-MultiClass']
CLASS_NUM = [1113, 6705, 514, 327, 1099, 115, 142]
CLASS_WEIGHT = torch.Tensor([10000/i for i in CLASS_NUM]).cuda()

#CLASS_WEIGHT = torch.ones(7).cuda()
class Loss_Zeros(object):
    """
    map all uncertainty values to 0
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCELoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 0
        return self.base_loss(output, target)

class Loss_Ones(object):
    """
    map all uncertainty values to 1
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 1
        return self.base_loss(output, target)

class cross_entropy_loss(object):
    """
    map all uncertainty values to a unique value "2"
    """
    
    def __init__(self):
        self.base_loss = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT, reduction='mean')
    
    def __call__(self, output, target):
        # target[target == -1] = 2
        output_softmax = F.softmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        return self.base_loss(output_softmax, target.long())

class NCELoss(nn.Module):
    def __init__(self, batch_size, temperature=0.07):
        super(NCELoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

    def forward(self, feature_map1, feature_map2):
        # Flatten the spatial dimensions
        feature_map1_avg = F.adaptive_avg_pool2d(feature_map1, (1, 1)).squeeze()
        feature_map2_avg = F.adaptive_avg_pool2d(feature_map2, (1, 1)).squeeze()

        # Normalize feature vectors
        feature_map1_normalized = F.normalize(feature_map1_avg, dim=1)
        feature_map2_normalized = F.normalize(feature_map2_avg, dim=1)

        # Compute dot products
        dot_products = torch.matmul(feature_map1_normalized, feature_map2_normalized.t())

        # Compute InfoNCE loss
        numerator = torch.diag(dot_products)
        denominator = torch.exp(dot_products / self.temperature).sum(dim=1)
        infonce_loss = -torch.log(numerator / (denominator - torch.exp(numerator / self.temperature))).mean()
        
        return infonce_loss

        
# class weighted_cross_entropy_loss(object):
#     """
#     map all uncertainty values to a unique value "2"
#     """
    
#     def __init__(self):
#         self.base_loss = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT, reduction='mean')
    
#     def __call__(self, output, target):
#         # target[target == -1] = 2
#         output_softmax = F.softmax(output, dim=1)
#         target = torch.argmax(target, dim=1)
#         return self.base_loss(output_softmax, target.long())

def get_UncertaintyLoss(method):
    assert method in METHODS
    
    if method == 'U-Zeros':
        return Loss_Zeros()

    if method == 'U-Ones':
        return Loss_Ones()
    
    if method == 'U-MultiClass':
        return Loss_MultiClass()

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2 * CLASS_WEIGHT
    return mse_loss

def cam_attention_map(activations, channel_weight):
    # activations 48*49*1024
    # channel_weight 48*1024
    attention = activations.permute(1,0,2).mul(channel_weight)
    attention = attention.permute(1,0,2)
    attention = torch.sum(attention, -1)
    attention = torch.reshape(attention, (48, 7, 7))

    return attention

def cam_activation(batch_feature, channel_weight):
    # batch_feature = batch_feature.permute(0,2,3,1)#48 7 7 1024
    # activations = torch.reshape(batch_feature, (batch_feature.shape[0], -1, batch_feature.shape[3]))#48*49*1024
    
    # attention = activations.permute(1,0,2)#.mul(channel_weight)#49*48*1024
    # attention = attention.permute(1,2,0)#48*1024*49
    # attention = F.softmax(attention, -1)#48*1024*49

    # activations2 = activations.permute(0, 2, 1) #48 1024 49
    # activations2 = activations2 * attention 
    # activations2 = torch.sum(activations2, -1)#48*1024
    batch_feature = batch_feature.permute(0,2,3,1)
    #48*49*1024
    activations = torch.reshape(batch_feature, (batch_feature.shape[0], -1, batch_feature.shape[3]))
    
    #49*48*1024
    attention = activations.permute(1,0,2).mul(channel_weight)
    #48*49*1024
    attention = attention.permute(1,0,2)
    #48*49
    attention = torch.sum(attention, -1)
    attention = F.softmax(attention, -1)

    activations2 = activations.permute(2, 0, 1) #1024*48*49
    activations2 = activations2 * attention 
    activations2 = torch.sum(activations2, -1) #1024*48
    #48 1024 
    activations2 = activations2.permute(1,0)

    return activations2

# def relation_mse_loss_cam(activations, ema_activations, model, label):
#     """Takes softmax on both sides and returns MSE loss

#     Note:
#     - Returns the sum over all examples. Divide by the batch size afterwards
#       if you want the mean.
#     - Sends gradients to inputs but not the targets.
#     """
#     weight = model.module.densenet121.classifier[0].weight
#     #48*1024
#     channel_weight = label.mm(weight)

#     activations = cam_activation(activations.clone(), channel_weight)
#     ema_activations = cam_activation(ema_activations.clone(), channel_weight)
    
#     assert activations.size() == ema_activations.size()

#     activations = torch.reshape(activations, (activations.shape[0], -1))
#     ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

#     similarity = activations.mm(activations.t())
#     norm_similarity = similarity / torch.norm(similarity, p=2)

#     ema_similarity = ema_activations.mm(ema_activations.t())
#     norm_ema_similarity = ema_similarity / torch.norm(ema_similarity, p=2)

#     similarity_mse_loss = (norm_similarity-norm_ema_similarity)**2
#     return similarity_mse_loss


def relation_mse_loss_cam(activations, ema_activations, model, label):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    weight = model.module.densenet121.classifier[0].weight
    #48*1024
    channel_weight = label.mm(weight)

    activations = cam_activation(activations.clone(), channel_weight)
    ema_activations = cam_activation(ema_activations.clone(), channel_weight)
    
    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss

def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss


def feature_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    # similarity = activations.mm(activations.t())
    # norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    # norm_similarity = similarity / norm

    # ema_similarity = ema_activations.mm(ema_activations.t())
    # ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    # ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (activations-ema_activations)**2
    return similarity_mse_loss


def sigmoid_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.sigmoid(input_logits)
    target_softmax = torch.sigmoid(target_logits)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    mse_loss = loss_fn(input_softmax, target_softmax)
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
    

from sklearn.cluster import KMeans


def cluster_consistency_loss(activations, ema_activations,k = 3):
    """Calculates the consistency loss based on KMeans of images.

    Parameters:
    - activations: A 2D array of shape (n_samples, n_features) containing the input images.
    - ema_activations: A 2D array of shape (n_clusters, n_features) containing the cluster centers
      predicted by the model at the previous time step.

    Returns:
    - loss: A float value representing the consistency loss.
    """
    # Create a k-means model


    kmeans_input = KMeans(n_clusters = k,random_state = 0).fit(activations.detach().cpu().numpy())
    kmeans_ema = KMeans(n_clusters = k,random_state = 0).fit(ema_activations.detach().cpu().numpy())
    
    # Get the cluster centers predicted by the model for the current time step
    origin_centers = kmeans_input.cluster_centers_
    ema_centers = kmeans_ema.cluster_centers_
    
    # Calculate the average distance between the current and previous cluster centers
    
    loss = np.mean(np.linalg.norm(origin_centers - ema_centers, axis=1))
    
    return loss
 

