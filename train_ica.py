import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader

import math
import argparse
import numpy as np
import torch.nn.functional as F

from ica import ica_mlp
from data_ica import ica_dataset, resample_rows_per_column
from utils import log_sum_exp, raise_measure_error


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="training epochs")
    parser.add_argument("--bz", type=int, default=128, help="batch size")
    parser.add_argument("--milestones", type=list, default=[30, 100], help="milestones for learning rate scheduling")
    parser.add_argument("--gamma", type=float, default=0.5, help="gamma for learning rate scheduling")

    parser.add_argument("--dims", type=list, default=[2048, 100, 50, 1])
    parser.add_argument("--features_path", type=str, default="D://projects//open_cross_entropy//osr_closed_set_all_you_need-main//features//cifar-10-10_msp_optimal_classifier32_0", help="path to the features")
    parser.add_argument("--feature_name", type=str, default="module.avgpool", help="name of the features")
    parser.add_argument("--save_path", type=str, default="D://projects//open_cross_entropy//osr_closed_set_all_you_need-main//ica.pth")
    parser.add_argument("--loss_path", type=str, default="D://projects//open_cross_entropy//osr_closed_set_all_you_need-main//ica.txt")

    args = parser.parse_args()

    return args


def ica_loss(marg_logit, joint_logit):

    # L268-L324 in https://github.com/pbrakel/anica/blob/master/train.py#L216
    # f_joint should be small whereas f_marginal should be large
    disc_loss = torch.mean(F.softplus(-marg_logit)) + torch.mean(F.softplus(joint_logit))

    # Following based on https://github.com/rdevon/DIM/blob/bac4765a8126746675f517c7bfa1b04b88044d51/cortex_DIM/models/ndm.py#L29
    #E_pos = get_positive_expectation(joint_logit, measure="DV")
    #E_neg = get_negative_expectation(marg_logit, measure="DV")
    #score = E_pos - E_neg
    #disc_loss = -score

    return disc_loss


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    https://github.com/rdevon/DIM/blob/bac4765a8126746675f517c7bfa1b04b88044d51/cortex_DIM/functions/gan_losses.py#L22
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    https://github.com/rdevon/DIM/blob/bac4765a8126746675f517c7bfa1b04b88044d51/cortex_DIM/functions/gan_losses.py#L22
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq



def train(args, train_data):

    train_loader = DataLoader(train_data, batch_size=args.bz, shuffle=True)
    model = ica_mlp(dims=args.dims) 
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    args.milestones = [int(args.epochs*0.35), int(args.epochs*0.7)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)

    losses = []
    best_loss = 1e10
    for e in range(args.epochs):
        loss_epoch = 0
        for i, features_joint in enumerate(train_loader):
            features_marginal = resample_rows_per_column(features_joint.numpy())
            features_marginal = torch.from_numpy(features_marginal)
            features_joint = features_joint.float()
            features_marginal = features_marginal.float()
            pred_joint = model(features_joint)
            pred_marginal = model(features_marginal)
            loss = ica_loss(pred_marginal, pred_joint)

            loss.backward()
            optimizer.step()
            loss_epoch += loss.detach()

        losses.append(loss_epoch)
        scheduler.step()
        if loss_epoch < best_loss:
            torch.save(model.state_dict(), args.save_path)  
        print("Epoch ", e, "loss is ", loss_epoch)

    with open(args.loss_path, "w") as f:
         f.write(str(losses))

    return losses


if __name__ == "__main__":

    """
    smaller it is, more independent
    """
    # load parameters
    args = get_parser()
    
    train_data = ica_dataset(feature_path=args.features_path, feature_name=args.feature_name)
    losses = train(args, train_data)

