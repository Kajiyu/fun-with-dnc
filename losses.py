import torch
from visualize import logger as sl
from utils import flat, repackage, _variable


def action_loss(logits, action, criterion, log=None):
    """
        Sum of losses of one hot vectors encoding an action
        :param logits: network output vector of [action, [[type_i, ent_i], for i in ents]]
        :param action: target vector size [7]
        :param criterion: loss function
        :return:
        """
    losses = []
    # print("flat action ::: ", flat(action))
    for idx, action_part in enumerate(flat(action)):
        tgt = _variable(torch.LongTensor([action_part]))
        # print("tgt: ", tgt)
        # print("logits[idx]: ", logits[idx])
        # print("critation: ", criterion(logits[idx], tgt))
        losses.append(criterion(logits[idx], tgt))
    loss = torch.stack(losses, 0).mean()
    if log is not None:
        sl.log_loss(losses, loss)
    return loss


def get_top_prediction(expanded_logits, idxs=None):
    max_idxs = []
    idxs = range(len(expanded_logits)) if idxs is None else idxs
    for idx in idxs:
        _, pidx = expanded_logits[idx].data.topk(1)
        max_idxs.append(pidx.squeeze()[0])
    return tuple(max_idxs)


def combined_ent_loss(logits, action, criterion, log=None):
    """
        some hand tunining of penalties for illegal actions...
            trying to force learning of types.

        action type => type_e...
        :param logits: network output vector of one_hot distributions
            [action, [type_i, ent_i], for i in ents]
        :param action: target vector size [7]
        :param criterion: loss function
        :return:
        """
    losses = []
    for idx, action_part in enumerate(flat(action)):
        tgt = _variable(torch.Tensor([action_part]).float())
        losses.append(criterion(logits[idx], tgt))
    lfs = [[losses[0]]]
    n = 2
    for l in(losses[i:i+n] for i in range(1, len(losses), n)):
        lfs.append(torch.stack(losses, 0).sum())
    loss = torch.stack(lfs, 0).mean()
    if log is not None:
        sl.log_loss(losses, loss)
    return loss


def naive_loss(logits, targets, criterion, log=None):
    """
        Calculate best choice from among targets, and return loss

        :param logits:
        :param targets:
        :param criterion:
        :return: loss
        """
    # copy_logits = depackage(logits)
    # final_action = closest_action(copy_logits, targets)
    loss_idx, _ = min(enumerate([action_loss(repackage(logits), a, criterion) for a in targets]))
    final_action = targets[loss_idx]
    return final_action, action_loss(logits, final_action, criterion, log=log)


def action_loss_for_shortest_path(logits, action, current_state, criterion, log=None):
    from_e = "{0:03d}".format(current_state)
    to_e = "{0:03d}".format(action)
    a_str = from_e + to_e
    a_char_list = list(a_str)
    a_list = [int(_a_char) for _a_char in a_char_list]
    losses = []
    for idx, action_part in enumerate(a_list):
        if idx > 2:
            tgt = _variable(torch.LongTensor([action_part]))
            # print("logits: ", logits[idx])
            # print("tgt: ", tgt)
            # if idx == 5:
                # print(logits[idx].data.numpy()[0].tolist())
            losses.append(criterion(logits[idx], tgt))
    loss = torch.stack(losses, 0).mean()
    if log is not None:
        sl.log_loss(losses, loss)
    return loss


def naive_loss_for_shortest_path(logits, targets, current_state, criterion, log=None):
    loss_idx, _ = min(enumerate([action_loss_for_shortest_path(repackage(logits), a, current_state, criterion) for a in targets]))
    final_action = targets[loss_idx]
    return final_action, action_loss_for_shortest_path(logits, final_action, current_state, criterion, log=log)