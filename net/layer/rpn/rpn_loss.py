from net.utils.func_utils import weighted_focal_loss_for_cross_entropy, weighted_smooth_l1

def rpn_cls_loss(logits, labels, label_weights):
    """
    :param logits: (B, N, 2),    unnormalized foreground/background score
    :param labels: (B, N)        {0, 1} for bg/fg
    :param label_weights: (B, N) float \in (0,1] for rareness, otherwise 0 (don't care)
    :return: float
    """
    batch_size, num_anchors, num_classes = logits.size()
    labels = labels.long()
    batch_num_anchors = batch_size * num_anchors

    # classification ---
    logits = logits.view(batch_num_anchors, num_classes)
    labels = labels.view(batch_num_anchors, 1)
    label_weights = label_weights.view(batch_num_anchors, 1)

    return weighted_focal_loss_for_cross_entropy(logits, labels, label_weights)


def rpn_reg_loss(labels, deltas, target_deltas, target_weights, delta_sigma=3.0):
    """
    :param labels: (B, N) {0, 1} for bg/fg. used to catch positive samples
    :param deltas: (B, N, 2, 4) bbox regression
    :param target_deltas: (B, N, 4) target deltas from make_rpn_target
    :param target_weights: (B, N) float \in (0,1] for rareness, otherwise 0 (don't care)
    :param delta_sigma: float
    :return: float
    """
    batch_size, num_anchors, num_classes, num_deltas = deltas.size()
    assert num_deltas == 4
    labels = labels.long()
    batch_num_anchors = batch_size * num_anchors

    # one-hot encode
    labels = labels.view(batch_num_anchors, 1)
    deltas = deltas.view(batch_num_anchors, num_classes, 4)
    target_deltas = target_deltas.view(batch_num_anchors, 4)
    target_weights = target_weights.view(batch_num_anchors, 1)
    # calc positive samples only
    index = (labels != 0).nonzero()[:, 0]
    deltas = deltas[index]
    target_deltas = target_deltas[index]
    target_weights = target_weights[index].expand((-1, 4)).contiguous()

    select = labels[index].view(-1, 1).expand((-1, 4)).contiguous().view(-1, 1, 4)
    deltas = deltas.gather(1, select).view(-1, 4)

    return weighted_smooth_l1(deltas, target_deltas, target_weights, delta_sigma)