
import paddle.fluid as fluid


eps = 1e-8


def Basic_Loss(preds, labels, ignore_index=255):
    n, c, h, w = preds.shape

    preds = fluid.layers.transpose(preds, [0, 2, 3, 1])
    
    mask = labels!=ignore_index
    mask = fluid.layers.cast(mask, 'float32')

    loss = fluid.layers.softmax_with_cross_entropy(preds, labels)
    loss = loss * mask
    avg_loss = fluid.layers.mean(loss) / (fluid.layers.mean(mask) + eps)

    return avg_loss


