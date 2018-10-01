from torch.autograd import Variable
import numpy as np


def make_pred_multilabel(dataloader, model):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        dataloader: torch dataloader
        model: densenet-121 from torchvision previously fine tuned to training data
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    preds = []
    obs = []

    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, labels, _ = data
        obs.append(labels.cpu().numpy())
        inputs = Variable(inputs.cuda())

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()
        pred = np.argmax(probs, axis=1)
        preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    obs = np.concatenate(obs)
    return preds, obs