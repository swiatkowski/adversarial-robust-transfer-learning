"""
Base utils file for the Colab:
https://colab.research.google.com/drive/14O1yyQaT1GEFx2BWj-BAkHLTuuJnC8AR?usp=sharing

Originally from:
https://gitlab.com/msatkiewicz/robustness_exercises.git
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from robustness.tools import helpers
from robustness.datasets import ImageNet, RestrictedImageNet  # , DATASETS
from robustness.tools.label_maps import CLASS_DICT
from robustness.attacker import AttackerModel
from robustness import model_utils, datasets
from tqdm import tqdm, trange

####
## DATA
####


# Data Augmentation defaults
TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
    transforms.RandomCrop(size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(.25, .25, .25),
    transforms.RandomRotation(2),
    transforms.ToTensor(),
])
"""
Generic training data transform, given image side length does random cropping,
flipping, color jitter, and rotation. Called as, for example,
:meth:`robustness.data_augmentation.TRAIN_TRANSFORMS_DEFAULT(32)` for CIFAR-10.
"""

TEST_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor()
])
"""
Generic test data transform (no augmentation) to complement
:meth:`robustness.data_augmentation.TEST_TRANSFORMS_DEFAULT`, takes in an image
side length.
"""

# for transfer of ImageNet models
TRAIN_TRANSFORMS_TRANSFER = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

TEST_TRANSFORMS_TRANSFER = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

CIFAR_LABEL_MAP = {0: 'airplane', 1: 'automobile',
                   2: 'bird', 3: 'cat', 4: 'deer',
                   5: 'dog', 6: 'frog', 7: 'horse',
                   8: 'ship', 9: 'truck'}

CIFAR_NORMALIZATION = helpers.InputNormalize(
    torch.tensor([0.4914, 0.4822, 0.4465]).cuda(),  # mean
    torch.tensor([0.2023, 0.1994, 0.2010]).cuda()  # std
)


class Subset(torch.utils.data.Dataset):

    def __init__(self, dataset, targets, per_target, random=False):
        """
            Picks subset of the dataset.
        Arguments
            random: if False, always picks the same indices for given parameters.
            targets are converted to labels: [3,5,1] -> [0,1,2]
        """
        self.dataset = dataset
        self.indices = []
        self.target_to_label = {t: i for i, t in enumerate(targets)}

        iterable = range(len(dataset))
        if random:
            iterable = torch.utils.data.SubsetRandomSampler(iterable)

        num_per_target = {t: 0 for t in targets}
        for i in iterable:
            _, target = dataset[i]
            if target in targets and num_per_target[target] < per_target:
                num_per_target[target] += 1
                self.indices.append(i)
            if len(self.indices) == len(targets) * per_target:
                break

    def __getitem__(self, idx):
        data, target = self.dataset[self.indices[idx]]
        return data, self.target_to_label[target]

    def __len__(self):
        return len(self.indices)


def get_custom_imagenet(restricted=False, data_path='./data', data_aug=False, shuffle_val=False, batch_size=20):
    """
        We use helpers from robustness library
    Returns:
        loader: dataset loader
        norm: normalization function for dataset
        label_map: label map (class numbers to names) for dataset
    """
    if restricted:
        ds = RestrictedImageNet(data_path)
        label_map = CLASS_DICT['RestrictedImageNet']
    else:
        ds = ImageNet(data_path)
        label_map = CLASS_DICT['ImageNet']
        label_map = {k: v.split(',')[0] for k, v in label_map.items()}

    normalization = helpers.InputNormalize(ds.mean.cuda(), ds.std.cuda())
    loaders = ds.make_loaders(1, batch_size=batch_size, data_aug=data_aug, shuffle_val=shuffle_val)

    return loaders, normalization, label_map


def get_cifar(root='./data', data_aug=False, transfer=False):
    test_transform = TEST_TRANSFORMS_TRANSFER if transfer else TEST_TRANSFORMS_DEFAULT(32)
    if data_aug:
        train_transform = TRAIN_TRANSFORMS_TRANSFER if transfer else TRAIN_TRANSFORMS_DEFAULT(32)
    else:
        train_transform = test_transform

    train = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform)
    test = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform)

    return train, test


def get_binary_dataset(root='./data', batch_size=128, data_aug=False, targets=[0, 1], per_target=5000, random=False,
                       transfer=False):
    train_full, test_full = get_cifar(root=root, data_aug=data_aug, transfer=transfer)

    train = Subset(dataset=train_full, targets=targets, per_target=per_target, random=random)
    test = Subset(dataset=test_full, targets=targets, per_target=1000, random=False)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    label_map = {i: CIFAR_LABEL_MAP[t] for i, t in enumerate(targets)}

    return (train_loader, test_loader), CIFAR_NORMALIZATION, label_map


####
## MODELS
####


# !mkdir pretrained-models
# !wget -O pretrained-models/resnet-18-l2-eps0.ckpt "https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps0.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D"
# !wget -O pretrained-models/resnet-18-l2-eps3.ckpt "https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet18_l2_eps3.ckpt?sv=2019-10-10&ss=b&srt=sco&sp=rlx&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D"
def load_imagenet_model_from_checkpoint(model, checkpoint):
    """
        Loads pretrained ImageNet models from the https://github.com/Microsoft/robust-models-transfer
    """
    # Makes us able to load models saved with legacy versions
    state_dict_key = 'model'
    if not ('model' in checkpoint):
        state_dict_key = 'state_dict'

    sd = checkpoint[state_dict_key]
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model = AttackerModel(model, ImageNet(''))
    model.load_state_dict(sd)

    while hasattr(model, 'model'):
        model = model.model

    return model.cuda()


# model = torchvision.models.resnet18(pretrained=True).cuda()
# model.eval()
def load_restricted_imagenet_model():
    model_kwargs = {
        'arch': 'resnet50',
        'dataset': RestrictedImageNet('./data'),
        'resume_path': f'./models/RestrictedImageNet.pt'
    }

    model, _ = model_utils.make_and_restore_model(**model_kwargs)

    try:
        model = model.module.model
    except:
        model = model.model

    return model


####
## TRAINING
####


def forward_pass(mod, im, normalization=None):
    '''
    Compute model output (logits) for a batch of inputs.
    Args:
        mod: model
        im (tensor): batch of images
        normalization (function): normalization function to be applied on inputs
    Returns:
        logits: logits of model for given inputs
    '''
    if normalization is not None:
        im_norm = normalization(im.cuda())
    else:
        im_norm = im
    logits = mod(im_norm.cuda())
    return logits


def get_gradient(mod, im, targ, normalization, custom_loss=None):
    '''
    Compute model gradients w.r.t. inputs.
    Args:
        mod: model
        im (tensor): batch of images
        normalization (function): normalization function to be applied on inputs
        custom_loss (function): custom loss function to employ (optional)

    Returns:
        grad: model gradients w.r.t. inputs
        loss: model loss evaluated at inputs
    '''

    def compute_loss(inp, target, normalization):
        if custom_loss is None:
            output = forward_pass(mod, inp, normalization)
            return torch.nn.CrossEntropyLoss()(output, target.cuda())
        else:
            return custom_loss(mod, inp, target.cuda(), normalization)

    x = im.clone().detach().requires_grad_(True)
    loss = compute_loss(x, targ, normalization)
    grad, = torch.autograd.grad(loss, [x])
    return grad.clone(), loss.detach().item()


def visualize_gradient(t):
    '''
    Visualize gradients of model. To transform gradient to image range [0, 1], we
    subtract the mean, divide by 3 standard deviations, and then clip.

    Args:
        t (tensor): input tensor (usually gradients)
    '''
    mt = torch.mean(t, dim=[2, 3], keepdim=True).expand_as(t)
    st = torch.std(t, dim=[2, 3], keepdim=True).expand_as(t)
    return torch.clamp((t - mt) / (3 * st) + 0.5, 0, 1)


def L2PGD(mod, im, targ, normalization, step_size, Nsteps,
          eps=None, targeted=True, custom_loss=None, use_tqdm=False):
    '''
    Compute L2 adversarial examples for given model.
    Args:
        mod: model
        im (tensor): batch of images
        targ (tensor): batch of labels
        normalization (function): normalization function to be applied on inputs
        step_size (float): optimization step size
        Nsteps (int): number of optimization steps
        eps (float): radius of L2 ball
        targeted (bool): True if we want to maximize loss, else False
        custom_loss (function): custom loss function to employ (optional)

    Returns:
        x: batch of adversarial examples for input images
    '''
    assert targ is not None
    prev_training = bool(mod.training)
    mod.eval()

    sign = -1 if targeted else 1

    it = iter(range(Nsteps))
    if use_tqdm:
        it = tqdm(iter(range(Nsteps)), total=Nsteps)

    x = im.detach()
    im_dims = len(x.shape) - 1  # how many dimensions are there for single image

    for i in it:
        x = x.clone().detach().requires_grad_(True)
        g, loss = get_gradient(mod, x, targ, normalization,
                               custom_loss=custom_loss)

        if use_tqdm:
            it.set_description(f'Loss: {loss}')

        with torch.no_grad():

            # Compute gradient step
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1] * im_dims))
            scaled_g = g / (g_norm + 1e-10)
            # x += ... TODO fill this line
            x += sign * scaled_g * step_size

            # Project back to L2 eps ball
            if eps is not None:
                diff = x - im
                diff = diff.renorm(p=2, dim=0, maxnorm=eps)
                x = im + diff
            x = torch.clamp(x, 0, 1)

    if prev_training:
        mod.train()

    return x


def get_features(mod, im, normalization):
    '''
    Get feature representation of model  (output of layer before final linear
    classifier) for given inputs.

    Args:
        mod: model
        im (tensor): batch of images
        targ (tensor): batch of labels
        normalization (function): normalization function to be applied on inputs

    Returns:
        features: batch of features for input images
    '''
    feature_rep = torch.nn.Sequential(*list(mod.children())[:-1])
    im_norm = normalization(im.cuda())
    features = feature_rep(im_norm)[:, :, 0, 0]
    return features


class Linear(nn.Module):
    '''
        Class for linear classifiers.
    '''

    def __init__(self, Nfeatures, Nclasses):
        '''
        Initializes the linear classifier.
        Args:
            Nfeatures (int): Input dimension
            Nclasses (int): Number of classes
        '''
        super(Linear, self).__init__()
        self.fc = nn.Linear(Nfeatures, Nclasses)

    def forward(self, im):
        '''
        Perform a forward pass through the linear classifier.
        Args:
            im (tensor): batch of images

        Returns:
            logits (tensor): batch of logits
        '''
        imr = im.view(im.shape[0], -1)
        logits = self.fc(imr)
        return logits


def get_predictions(mod, im, normalization=None):
    '''
    Determine predictions of linear classifier.
    Args:
        im (tensor): batch of images
        mod: model

    Returns:
        pred (tensor): batch of predicted labels
    '''
    with torch.no_grad():
        logits = forward_pass(mod, im, normalization)
        pred = logits.argmax(dim=1)
    return pred


def accuracy(net, im, targ, normalization=None):
    '''
    Evaluate the accuracy of a given linear classifier.
    Args:
        net: model
        im (tensor): batch of images
        targ (tensor): batch of labels

    Returns:
        x: batch of adversarial examples for input images
    '''
    pred = get_predictions(net, im, normalization)
    acc = (pred == targ.cuda()).sum().item() / len(im) * 100
    return acc