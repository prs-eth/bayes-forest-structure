import sys

import torch
from blowtorch import Run
from blowtorch.loggers import WandbLogger

from data import ForestData
from models import ResNext
from losses import negative_log_likelihood
from utils import nanmean, limit

run = Run(config_files=[sys.argv[1]])
run.set_deterministic(run['training.deterministic'])
run.seed_all(run['training.random_seed'])

num_orbit_directions = 2 if run['data.both_orbit_directions'] else 1
num_s1_channels = len(run['data.s1_image_bands']) * num_orbit_directions
in_channels = len(run['data.s2_image_bands']) + num_s1_channels
out_channels = len(run['data.labels_bands'])
model_type = run['model'].pop('type')
if model_type == 'resnext':
    model = ResNext(in_channels, out_channels, num_s1_channels=num_s1_channels, **run['model'])
else:
    raise NotImplementedError(model_type)

data = ForestData(**run['data'])

labels_names = [run['logging.labels_names'][i - 1] for i in run['data.labels_bands']]


@run.train_step
@run.validate_step
def step(batch, model):
    x, y = batch
    mu, log_var = model(x)
    log_var = limit(log_var)

    if run['training'].get('activate_mean', False):
        assert not run['data.normalize_labels']
        # p95, mean_height
        mu[:, [0, 1]] = torch.exp(limit(mu[:, [0, 1]]))
        # density, gini, cover
        mu[:, [2, 3, 4]] = torch.sigmoid(mu[:, [2, 3, 4]])

    # these ground truth locations are invalid and should not be considered for the loss calculation
    mask = torch.isnan(y)

    nll = nanmean(negative_log_likelihood(mu, log_var, y), mask, dim=(0, 2, 3))

    error = (mu - y).detach()
    mae = nanmean(error.abs(), mask, dim=(0, 2, 3))
    mse = nanmean(error ** 2, mask, dim=(0, 2, 3))
    me = nanmean(error, mask, dim=(0, 2, 3))

    log_var_mean = nanmean(log_var, mask, dim=(0, 2, 3)).detach()

    return {
        'loss': nll.mean(),
        **{'loss_' + m: nll[i] for i, m in enumerate(labels_names)},
        **{'mse_' + m: mse[i] for i, m in enumerate(labels_names)},
        **{'mae_' + m: mae[i] for i, m in enumerate(labels_names)},
        **{'me_' + m: me[i] for i, m in enumerate(labels_names)},
        **{'log_var_' + m: log_var_mean[i] for i, m in enumerate(labels_names)}
    }


@run.configure_optimizers
def configure_optimizers(model):
    optim = torch.optim.Adam(model.parameters(), lr=run['training.lr'], weight_decay=run['training.weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, **run['training.scheduler'])
    return {'optimizers': optim, 'schedulers': scheduler}


loggers = WandbLogger(project='forest_structure') if run['training.use_wandb_logger'] else None

run(
    model,
    data.train_loader,
    data.val_loader,
    loggers=loggers,
    optimize_first=False,
    resume_checkpoint=run['training.resume_checkpoint'],
    max_epochs=run['training.epochs']
)
