import json

import wandb
import yaml
from wandb.apis.public import Sweep
from .configs import TrainConfig
from .train import train_main


def load_sweep_config() -> dict:
    with open("sweep_action.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)
    return sweep_config


def get_sweep(sweep_config: dict) -> tuple[Sweep, str]:
    api = wandb.Api()
    best_metric_name = sweep_config.get("metric").get("name")
    wandb_entity = sweep_config["wandb_entity"]
    wandb_project = sweep_config["wandb_project"]
    sweep_id = sweep_config["sweep_id"]
    sweep = api.sweep(
        f"{wandb_entity}/{wandb_project}/{sweep_id}"
    )

    return sweep, best_metric_name


def extract_train_config(sweep: Sweep, best_metric_name: str) -> TrainConfig:
    best_run = sweep.best_run(order=best_metric_name)
    best_parameters = best_run.config
    best_dict: dict = json.loads(best_parameters)
    best_dict.pop('_wandb')

    config_params = set(TrainConfig().valid_params())
    same_keys = set(best_dict.keys()).intersection(config_params)

    best_dict = {k: v for k, v in best_dict.items() if k in same_keys}

    return TrainConfig(**best_dict)

if __name__ == "__main__":
    sweep_config = load_sweep_config()
    sweep, best_metric_name = get_sweep(sweep_config)
    best_train_config = extract_train_config(sweep, best_metric_name)
    train_main(best_train_config)

