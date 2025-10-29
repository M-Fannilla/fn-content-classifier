import argparse
import json

import wandb
import yaml
from wandb.apis.public import Sweep
from .configs import TrainConfig
from .train import train_main


def load_sweep_config(model_type: str) -> dict:
    with open(f"sweep_{model_type}.yaml", "r") as f:
        return yaml.safe_load(f)


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
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model_type",
        type=str,
        default="action",
        help="Type of model to sweep (e.g., 'action', 'bodyparts')",
    )

    parsed_args = args.parse_args()
    model_to_sweep = parsed_args.model_type

    config = load_sweep_config(model_type=model_to_sweep)
    sweep_obj, metric_name = get_sweep(config)
    best_train_config = extract_train_config(sweep_obj, metric_name)
    best_train_config.info()
    train_main(best_train_config)

