import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Dataset settings
    model_type: str = 'action'
    train_size_perc: float = 0.8
    test_size_perc: float = 0.2

    dataset_src: str = './fn-content-dataset/compiled'
    label_dataframe: str = f"{dataset_src}/{model_type}_labels.csv"
    models_dir: str = f'./training/models'

    # Training settings
    seed: int = 42
    batch_size: int = 32
    num_workers: int = os.cpu_count() // 2
    img_size: int = 384
    model_name: str = 'convnextv2_tiny'

    # Finetuning settings
    weight_decay: float = 1e-4
    learning_rate: float = 1.25E-05
    epochs: int = 15

    # Class imbalance handling
    bce_power: float = 0.5
    tau_logit_adjust: float = 0.5

    # Weights & Biases settings
    use_wandb: bool = True
    wandb_project: str = 'fn-content-classifier'
    wandb_entity: str = 'miloszbertman'  # Set to your wandb username/team
    wandb_run_name: str = None  # Will be auto-generated if None
    wandb_tags: list = None

    # Model settings
    reduce_metric: str = 'pr_auc_macro'
    early_stop_metric: str = 'pr_auc_macro'
    best_model_metric: str = "pr_auc_macro"

    cosine_annealing_min: float = 0.1

    pretrained: bool = True

    def __post_init__(self):
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

    @property
    def model_catalog(self) -> dict[str, int]:
        return {
            'convnextv2_nano': self.img_size,
            'convnextv2_tiny': self.img_size,
            'convnextv2_base': self.img_size,
            'convnextv2_large': self.img_size,
            'convnextv2_huge': self.img_size,
        }


    def info(self) -> None:
        print("Configuration:")
        for field in self.__dataclass_fields__:
            print(f"  {field}: {getattr(self, field)}")

    def wandb_config(self):
        if not self.use_wandb:
            print("Wandb is disabled.")
            return

        print("Wandb Configuration:")
        print(f"  Enabled: {self.use_wandb}")
        print(f"  Project: {self.wandb_project}")
        print(f"  Entity: {self.wandb_entity}")
        print(f"  Tags: {self.wandb_tags}")