import os
from dataclasses import dataclass

@dataclass
class Config:
    # Dataset settings
    train_size_perc: float = 0.8
    test_size_perc: float = 0.2
    dataset_src: str = './fn-content-dataset/compiled'
    label_dataframe: str = './fn-content-dataset/compiled/action_labels.csv'
    output_dir: str = './outputs'
    
    # Training settings
    seed: int = 42
    batch_size: int = 32
    num_workers: int = os.cpu_count() // 2
    threshold: float = 0.5
    img_size: int = 224
    model_name: str = 'convnextv2_tiny'
    # Finetuning settings
    learning_rate: float = 1e-5
    epochs: int = 20
    
    # Early stopping settings
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Learning rate reduction on plateau
    lr_reduce_patience: int = 5
    lr_reduce_factor: float = 0.5
    lr_reduce_min_lr: float = 1e-7
    
    # Class imbalance handling
    use_class_weights: bool = True
    class_weight_method: str = 'inverse_freq'  # 'inverse_freq', 'balanced', 'sqrt_inverse_freq'
    loss_type: str = 'focal'  # 'focal', 'asymmetric', 'weighted_bce', 'bce'
    
    # Weights & Biases settings
    use_wandb: bool = False
    wandb_project: str = 'fn-content-classifier'
    wandb_entity: str = 'miloszbertman'  # Set to your wandb username/team
    wandb_run_name: str = None  # Will be auto-generated if None
    wandb_tags: list = None
    
    # Model catalog
    model_catalog: dict[str, int] = None

    def __post_init__(self):
        if self.model_catalog is None:
            self.model_catalog = {
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

