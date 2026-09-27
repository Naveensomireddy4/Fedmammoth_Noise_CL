from utils.tools import str_to_bool

LOG_LOSS_INTERVAL = 10
DATASET_PATH = "./data"

# TRAINING CONFIG TEMPLATE
ADDITIONAL_ARGS = {
    "num_epochs": (int, 5),
    "num_comm_rounds": (int, 5),
    "num_clients": (int, 10),
    "checkpoint": (str, ""),
    "checkpoint_interval": (int, float("inf")),
    "device": (str, "cuda:0"),
    "output_folders_root": (str, "./output"),
    "wandb": (str_to_bool, True),
    "wandb_project": (str, "FCL"),
    "wandb_entity": (str, ""),
    "debug_mode": (str_to_bool, False),
    "precision": (str, "16-mixed"),
    "participation_rate": (float, 1.0),
    "test_local": (str_to_bool, False),
    "test_local_transfer": (str_to_bool, False),
    "validation_interval": (int, 0),  # 0 means no validation
    "save_models": (str_to_bool, False),
    "random_seed": (int, 42),
    "train_transform": (str, "default_train"),
    "test_transform": (str, "default_test"),
    "snr": (int, 0),
    "memory_size": (int, 500),

    # Memory management args
    "update_strategy": (str, "balanced"),
    "sampling_strategy": (str, "random"),
    "balanced_update": (str, "uncertainty"),
    "uncertainty_score": (str, "bregman"),
    "subsample_size": (int, 50),
    "balanced_step": (str, "bottomk"),
    "step_str": (str, "bottomk")
}
