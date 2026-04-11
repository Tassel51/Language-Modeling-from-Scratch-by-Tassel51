import wandb

with wandb.init(project="wandb_test_minimal") as run:
    run.log({"x": 1})
