class cfg: 
    data_name = "CIFAR" # or "MNIST"
    wandb_project_name = "Test" # The name of the wandb project
    embed_dim = 64 
    num_layers = 6
    patch_size = {"MNIST":7, "CIFAR":8}
    image_size = {"MNIST":28, "CIFAR":32}
    channels = {"MNIST":1, "CIFAR":3}
    n_classes = {"MNIST":10, "CIFAR":10}
    dropout = 0.1
    lr = 0.001
    epochs = 100
    hyperparameter_combinations = [
                (1, True, "unchanged"),  # (n_heads, MLP or not, modification)
                (4, True, "unchanged"),
                (1, True, "Wqk+noWvWo"),
                (1, False, "unchanged"),
                (4, False, "unchanged"),
                (1, False, "symmetry"),
                (4, False, "symmetry"),
                (1, False, "Wqk"),
                (1, False, "Wqk+noWvWo"),
    ]
    


    