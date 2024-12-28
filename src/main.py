import wandb
import torch 
import torch.nn as nn
import torch.optim as optim
from config import cfg
from modules import ReducedTransformer
from trainer import train, test
from utils import get_loaders, count_parameters


def main():
    # Loading Data
    train_loader, test_loader = get_loaders(cfg.data_name)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_dim = cfg.embed_dim
    num_layers = cfg.num_layers
    image_size = cfg.image_size[cfg.data_name]  # the edge size of the iamge
    patch_size = cfg.patch_size[cfg.data_name]
    dropout = cfg.dropout
    in_channels = cfg.channels[cfg.data_name]
    n_classes = cfg.n_classes[cfg.data_name]
    learning_rate = cfg.lr
    num_epochs = cfg.epochs
    
    # Experiments 
    hyperparameter_combinations = cfg.hyperparameter_combinations
    
    # Data preparation
    criterion = nn.CrossEntropyLoss()
    K = len(train_loader.dataset)  # Number of training examples
    M = n_classes # Number of classes
    
    # Training multiple experiments
    for idx, (num_heads, include_mlp, modification) in enumerate(hyperparameter_combinations):
        # Initialize model to calculate P (number of parameters)
        model = ReducedTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            n_classes=M,
            patch_size=patch_size,
            num_patches=(image_size // patch_size) ** 2,
            dropout=dropout,
            in_channels=in_channels,
            include_mlp=include_mlp,
            modification = modification
        ).to(device)
        P = count_parameters(model)  # Number of trainable parameters
        Q = K * M / P  # Calculate Q
    
        # Initialize Weights & Biases for each run 
        wandb.init(
            project=cfg.wandb_project_name,
            name=f"Exper-{idx + 1}",
            config={
                "num_heads": num_heads,
                "MLP": include_mlp,
                "Modification":modification,
                "Parameters": P,
                "Q": Q,
            }
        )
        config = wandb.config
    
        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
        # Train and test
        for epoch in range(num_epochs):
            print(f"Experiment {idx + 1}, Epoch {epoch + 1}/{num_epochs}")
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = test(model, test_loader, criterion, device)
    
            # Log results to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            })
    
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\n")
    
        # Log final results
        wandb.log({
            "final_train_accuracy": train_acc,
            "final_test_accuracy": test_acc,
        })
    
        # Finish the wandb run
        wandb.finish()


if __name__ == "__main__":
    main()
