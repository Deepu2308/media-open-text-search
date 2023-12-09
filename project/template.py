import torch
from torch import nn
import numpy as np


class NGram(nn.Module):
    """
        A PyTorch module for an n-gram neural network with multiple layers and hyperparameters.

    Args:
        vocab_size (int): number of words in the vocabulary
        ngram (int): n in n-gram, i.e., the number of context words to consider
        emb_dim (int): embedding dimension; default is 2
        size (int): hidden unit dimension; default is 10
        nlayers (int): number of layers in the model; default is 4
        activation_function (str): activation function to use; must be 'ReLu' or 'Tanh'; default is 'Tanh'

    Attributes:
        embedding_module (nn.ModuleDict): dictionary containing the embedding module
        linear_module (nn.ModuleDict): dictionary containing the linear transformation modules

    Methods:
        forward(x): computes forward pass with given input x

    Visualize:
        Use plot_utils.py
        plot_activation_distribution(model,keyword='Linear')
        plot_weight_distribution(model,keyword='Linear')
        plot_grad_distribution(model,keyword='Linear')
    """

    def __init__(
        self,
        vocab_size,
        ngram,
        emb_dim=2,
        size=10,
        nlayers=4,
        activation_function="Tanh",
    ):
        super(NGram, self).__init__()

        assert size >= 2, "Pick higher hidden unit dimension"
        assert nlayers >= 2, "Pick higher nlayers"
        assert activation_function in ["ReLu", "Tanh"], "Choose from ReLu or Tanh"

        # compute total dim
        self.ngram = ngram
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.embed_total_dim = self.ngram * self.emb_dim

        # save hyperparameters
        self.size = size
        self.nlayers = nlayers
        self.activation_function = activation_function

        # create embedding module
        self.embedding_module = nn.ModuleDict(
            {"Embedding": nn.Embedding(vocab_size, self.emb_dim)}
        )

        # initialize embedding
        _ = nn.init.normal_(self.embedding_module["Embedding"].weight, mean=0, std=1)

        # add first linear module
        self.linear_module = nn.ModuleDict(
            {
                "LinearInput": nn.Linear(
                    in_features=self.embed_total_dim, out_features=self.size
                ),
                f"{self.activation_function}Input": nn.ReLU()
                if self.activation_function.lower() == "relu"
                else nn.Tanh(),
            }
        )

        # add intermediate linear modules
        for i in range(1, self.nlayers):
            # add linear
            self.linear_module.update(
                nn.ModuleDict(
                    {
                        f"Linear{i}": nn.Linear(
                            in_features=self.size, out_features=self.size
                        )
                    }
                )
            )
            # add non linear
            self.linear_module.update(
                nn.ModuleDict(
                    {
                        f"{self.activation_function}{i}": nn.ReLU()
                        if self.activation_function.lower() == "relu"
                        else nn.Tanh()
                    }
                )
            )

        # add output linear module
        self.linear_module.update(
            nn.ModuleDict(
                {
                    "LinearOutput": nn.Linear(
                        in_features=self.size, out_features=self.vocab_size
                    )
                }
            )
        )

        # initialize first linear module
        _ = nn.init.normal_(
            self.linear_module["LinearInput"].weight,
            mean=0,
            std=1 / np.sqrt(self.embed_total_dim),
        )

        # initialize remaining linear modules
        gain = nn.init.calculate_gain(self.activation_function.lower())
        recommended_scale = gain / np.sqrt(self.size)
        for name, layer in self.linear_module.items():
            if name.startswith("Linear") and name != "LinearInput":
                _ = nn.init.normal_(layer.weight, mean=0, std=recommended_scale)
                _ = nn.init.normal_(layer.bias, mean=0, std=recommended_scale)

        self.activations = {}

        for name, layer in self.embedding_module.items():
            layer.register_forward_hook(self.forward_hook(name))

        for name, layer in self.linear_module.items():
            layer.register_forward_hook(self.forward_hook(name))

    def forward_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()

        return hook

    def forward(self, x):
        for _, layer in self.embedding_module.items():
            x = layer(x).view(-1, self.embed_total_dim)

        for _, layer in self.linear_module.items():
            x = layer(x)

        return x
