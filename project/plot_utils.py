import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_activation_distribution(model, keyword):
    """
    Plots the activation distribution of layers that contains the keyword.

    Parameters:
    model (torch.nn.Module): The model instance.
    keyword (str): The keyword to select layers for which activations distribution will be plotted.
    """

    legends = []
    plt.figure(figsize=(10, 6))
    for key, values in model.activations.items():
        if key.startswith(keyword):
            if keyword.lower().startswith("relu"):
                dead_perc = " {:.2f}% dead".format(
                    100 * (values.sum(0) == 0).float().mean().item()
                )
            elif keyword.lower().startswith("tanh"):
                dead_perc = " {:.2f}% dead".format(
                    100 * (values.abs() > 0.99).float().mean().item()
                )
            else:
                dead_perc = ""

            mu, std = values.mean().item(), values.std().item()
            hy, hx = torch.histogram(values.ravel(), density=True)
            print(key, mu, std)
            plt.plot(hx[:-1], hy, alpha=0.8)
            legends.append("{} ({:.2f},{:.2f})".format(key, mu, std) + dead_perc)

    plt.legend(legends)
    plt.title(f"Activation Distribution {keyword} layers")
    plt.show()


def plot_weight_distribution(model, module, keyword):
    """
    Plots the weight distribution of layers that contains the keyword.

    Parameters:
    model (torch.nn.Module): The model instance.
    module (str): The name of the module where layers are contained.
    keyword (str): The keyword to select layers for which weight distribution will be plotted.
    """

    legends = []
    plt.figure(figsize=(10, 6))

    module = model.__getattr__(module)

    for key, values in module.items():
        if key.startswith(keyword):
            values = values.weight.data

            mu, std = values.mean().item(), values.std().item()
            hy, hx = torch.histogram(values.ravel(), density=True)
            print(key, mu, std)
            plt.plot(hx[:-1], hy, alpha=0.8)
            legends.append("{} ({:.2f},{:.2f})".format(key, mu, std))

    plt.legend(legends)
    plt.title("Weight Distribution")
    plt.show()


def plot_grad_distribution(model, module, keyword):
    """
    Plots the gradient distribution of layers that contains the keyword.
    Run forward on a batch or entire train set in model.train() mode before calling this function.

    Parameters:
    model (torch.nn.Module): The model instance.
    module (str): The name of the module where layers are contained.
    keyword (str): The keyword to select layers for which gradient distribution will be plotted.
    """

    legends = []
    plt.figure(figsize=(10, 6))

    module = model.__getattr__(module)

    for key, values in module.items():
        if key.startswith(keyword):
            values = values.weight.grad

            mu, std = values.mean().item(), values.std().item()
            hy, hx = torch.histogram(values.ravel(), density=True)
            print(key, mu, std)
            plt.plot(hx[:-1], hy, alpha=0.8)
            legends.append("{} ({:.2f},{:.2f})".format(key, mu, std))

    plt.legend(legends)
    plt.title("Gradient Distribution")
    plt.show()


def plot_character_distribution(xs, ys, itoc):
    # Create a figure and two subplots (side by side)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    # Create a histogram on the first subplot
    axs[0].hist(xs.ravel().detach().numpy(), bins=30, color="blue", alpha=0.7)
    axs[0].set_title("xs")

    axs[0].set_xticks(list(itoc.keys()))
    axs[0].set_xticklabels(list(itoc.values()))

    # Create a histogram on the second subplot
    axs[1].hist(ys.ravel().detach().numpy(), bins=30, color="red", alpha=0.7)
    axs[1].set_title("ys")

    axs[1].set_xticks(list(itoc.keys()))
    axs[1].set_xticklabels(list(itoc.values()))

    plt.show()


@torch.no_grad()
def plot_emb(embed):
    embs = embed(torch.arange(0, len(ctoi), 1)).detach().numpy()
    embx = embs[:, 0]
    emby = embs[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(embx, emby)

    for i, txt in enumerate(list(ctoi.keys())):
        ax.annotate(txt, (embx[i], emby[i]))

    plt.show()
