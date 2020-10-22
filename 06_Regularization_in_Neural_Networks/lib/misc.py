import matplotlib.pyplot as plt
import lib.model as m
import torch
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import lib.train as tr
import lib.test as ts

def model_start(train_loader, test_loader, bn, l1, l2, dropout, epochs):
    
    losses = []
    accuracies = []
    incorrect_samples = []
    
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = m.Net(bn, dropout).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=l2)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.10)
    epochs = epochs

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}:')
        tr.train(model, device, train_loader, optimizer, epoch, l1)
        scheduler.step()
        ts.test(model, device, test_loader, losses, accuracies, incorrect_samples)
    
    return losses, accuracies, incorrect_samples

def plot(obs, metric):
    # Initialize a figure
    fig = plt.figure(figsize=(13, 11))
    
    if metric == "loss":
        i = 0
    else:
        i = 1
    p1, = plt.plot(obs[list(obs.keys())[0]][i])
    p2, = plt.plot(obs[list(obs.keys())[1]][i])
    p3, = plt.plot(obs[list(obs.keys())[2]][i])
    p4, = plt.plot(obs[list(obs.keys())[3]][i])
    p5, = plt.plot(obs[list(obs.keys())[4]][i])
    
    
    # Set plot title
    plt.title(f'Validation {metric}')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    # Set legend
    location = 'upper' if metric == 'loss' else 'lower'
    plt.legend(
        (p1, p2, p3, p4, p5),
        tuple(obs.keys()),
        loc=f'{location} right',
        shadow=True,
        prop={'size': 20}
    )

    # Save plot
    fig.savefig(f'{metric.lower()}_change.png')


def missclassified(results, exp):
    # Initialize plot
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.tight_layout()
    
    data = results[exp][2]
    
    exp = "with_" + exp.split("+ ")[-1]

    for idx, result in enumerate(data):

        # If 25 samples have been stored, break out of loop
        if idx > 24:
            break
        
        label = result['label'].item()
        prediction = result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        axs[row_count][idx % 5].set_title(f'Label: {label}\nPrediction: {prediction}')
        axs[row_count][idx % 5].imshow(result['image'][0], cmap='gray_r')

        # Save each image individually in labelled format
        extent = axs[row_count][idx % 5].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{exp}/{exp}_{idx + 1}.png', bbox_inches=extent.expanded(1.1, 1.5))
    
    # Save image
    fig.savefig(f'{exp}/{exp}_missclassified.png', bbox_inches='tight')