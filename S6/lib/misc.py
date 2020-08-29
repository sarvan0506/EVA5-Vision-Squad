import matplotlib.pyplot as plt

def model_start(bn, l1, l2, epochs):
    
    losses = []
    accuracies = []
    incorrect_samples = []
    
    model = Net(bn).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=l2)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.10)
    epochs = epochs

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}:')
        train(model, device, train_loader, optimizer, epoch, l1)
        scheduler.step()
        val(model, device, val_loader, losses, accuracies, incorrect_samples)
    
    return losses, accuracies, incorrect_samples

def plot(plain, obs, metric):
    # Initialize a figure
    fig = plt.figure(figsize=(13, 11))

    # Plot values
    plain_plt, = plt.plot(plain)
    l1_plt, = plt.plot(l1)
    l2_plt, = plt.plot(l2)
    l1_l2_plt, = plt.plot(l1_l2)

    # Set plot title
    plt.title(f'Validation {metric}')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    # Set legend
    location = 'upper' if metric == 'Loss' else 'lower'
    plt.legend(
        (plain_plt, l1_plt, l2_plt, l1_l2_plt),
        ('Plain', 'L1', 'L2', 'L1 + L2'),
        loc=f'{location} right',
        shadow=True,
        prop={'size': 20}
    )

    # Save plot
    fig.savefig(f'{metric.lower()}_change.png')