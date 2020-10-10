from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR
import matplotlib.pyplot as plt


def step_lr(optimizer, step_size, gamma=0.1, last_epoch=-1):
    """Create LR step scheduler.

    Args:
        optimizer (torch.optim): Model optimizer.
        step_size (int): Frequency for changing learning rate.
        gamma (float): Factor for changing learning rate. (default: 0.1)
        last_epoch (int): The index of last epoch. (default: -1)
    
    Returns:
        StepLR: Learning rate scheduler.
    """

    return StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)


def reduce_lr_on_plateau(optimizer, factor=0.1, patience=10, verbose=False, min_lr=0):
    """Create LR plateau reduction scheduler.

    Args:
        optimizer (torch.optim): Model optimizer.
        factor (float, optional): Factor by which the learning rate will be reduced.
            (default: 0.1)
        patience (int, optional): Number of epoch with no improvement after which learning
            rate will be will be reduced. (default: 10)
        verbose (bool, optional): If True, prints a message to stdout for each update.
            (default: False)
        min_lr (float, optional): A scalar or a list of scalars. A lower bound on the
            learning rate of all param groups or each group respectively. (default: 0)
    
    Returns:
        ReduceLROnPlateau instance.
    """

    return ReduceLROnPlateau(
        optimizer, factor=factor, patience=patience, verbose=verbose, min_lr=min_lr
    )


class OneCyclePolicy(OneCycleLR):
    
    """Create One Cycle Policy for Learning Rate.

    Args:
        optimizer (torch.optim): Model optimizer.
        max_lr (float): Upper learning rate boundary in the cycle.
        epochs (int): The number of epochs to train for. This is used along with
            steps_per_epoch in order to infer the total number of steps in the cycle.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the cycle.
        pct_start (float, optional): The percentage of the cycle (in number of steps)
            spent increasing the learning rate. (default: 0.5)
        div_factor (float, optional): Determines the initial learning rate via
            initial_lr = max_lr / div_factor. (default: 10.0)
        final_div_factor (float, optional): Determines the minimum learning rate via
            min_lr = initial_lr / final_div_factor. (default: 1e4)
    
    Returns:
        OneCycleLR instance.
    """
    
    def __init__(self, **kwargs):
        
        """
        Passes arguments to the super class constructor, **kwargs passes
        dynamic parameters captured at runtime.
        """
        
        self.epochs = kwargs["epochs"]
        self.steps_per_epoch = kwargs["steps_per_epoch"]
        
        super().__init__(**kwargs)
        
    
    def plot_policy(self):
        
        """
        Plots the Learning Rates Policy
        """
        
        ys = []
        for _ in range(self.epochs):
            for _ in range(self.steps_per_epoch):
                ys.append(self.optimizer.param_groups[0]['lr'])
                self.step()
        plt.figure(figsize=(10,8))
        plt.title('OneCycleLR schedule')
        plt.ylabel('Learning rate')
        plt.xlabel('Step')
        plt.plot(ys, c='red')
        plt.show()
        plt.savefig('ocp_plot.png')