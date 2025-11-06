# PyTorch: The main library, similar to NumPy, but with built-in tools for building and training neural networks.
import torch

# torch.nn is the submodule containing all the building blocks for NNs like layers, activation functions, and loss functions
import torch.nn as nn

# Python library used to plot graphs
import matplotlib.pyplot as plt

import torch.optim as optim


# This is how we make a PyTorch class
# The .nn module is the python syntax to inherit all of the built in stuff from PyTorch's Module class
# This includes tracking weights and automatic backpropagation
class neuralNetwork(nn.Module):

    # Constructor method that is called automatically and initializes the network's weights and biases
    def __init__(self, input_size, hidden_size, output_size):

        # This is a mandatory first line that calls the init method of the nn.Module to set everything up
        # Should always be included
        super(neuralNetwork, self).__init__()

        # In PyTorch, this one line creates the weights and biases rather than us manually declaring it
        # nn.Linear represents one fully connected layer. It initializes the (input_size x hidden_size) weight matrix and the (1 x hidden_size) bias vector
        self.hidden_layer = nn.Linear(input_size, hidden_size)

        # This object simply utilizes the built in sigmoid equation for our activation function
        self.activation = nn.Sigmoid()

        # Creates a fully connected layer from the hidden layers to the output. Same way as the self.hidden_layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        # We pass the input "x" through our first connected layer
        # Without PyTorch, we would need to use the NumPy dot product but we can just call the initialized layer object from before as it does all of that for us
        # It does the exact same matrix math: (input * weights.T + bias)
        raw_hidden = self.hidden_layer(x)

        # Simply calls the activation object, sigmoid is calculated
        activated_hidden = self.activation(raw_hidden)

        # Does the same thing as raw_hidden but for the final output layer, taking in 4 inputs from the hidden layers and returning 1
        raw_output = self.output_layer(activated_hidden)
        
        # Applies activation function to the raw output
        output = self.activation(raw_output)

        return output
    
# This function runs the user's original code, modified to capture and plot the loss from a single run.
def run_and_plot_single_run():
    print("==============================================")
    print("  Generating Plot for a Single Training Run   ")
    print("==============================================")

    # This is our XOR input data as PyTorch tensors
    X = torch.tensor([
        [0.0,0.0],
        [0.0,1.0],
        [1.0,0.0],
        [1.0,1.0]
    ],dtype=torch.float32)

    # This our XOR target output data
    y = torch.tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ], dtype=torch.float32)

    epochs = 50000 # Number of training cycles
    hidden_size = 4 # Number of hidden loops
    rate  = 0.1

    # Creates an instance of our network
    model = neuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1)

    # PyTorch automatically calculates our error with the Mean Squared Error function (y - prediction = error)
    loss_func = nn.MSELoss()

    # Instead of manually updating the weights, we declare an optimizer object to do that
    optimizer = optim.SGD(model.parameters(), lr = rate)

    print("--- Starting Training (Single Run) ---")
    print(f"Model: {model}")
    print(f"Epochs: {epochs}, LR: {rate}, Hidden Size: {hidden_size}")

    # Lists to store data for plotting
    epoch_losses = []
    plot_indices = []

    for i in range(epochs):

        # We pass all 4 X inputs through the model
        predictions = model(X)

        # MSELoss compares our predictions to the correct y values
        loss = loss_func(predictions, y)

        # PyTorch automatically adds gradients. We manually set them to zero at the start of each loop
        optimizer.zero_grad()

        # PyTorch automatically backpropagates and calculates a gradient for every single weight and bias
        loss.backward()

        # This would be our "self.weights += ... " lines but we just call step to apply the gradients to the weights and biases using the learning rate
        optimizer.step()

        # --- MODIFICATION: Store data for plotting ---
        if (i + 1) % 1000 == 0:
            
            print(f"  Epoch {i + 1}/{epochs}, Loss: {loss.item():.6f}")
            
            # .item() is a PyTorch command to get the actual Python number out of a 0-dimensional tensor
            epoch_losses.append(loss.item())
            plot_indices.append(i + 1)
    
    print("--- Training Complete ---")

    print("\n--- Testing the Network After Training ---")

    # no_grad() tells PyTorch that we are not training
    with torch.no_grad():
        for input_data, target in zip(X, y):
            # Pass one input row at a time
            prediction = model(input_data)
            
            # .item() gets the single number (e.g., 0.987)
            pred_value = prediction.item()
            target_value = target.item()
            
            # Round the prediction to 0 or 1
            rounded = 1 if pred_value > 0.5 else 0
            
            print(f"Input: {input_data.tolist()}, Target: {int(target_value)}, Prediction: {pred_value:.4f}, Rounded: {rounded}")

    # Plotting code
    print("\n--- Generating Plot ---")
    plt.figure(figsize=(10, 6))
    plt.plot(plot_indices, epoch_losses)
    plt.title('Training Loss vs. Epoch (Single Run)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.grid(True)
    plt.savefig('single_run_loss.png')
    print("Saved 'single_run_loss.png'")
    plt.clf() # Clear the figure for the next plot

#This function runs a new experiment to compare different learning rates
def run_and_plot_comparison():
    print("\n==============================================")
    print("  Generating Plot to Compare Learning Rates   ")
    print("==============================================")

    # This is our XOR input data as PyTorch tensors
    X = torch.tensor([
        [0.0,0.0],
        [0.0,1.0],
        [1.0,0.0],
        [1.0,1.0]
    ],dtype=torch.float32)

    # This our XOR target output data
    y = torch.tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ], dtype=torch.float32)

    epochs = 20000 # Number of training cycles
    hidden_size = 4 # Number of hidden loops
    rates = [0.5, 0.1, 0.01]

    # We will log data every 'log_interval' epochs
    log_interval = 200 
    
    # Store results here all_losses[rate] = (epoch_list, loss_list)
    all_losses = {}

    print("--- Starting Training (Multiple Runs) ---")

    for i in rates:
        print(f"\n--- Training with LR = {i} ---")

        # We re-initialize the model and optimizer for each run
        model = neuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1)
        loss_func = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr = i)

        # Store losses for this specific run
        current = []
        epoch_log = []

        for j in range(epochs):
            predictions = model(X)
            loss = loss_func(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(j+1) % log_interval == 0:
                current.append(loss.item())
                epoch_log.append(j+1)
        
        # Stores our result in the dictionary
        all_losses[i] = (epoch_log, current)

        print(f"--- Finished training with LR = {i}, Final Loss: {current[-1]:.6f} ---")

    print("\n--- All Training Complete ---")


    # --- Plotting the comparison ---
    plt.figure(figsize=(12, 8))
    
    for rate, (epoch_list, loss_list) in all_losses.items():
        plt.plot(epoch_list, loss_list, label=f'Learning Rate = {rate}')
        
    plt.title('Training Loss vs. Epoch for Different Learning Rates')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.grid(True)
    
    # Using a log scale for the Y-axis is often helpful for loss plots to see differences more clearly as the loss gets small.
    plt.yscale('log') 
    
    plt.savefig('comparison_loss_plot.png')
    print("Saved 'comparison_loss_plot.png'")
    plt.clf() # Clear the figure
    print("--- Comparison Plot Generated ---")

if __name__ == "__main__":

    # Run the first script (modified original)
    run_and_plot_single_run()
    
    # Run the second script (comparison)
    run_and_plot_comparison()





  