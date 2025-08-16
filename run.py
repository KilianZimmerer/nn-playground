import torch
import matplotlib.pyplot as plt
import argparse
from model import Model
from matplotlib.gridspec import GridSpec

LAYER_SIZES = [10, 15, 10]
EPOCHS = 10000
PLOT_HEIGHT_ROWS = 8
PADDING_ROWS = 2

def target_function(x):
    return x**3 + x**2 + x + 1

def train_model(model, X, y, epochs):
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training for model with layer sizes: {LAYER_SIZES}")
    for epoch in range(epochs):
        y_pred = model(X)
        loss = loss_function(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return model

def generate_predictions(model, x_range):
    new_x = torch.linspace(x_range[0], x_range[1], 1000).unsqueeze(1)
    predicted_y, neuron_outputs = model(new_x, return_neuron_outputs=True)
    true_y = target_function(new_x)
    return new_x, true_y, predicted_y, neuron_outputs

def plot_results(new_x, true_y, predicted_y, neuron_outputs, output_filename):
    max_neurons_in_layer = max(LAYER_SIZES) if LAYER_SIZES else 0
    num_hidden_layers_plotted = len(neuron_outputs)
    num_total_cols = num_hidden_layers_plotted + 1
    TOTAL_GRID_ROWS = 10 * max_neurons_in_layer

    fig = plt.figure(figsize=(4.5 * num_total_cols, 12))
    fig.suptitle('Neuron Activations', fontsize=20)
    gs = GridSpec(TOTAL_GRID_ROWS, num_total_cols, figure=fig, width_ratios=[1] * num_hidden_layers_plotted + [1.5])

    for i in range(num_hidden_layers_plotted):
        num_neurons = LAYER_SIZES[i]
        layer_output_np = neuron_outputs[i].detach().numpy()
        content_height = (num_neurons * PLOT_HEIGHT_ROWS) + (max(0, num_neurons - 1) * PADDING_ROWS)
        top_margin_rows = (TOTAL_GRID_ROWS - content_height) // 2

        for j in range(num_neurons):
            start_row = top_margin_rows + j * (PLOT_HEIGHT_ROWS + PADDING_ROWS)
            end_row = start_row + PLOT_HEIGHT_ROWS
            ax = fig.add_subplot(gs[start_row:end_row, i])
            ax.plot(new_x.numpy(), layer_output_np[:, j], color='teal')
            ax.grid(True, linestyle='--', alpha=0.6)

            if j == 0:
                ax.set_title(f'Layer {i+1}', fontsize=14)
            if j < num_neurons - 1:
                ax.tick_params(axis='x', labelbottom=False)
            if i > 0:
                ax.tick_params(axis='y', labelleft=False)
            else:
                ax.set_ylabel(f'N{j+1}', rotation=0, labelpad=20, ha='right', va='center')

    output_plot_height_rows = 40
    start_row = (TOTAL_GRID_ROWS - output_plot_height_rows) // 2
    end_row = start_row + output_plot_height_rows
    ax_out = fig.add_subplot(gs[start_row:end_row, -1])
    ax_out.set_title("Final Result", fontsize=14)
    ax_out.plot(new_x.numpy(), true_y.numpy(), color='red', linestyle='--', linewidth=2, label='True Function')
    ax_out.plot(new_x.numpy(), predicted_y.detach().numpy(), color='blue', linewidth=2, label='Model Prediction')
    ax_out.legend()
    ax_out.grid(True)
    fig.supxlabel('Input (x)', fontsize=14)
    plt.savefig(output_filename)

def main():
    parser = argparse.ArgumentParser(description='Train a neural network and plot neuron activations.')
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='plots/neuron_grid.png',
        help='Path to save the output plot file.'
    )
    args = parser.parse_args()

    model = Model(layer_sizes=LAYER_SIZES)
    X = torch.randn(1000, 1) * 10
    y = target_function(X)

    trained_model = train_model(model, X, y, EPOCHS)

    new_x, true_y, predicted_y, neuron_outputs = generate_predictions(trained_model, (-10, 10))

    plot_results(new_x, true_y, predicted_y, neuron_outputs, args.output)

if __name__ == "__main__":
    main()