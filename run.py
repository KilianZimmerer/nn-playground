import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers):
        super().__init__()
        self.input_layer = nn.Linear(in_features=1, out_features=hidden_size)
        
        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        # Apply ReLU after the input layer
        x = torch.relu(self.input_layer(x))
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


def function(x_arg):
    return 2*x_arg**2 + 1

# Instantiate the model
model = Model(hidden_size=10, num_hidden_layers=2)

X = torch.randn(1000, 1) * 10
y = function(X)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10000
for epoch in range(epochs):
    y_pred = model(X)
    loss = loss_function(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


print("\nLearned Parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# Create a sorted range of x values for a clean plot
new_x = torch.linspace(-10, 10, 1000).unsqueeze(1)
predicted_y = model(new_x)
true_y = function(new_x)

# Plot predictions and true values
plt.figure(figsize=(10, 6))
plt.plot(new_x.numpy(), predicted_y.detach().numpy(), label='Predictions', color='blue')
plt.plot(new_x.numpy(), true_y.numpy(), label='True Parabola', color='red', linestyle='--')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('Model Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()
