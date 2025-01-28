import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.inputs = []
        self.outputs = []
        for key_str, output in data.items():
            # Convert key string to list of floats (adjust based on your key format)
            input_arr = list(map(float, key_str.split(',')))
            self.inputs.append(input_arr)
            self.outputs.append(output)
        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.outputs = torch.tensor(np.array(self.outputs), dtype=torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# Custom Checkpoint Layer
class CheckpointLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.main_layer = nn.Linear(input_dim, output_dim)
        self.confidence_layer = nn.Linear(output_dim, 1)
        
    def forward(self, x):
        output = self.main_layer(x)
        confidence = torch.sigmoid(self.confidence_layer(output))
        return output, confidence

# Main Model
class CustomModel(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.layers = nn.ModuleList()
        current_dim = self.config['input_dim']
        
        for layer_config in self.config['hidden_layers']:
            layer_type = layer_config['type']
            if layer_type == 'fully_connected':
                layer = nn.Linear(current_dim, layer_config['size'])
                self.layers.append(layer)
                self.layers.append(nn.ReLU())
                current_dim = layer_config['size']
            elif layer_type == 'checkpoint':
                layer = CheckpointLayer(current_dim, layer_config['size'])
                self.layers.append(layer)
                current_dim = layer_config['size']
            elif layer_type == 'convolutional':
                # Example for Conv1D (adjust parameters as needed)
                in_channels = current_dim
                out_channels = layer_config['size']
                kernel_size = layer_config.get('kernel_size', 3)
                layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
                self.layers.append(layer)
                self.layers.append(nn.ReLU())
                current_dim = out_channels  # Adjust based on actual output dimensions
        
        self.final_layer = nn.Linear(current_dim, self.config['output_dim'])
        self.learning_rate = self.config.get('learning_rate', 0.001)
    
    def forward(self, x, training=True, threshold=0.1):
        checkpoint_outputs = []
        confidences = []
        x = x.view(x.size(0), -1)  # Flatten input for fully connected layers
        
        for layer in self.layers:
            if isinstance(layer, CheckpointLayer):
                x, confidence = layer(x)
                checkpoint_outputs.append(x)
                confidences.append(confidence)
                
                if not training:
                    batch_avg_confidence = torch.mean(confidence)
                    if batch_avg_confidence < threshold:
                        return x
            else:
                x = layer(x)
        
        final_output = self.final_layer(x)
        
        if training:
            return final_output, checkpoint_outputs, confidences
        else:
            return final_output

# Training Function
def train_model(model, dataloader, num_epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            final_output, checkpoint_outputs, confidences = model(inputs, training=True)
            
            main_loss = criterion(final_output, targets)
            confidence_loss = 0.0
            
            for co, conf in zip(checkpoint_outputs, confidences):
                delta = torch.mean((co - final_output) ** 2, dim=1)
                scaled_delta = delta / (delta + 1)  # Scale delta to [0, 1)
                conf_loss = torch.mean((conf.squeeze() - scaled_delta) ** 2)
                confidence_loss += conf_loss
            
            total_batch_loss = main_loss + confidence_loss
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

# Example Usage
if __name__ == "__main__":
    # Initialize dataset and model
    dataset = CustomDataset('dataset.json')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = CustomModel('config.json')
    
    # Train the model
    train_model(model, dataloader, num_epochs=10)
    
    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')