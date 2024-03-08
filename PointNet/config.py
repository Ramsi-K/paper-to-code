# Model configurations
num_classes = 10  # Number of output classes
num_points = 1024  # Number of input points
input_channels = 3  # Number of input channels (x, y, z coordinates)

# Training configurations
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Data configurations
train_data_path = "data/train_dataset.npy"
test_data_path = "data/test_dataset.npy"
