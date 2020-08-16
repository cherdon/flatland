import torch.nn as nn


def dim_output(input_dim, filter_dim, stride_dim):
    return (input_dim - filter_dim) // stride_dim + 1


class Dueling_DQN(nn.Module):
    """
        :input_shape: (batch_size, in_channels = height/num_rails, width/prediction_depth + 1)
        :output_shape: (batch_size, out_channels, conv_width)
    """
    def __init__(self, width, height, action_space):
        super(Dueling_DQN, self).__init__()
        self.action_space = action_space
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, width))

        self.fc1_advance = nn.Linear(in_features=64 * height, out_features=512)
        self.fc1_value = nn.Linear(in_features=64 * height, out_features=512)
        self.fc2_advance = nn.Linear(in_features=512, out_features=action_space)
        self.fc2_value = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_advance(x))
        val = self.relu(self.fc1_value(x))

        adv = self.fc2_advance(adv)
        val = self.fc2_value(val).expand(x.size(0), self.action_space)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_space)
        return x
