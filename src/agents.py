import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        module.bias.data.fill_(0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)


# TODO: Merge into the other classes, not needed anymore after the changes
class LSTMLayer(nn.Module):
    def __init__(self, input_dim: int = 600, hidden_dim_lstm: int = 1024):
        super().__init__()
        self.input_dim = input_dim
        self.hiddem_dim_lstm = hidden_dim_lstm
        self.lstm = nn.LSTM(self.input_dim, self.hiddem_dim_lstm, batch_first=True)

    def weight_init(self):
        self.apply(init_weights)

    def get_initial_state(self, device, batch_size=1):
        return torch.zeros((1, batch_size, self.hiddem_dim_lstm), dtype=torch.float32, device=device), \
               torch.zeros((1, batch_size, self.hiddem_dim_lstm), dtype=torch.float32, device=device)

    def forward(self, observations, agent_state):
        obs1, agent_state = self.lstm(observations, agent_state)
        return obs1, agent_state


class MLPReinforceAgent(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim_mlp: int = 2048, output_dim: int = 600):
        super().__init__()
        self.input_dim = input_dim
        self.hiddem_dim_mlp = hidden_dim_mlp
        self.output_dim = output_dim
        self.fc_1 = nn.Linear(self.input_dim, self.hiddem_dim_mlp)
        self.fc_2 = nn.Linear(self.hiddem_dim_mlp, self.output_dim)

    def weight_init(self):
        self.apply(init_weights)

    def get_initial_state(self):
        return ()

    def forward(self, observations, actions, _):
        observations = self.fc_2(F.relu(self.fc_1(observations)))
        scores = torch.matmul(actions, observations.unsqueeze(-1))
        return scores.squeeze(), _, _


class LSTMReinforceAgent(nn.Module):
    def __init__(self, input_dim: int = 600,
                 output_dim: int = 600,
                 hidden_dim_mlp: int = 2048,
                 hidden_dim_lstm: int = 1024):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim_mlp = hidden_dim_mlp
        self.hidden_dim_lstm = hidden_dim_lstm
        self.lstm_layer = LSTMLayer(input_dim=self.input_dim, hidden_dim_lstm=self.hidden_dim_lstm)
        self.head = MLPReinforceAgent(input_dim=self.hidden_dim_lstm,
                                      hidden_dim_mlp=self.hidden_dim_mlp,
                                      output_dim=self.output_dim)

    def weight_init(self):
        self.apply(init_weights)

    def get_initial_state(self, device, batch_size=1):
        return self.lstm_layer.get_initial_state(device, batch_size)

    def forward(self, observations, actions, agent_state):
        obs, agent_state = self.lstm_layer(observations, agent_state)
        obs1, _, agent_state = self.head(obs, actions, agent_state)
        return obs1, None, agent_state


class LSTMActorCriticAgent(nn.Module):
    def __init__(self, input_dim: int = 600,
                 output_dim: int = 600,
                 hidden_dim_mlp: int = 2048,
                 hidden_dim_lstm: int = 1024):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim_mlp = hidden_dim_mlp
        self.hidden_dim_lstm = hidden_dim_lstm
        self.lstm_layer = LSTMLayer(input_dim=self.input_dim, hidden_dim_lstm=self.hidden_dim_lstm)
        self.actor_head = MLPReinforceAgent(input_dim=self.hidden_dim_lstm,
                                            hidden_dim_mlp=self.hidden_dim_mlp,
                                            output_dim=self.output_dim)
        self.critic_head = nn.Sequential(nn.Linear(self.hidden_dim_lstm, self.hidden_dim_mlp), nn.ReLU(),
                                         nn.Linear(self.hidden_dim_mlp, 1))

    def weight_init(self):
        self.apply(init_weights)

    def get_initial_state(self, device, batch_size=1):
        return self.lstm_layer.get_initial_state(device, batch_size)

    def forward(self, observations, actions, agent_state):
        obs1, agent_state = self.lstm_layer(observations, agent_state)
        obs, _, agent_state = self.actor_head(obs1, actions, agent_state)
        values = self.critic_head(obs1)
        return obs, values.squeeze(), agent_state
