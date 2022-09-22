class Block(nn.Module):

    def __init__(self, trial, i):
        super(Block, self).__init__()

        activation = trial.suggest_categorical(f"activation_conv_layer{i}", ["LeakyReLU", "ReLU", "ELU"])

        self.activ = getattr(nn, activation)()

        padding_mode = "replicate"#trial.suggest_categorical(f"padding_mode_layer{i}", ["zeros", "reflect", "replicate"])

        kernel_size1 = trial.suggest_int(f"conv1_kernel_layer{i}", 2, 13)
        kernel_size2 = trial.suggest_int(f"conv2_kernel_layer{i}", 2, 13)
        kernel_size3 = trial.suggest_int(f"conv3_kernel_layer{i}", 2, 13)

        channel_size_conv_intermediate = 189#trial.suggest_int(f"channel_size_conv_intermediate_layer{i}", 100, 500, 50)


        self.conv1 = nn.Sequential(
            nn.Conv1d(189, channel_size_conv_intermediate, kernel_size=kernel_size1, stride=1, padding="same", padding_mode=padding_mode),
            nn.BatchNorm1d(channel_size_conv_intermediate),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(channel_size_conv_intermediate, channel_size_conv_intermediate, kernel_size=kernel_size2, stride=1, padding="same", padding_mode=padding_mode),
            nn.BatchNorm1d(channel_size_conv_intermediate),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(channel_size_conv_intermediate, 189, kernel_size=kernel_size3, stride=1, padding="same", padding_mode=padding_mode),
            nn.BatchNorm1d(189),
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.activ(x)
        x = self.conv3(x)
        x = x + identity
        x = self.activ(x)
        return x

class ResNet(nn.Module):


    def __init__(self, trial):
        super(ResNet, self).__init__()

        n_layers_blocks = trial.suggest_int("n_layers_blocks", 1, 10)

        self.blocks = nn.Sequential()

        for i in range(n_layers_blocks):
            self.blocks.append(Block(trial, i))
            
        self.linear = nn.Sequential()

        linear_in = 189

        n_layers_linear = trial.suggest_int("n_layers_linear", 1, 20)


        for k in range(n_layers_linear):

            linear_out = trial.suggest_int(f"linear_size_layer{k}", 100, 300, 50)

            self.linear.append(nn.Linear(linear_in, linear_out))

            self.linear.append(nn.LeakyReLU())

            linear_in = linear_out

        self.linear.append(nn.Linear(linear_in, 1))

        self.linear.append(nn.Sigmoid())


    def forward(self, x):
        x = self.blocks(x)

        avg = nn.AvgPool1d(13, stride=1)
        x = avg(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.linear = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)     
        out = self.linear(hidden)
        return out

model = LSTM(189, 10, 10)
model = model.to(device)