import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as f
from IPython import embed

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)

class VAEBEV(nn.Module):
    def __init__(self, channel_in=3, ch=32, h_dim=512, z=32):
        super(VAEBEV, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, ch, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z)
        self.fc2 = nn.Linear(h_dim, z)
        self.fc3 = nn.Linear(z, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, ch * 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch * 2, channel_in, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().cuda()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def recon(self, z):
        z = self.fc3(z)
        return self.decoder(z)

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return self.recon(z), mu, logvar

class StateLSTM(nn.Module):
    def __init__(self, latent_size, hidden_size, num_layers, encoder):
        super().__init__()
        self.encoder = encoder
        if encoder is not None:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True).cuda()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_hs(self):
        self.h_0 = Variable(torch.randn((self.num_layers, self.hidden_size))).to(device)
        self.c_0 = Variable(torch.randn((self.num_layers, self.hidden_size))).to(device)

    def forward(self, image):
        # x = torch.reshape(image, (-1,) + image.shape[-3:]).float()
        x = image
        z = self.encoder(x).float()
        z = torch.reshape(z, (1, image.shape[0], -1))
        # z = torch.reshape(z, image.shape[:2] + (-1,))
        outs, (self.h_0, self.c_0) = self.lstm(z.float(), (self.h_0, self.c_0))
        return outs


class StateActionLSTM(StateLSTM):
    def __init__(self, latent_size, action_size, hidden_size, num_layers, encoder=None, vae=None):
        super().__init__(latent_size=latent_size, hidden_size=hidden_size, num_layers=num_layers, encoder=encoder)
        self.vae = vae
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.lstm = nn.LSTM(latent_size + action_size, hidden_size, num_layers, batch_first=True)

    def encode(self, image):
        x = torch.reshape(image, (-1,) + image.shape[-3:])
        _, mu, logvar = self.vae(x)
        z = self.vae.reparameterize(mu, logvar)
        z = torch.reshape(z, image.shape[:2] + (-1,))
        return z, mu, logvar

    def decode(self, z):
        z_f = torch.reshape(z, (-1,) + (z.shape[-1],))
        img = self.vae.recon(z_f)
        return torch.reshape(img, z.shape[:2] + img.shape[-3:])

    def forward(self, action, latent):
        in_al = torch.cat([action, latent], dim=-1)
        outs, (self.h_0, self.c_0) = self.lstm(in_al.float(), (self.h_0, self.c_0))
        return outs


class MDLSTM(StateActionLSTM):
    def __init__(self, latent_size, action_size, hidden_size, num_layers, gaussian_size, encoder=None, vae=None):
        super().__init__(latent_size, action_size, hidden_size, num_layers, encoder, vae)
        self.gaussian_size = gaussian_size
        self.gmm_linear = nn.Linear(hidden_size, (2 * latent_size + 1) * gaussian_size)

    def forward(self, action, latent):
        seq_len = action.size(0)
        in_al = torch.cat([torch.Tensor(action), latent], dim=-1)
        outs, (self.h_0, self.c_0) = self.lstm(in_al.float(), (self.h_0, self.c_0))

        gmm_outs = self.gmm_linear(outs)
        stride = self.gaussian_size * self.latent_size

        mus = gmm_outs[:, :stride]
        mus = mus.view(seq_len, self.gaussian_size, self.latent_size)

        sigmas = gmm_outs[:, stride:2 * stride]
        sigmas = sigmas.view(seq_len, self.gaussian_size, self.latent_size)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, 2 * stride: 2 * stride + self.gaussian_size]
        pi = pi.view(seq_len, self.gaussian_size)
        logpi = f.log_softmax(pi, dim=-1)

        return mus, sigmas, logpi


'''
class LSTM(nn.Module):
    def __init__(self, hidden_layers=64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)

    def forward(self, y, future_preds=0):
        outputs, num_samples = [], y.size(0)
        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)

        for time_step in y.split(1, dim=1):
            # N, 1
            h_t, c_t = self.lstm1(input_t, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            outputs.append(output)

        for i in range(future_preds):
            # this only generates future predictions if we pass in future_preds>0
            # mirrors the code above, using last output/prediction as input
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        # transform list to tensor    
        outputs = torch.cat(outputs, dim=1)
        return outputs
'''


