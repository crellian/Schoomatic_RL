import rospy
from sklearn.metrics import mean_squared_error
from ray.rllib.policy.policy import Policy
import cv2
from .fpvbev import Encoder
from .models import VAEBEV, MDLSTM
import torch
import numpy as np
from std_msgs.msg import Int64, Float64
from geometry_msgs.msg import Twist

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

class Map:
    def __init__(self):
        rospy.init_node('map', anonymous=True)

        vae_model_path = "/home2/ckpts/pretrained/models/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
        naive_lstm_path = "/home2/ckpts/pretrained/carla/BEV_LSTM_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_1_512.pt"

        vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

        self.bev_lstm = MDLSTM(latent_size=32, action_size=2, hidden_size=256, num_layers=1, gaussian_size=5,
                          vae=vae).to(device)
        self.bev_lstm.eval()
        self.bev_lstm.init_hs()
        checkpoint = torch.load(naive_lstm_path, map_location="cpu")

        self.bev_lstm.load_state_dict(checkpoint['model_state_dict'])
        for param in self.bev_lstm.parameters():
            param.requires_grad = False

        vae_ckpt = torch.load(vae_model_path, map_location="cpu")
        self.bev_lstm.vae.load_state_dict(vae_ckpt['model_state_dict'])

        self.encoder = Encoder("/home2/ckpts/pretrained/models/FPV_BEV_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_128_512.pt",
                          False)

        self.i = 0
        self.window = []
        self.z = None

        self.map_pub = rospy.Publisher(
            '/occ_map', Int64, queue_size=1)

        self.aux_pub = rospy.Publisher(
            '/aux', Float64, queue_size=1)

        self.cmd_sub = rospy.Subscriber(
            '/cmd', Twist, self.cmd_cb)

        self.throttle = 0
        self.steer = 0


    def method(self, action, info):
        self.i += 1
        RGB_img = cv2.resize(info['rgb_obs'], (84, 84), interpolation=cv2.INTER_LINEAR)

        rid, score, image_embed = self.encoder(RGB_img)  # FPV embedding and approx

        if self.i < 11:
            id = rid
            s = image_embed  # input to lstm
            l_t = self.encoder.label[rid]

            r_ = r__ = torch.reshape(self.bev_lstm.vae.recon(image_embed),  # FPV-BEV
                               (64, 64)).cpu().numpy() * 255
            self.window.append(0)


        else:
            nid = self.encoder(self.z, False)[0]
            l_t = self.encoder.label[nid]  # label at t
            # o = encoder.anchors[nid].reshape((64, 64))  # BEV output at t

            r_ = torch.reshape(self.bev_lstm.vae.recon(image_embed),  # FPV-BEV
                               (64, 64)).cpu().numpy() * 255
            r__ = torch.reshape(self.bev_lstm.vae.recon(self.z),  # LSTM raw output
                                (64, 64)).cpu().numpy() * 255

            mse = mean_squared_error(r_.reshape((1, 64 * 64)),
                                     r__.reshape((1, 64 * 64)))

            if score > 0.85:
                self.window = self.window[1:]
                self.window.append(1 if mse > 10000 else 0)
                w = sum(self.window) / len(self.window)
            else:
                w = 0

            self.z = image_embed * w + self.z * (1 - w)
            id, _, _ = self.encoder(self.z, False)
            s = self.z

        if s is not None:
            out = self.bev_lstm(torch.Tensor([action]).to(device), s)
            mus = out[0][0]
            pi = torch.exp(out[2][0])
            self.z = (mus[0] * pi[0] + mus[1] * pi[1] + mus[2] * pi[2] + mus[3] * pi[3] + mus[4] * pi[4]).unsqueeze(0)
        print(score)
        return l_t, r__, r_

    def cmd_cb(self, twist):
        self.throttle = twist.linear.x
        self.steer = twist.angular.z


