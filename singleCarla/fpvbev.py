import numpy as np
import torch.nn as nn
import os
import cv2
import torch
from torchvision.models import resnet50, ResNet50_Weights
import random
GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else "cpu")

class BEVEncoder(nn.Module):
    def __init__(self, channel_in=3, ch=32, h_dim=512, z=32):
        super(BEVEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, ch, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Linear(h_dim, z)

    def forward(self, x):
        return self.fc(self.encoder(x))

class ResNet(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, embed_size)


    def forward(self, image):
        out = self.resnet(image)
        return out

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
class Encoder(nn.Module):
    def __init__(self, encoder_path, classification=False):
        super().__init__()
        self.fpvencoder = ResNet(32).to(device)

        # vaeencoder
        self.bevencoder = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        vae_model_path = "/home2/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
        vae_ckpt = torch.load(vae_model_path, map_location="cpu")
        self.bevencoder.load_state_dict(vae_ckpt['model_state_dict'])
        self.bevencoder.eval()
        for param in self.bevencoder.parameters():
            param.requires_grad = False

        # load models
        checkpoint = torch.load(encoder_path, map_location="cpu")
        print(checkpoint['epoch'])
        self.fpvencoder.load_state_dict(checkpoint['fpv_state_dict'])
        self.fpvencoder.eval()
        for param in self.fpvencoder.parameters():
            param.requires_grad = False


        # read anchor images and convert to latent representations
        self.anchors_lr = []
        self.anchors = []
        self.label = []
        self.fn = []
        if not classification:
            root = "/home/carla/img2cmd/anchor"
        else:
            root = "/home/carla/img2cmd/anchor1"

        for root, subdirs, files in os.walk(root):
            for f in sorted(files):
                if ".jpg" in f:
                    im = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
                    self.anchors.append(im)
                    self.label.append(int(root[-1]))
                    self.fn.append(f[:-4])
                    with torch.no_grad():
                        im = np.expand_dims(im, axis=(0, 1))
                        im = torch.tensor(im).to(device) / 255.0
                        _, embed_mu, embed_logvar = self.bevencoder(im)
                        embed_mu = embed_mu.cpu().numpy()[0]
                        embed_logvar = embed_logvar.cpu().numpy()[0]
                        self.anchors_lr.append(embed_mu)

        self.anchors_lr = np.array(self.anchors_lr)
        self.anchors_lr = torch.tensor(self.anchors_lr).to(device)


    def forward(self, img, fpv=True):
        # img - rgb observation, bev - ground truth bev observation
        if fpv:
            img = np.expand_dims(img, axis=0)
            img = np.transpose(img, (0,3,1,2))
            image_val = torch.tensor(img).to(device) / 255.0

            with torch.no_grad():
                # encode rgb image
                image_embed = self.fpvencoder(image_val)
        else:
            image_embed = img


        # compare images
        '''
        pred = self.bevencoder.recon(image_embed)[0][0]

        sims = []
        for an in self.anchors:
            sim = nn.functional.cosine_similarity(pred.reshape(1, 64*64), torch.tensor(an).to(device).reshape(1, 64*64))
            sims.append(sim)
        sims = torch.tensor(sims)
        # probs = nn.functional.softmax(sims)
        ys = torch.topk(sims, 2).indices'''
        # compare embeddings
        #sims = nn.functional.cosine_similarity(image_embed, self.anchors_lr)
        sims = nn.functional.cosine_similarity(image_embed, self.anchors_lr)
        # probs = nn.functional.softmax(sims)
        ys = torch.argmax(sims)

        return ys.cpu().numpy(), float(torch.max(sims).cpu().numpy()), image_embed


def readSim(top, fpv=True):
    cnt = 0

    if fpv:
        t = 0
        for root, subdirs, files in os.walk("/home/carla/img2cmd/test"):
            for f in sorted(files):
                if "bev" not in f:
                    rgb = cv2.imread(os.path.join("/home/carla/img2cmd/test", f))
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    RGB_img = cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_LINEAR)
                    id, score, image_embed = encoder(RGB_img)

                    #z = encoder.bevencoder.reparameterize(image_embed[:, 32], image_embed[:, 32:])
                    cv2.imwrite(os.path.join("/home/carla/img2cmd/test", f[:-4] + "_1.jpg"),
                                encoder.bevencoder.recon(image_embed)[0][0].cpu().numpy() * 255)
                    cv2.imwrite(os.path.join("/home/carla/img2cmd/test", f[:-4] + "_2.jpg"),
                                encoder.anchors[id])

                    if id == cnt:
                        t += 1
                    cnt += 1
                    print(f, id, score)
        print(t)
    else:
        for i in range(10):
            for root, subdirs, files in os.walk("/home2/sim_val/" + str(i) + "/"):
                t = 0
                cnt = np.zeros(10)
                total = 0
                for f in sorted(files):
                    if '.jpg' in f and "_1.jpg" not in f and "_2_" not in f and "_2.jpg" not in f:
                        total += 1
                        # print(os.path.join(root, f))
                        rgb = cv2.imread(os.path.join(root, f))
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                        RGB_img = cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_LINEAR)

                        id, id_2, image_embed = encoder(RGB_img)

                        # print(probs)

                        cv2.imwrite(os.path.join(root, f[:-4] + "_1.jpg"), encoder.bevencoder.recon(image_embed)[0][0].cpu().numpy()*255)
                        cv2.imwrite(os.path.join(root, f[:-4] + "_2_"+str(encoder.fn[id])+"_"+str(encoder.label[id])+".jpg"),
                                    encoder.anchors[id].cpu().numpy())

                        cnt[encoder.label[id]] += 1
                        if top == 1 and encoder.label[id] == i:
                            t += 1

                if files:
                    print(root, total, "accuracy:", t / total)
                    print(cnt / total)

def readReal(top, bag=False):
    if not bag:
        for i in range(10):
            for root, subdirs, files in os.walk("/home2/USC_GStView/" + str(i) + "/"):
                t = 0
                cnt = np.zeros(10)
                total = 0
                for f in sorted(files):
                    if '.jpg' in f and "_1.jpg" not in f and "_2_" not in f and "_2.jpg" not in f:
                        total += 1
                        # print(os.path.join(root, f))
                        rgb = cv2.imread(os.path.join(root, f))
                        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                        RGB_img = cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_LINEAR)

                        id, id_2, image_embed = encoder(RGB_img)

                        # print(probs)

                        cv2.imwrite(os.path.join(root, f[:-4] + "_1.jpg"), encoder.bevencoder.recon(image_embed)[0][0].cpu().numpy()*255)
                        cv2.imwrite(os.path.join(root, f[:-4] + "_2_"+str(encoder.fn[id])+"_"+str(encoder.label[id])+".jpg"),
                                    encoder.anchors[id])

                        #im = encoder.anchors[id.cpu().numpy()]
                        #im = np.expand_dims(im, axis=(0, 1))  # classifier
                        #im = torch.tensor(im).float().to(device) / 255.0
                        #_, mu, logvar = encoder.bevencoder(im)
                        #fs = torch.concat((mu, logvar), axis=-1).reshape(64, )
                        #output = net(fs).reshape(6, )
                        #prob = torch.exp(output) / torch.sum(torch.exp(output))
                        #id = torch.argmax(prob)
                        cnt[encoder.label[id]] += 1
                        if top == 1 and encoder.label[id] == i:
                            t += 1

                if files:
                    print(root, total, "accuracy:", t / total)
                    print(cnt / total)

    else:
        root = "/home2/carla/2023_08_20/test_2023-08-20-01-22-05"
        for file in sorted(os.listdir(root)):
            if "_1" not in file and "_2" not in file:
                print(file)
                rgb = cv2.imread(os.path.join(root, file))
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                RGB_img = cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_LINEAR)
                id, sim, image_embed = encoder(RGB_img)

                #z = encoder.bevencoder.reparameterize(image_embed[:, 32], image_embed[:, 32:])
                if sim > 0.9:
                    cv2.imwrite(os.path.join(os.path.join(root, file[:-4]+"_1.jpg")),
                                encoder.bevencoder.recon(image_embed)[0][0].cpu().numpy() * 255)
                    cv2.imwrite(os.path.join(os.path.join(root, file[:-4]+"_2_"+str(round(sim, 2))+".jpg")),
                                encoder.anchors[id.cpu().numpy()])




if __name__ == "__main__":
    net = FFN().to(device)
    net.load_state_dict(torch.load("/home/carla/img2cmd/bev.pt"))
    net.eval()

    encoder = Encoder("/lab/kiran/ckpts/pretrained/FPV_BEV_CARLA_OLD_STANDARD_0.1_0.01_128_512.pt", False)
    #encoder = Encoder("/lab/kiran/ckpts/pretrained/carlaFPV_BEV_CARLA_NEWRANDOM_BEV_CARLA_STANDARD_0.1_0.01_128_512.pt", True)
    #encoder = Encoder("/lab/kiran/ckpts/pretrained/carla/FPV_BEV_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_128_512.pt", False)
    #encoder = Encoder("/lab/kiran/ckpts/pretrained/FPV_RECONBEV_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_128_512.pt", False)

    readReal(1, False)
