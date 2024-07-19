import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import cv2
import subprocess
import random
import binascii
from base64 import b64encode, b64decode
import hashlib
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
import traceback
from pydub import AudioSegment
import soundfile as sf
import math
import ffmpeg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tempfile.tempdir = "/tmp"

IMG_SIZE = 64
LEARNING_RATE = 0.001
COVER_LOSS_WEIGHT = 1
SECRET_LOSS_WEIGHT = 1
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 1
EPOCHS = 1000
DECODER_LOSS_WEIGHT = 1


class PrepNetwork1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=50, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=5,
                               kernel_size=(5, 5), stride=1, padding=2)

        self.conv4 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2)

    def forward(self, secret_image):
        output_1 = F.relu(self.conv1(secret_image))
        output_2 = F.relu(self.conv2(secret_image))
        output_3 = F.relu(self.conv3(secret_image))

        concatenated_image = torch.cat([output_1, output_2, output_3], dim=1)
        output_4 = F.relu(self.conv4(concatenated_image))
        output_5 = F.relu(self.conv5(concatenated_image))
        output_6 = F.relu(self.conv6(concatenated_image))

        final_concat_image = torch.cat([output_4, output_5, output_6], dim=1)
        return final_concat_image


class HidingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=68, out_channels=50, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=68, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=68, out_channels=5, kernel_size=(5, 5), stride=1, padding=2)

        self.conv4 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2)

        self.conv7 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv9 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2)

        self.conv10 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1)
        self.conv11 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv12 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2)

        self.conv13 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1)
        self.conv14 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv15 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2)

        self.final_layer = nn.Conv2d(
            in_channels=65, out_channels=3, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, secret_image_1, cover_image):
        concatenated_secrets = torch.cat([cover_image, secret_image_1], dim=1)

        output_1 = F.relu(self.conv1(concatenated_secrets))
        output_2 = F.relu(self.conv2(concatenated_secrets))
        output_3 = F.relu(self.conv3(concatenated_secrets))
        concat_1 = torch.cat([output_1, output_2, output_3], dim=1)

        output_4 = F.relu(self.conv4(concat_1))
        output_5 = F.relu(self.conv5(concat_1))
        output_6 = F.relu(self.conv6(concat_1))
        concat_2 = torch.cat([output_4, output_5, output_6], dim=1)

        output_7 = F.relu(self.conv7(concat_2))
        output_8 = F.relu(self.conv8(concat_2))
        output_9 = F.relu(self.conv9(concat_2))
        concat_3 = torch.cat([output_7, output_8, output_9], dim=1)

        output_10 = F.relu(self.conv10(concat_3))
        output_11 = F.relu(self.conv11(concat_3))
        output_12 = F.relu(self.conv12(concat_3))
        concat_4 = torch.cat([output_10, output_11, output_12], dim=1)

        output_13 = F.relu(self.conv13(concat_4))
        output_14 = F.relu(self.conv14(concat_4))
        output_15 = F.relu(self.conv15(concat_4))
        concat_5 = torch.cat([output_13, output_14, output_15], dim=1)

        output_converted_image = F.relu(self.final_layer(concat_5))

        return output_converted_image


class Encoder(nn.Module):
    def __init__(self, prep_network_1, hiding_network):
        super(Encoder, self).__init__()
        self.prep_network1 = prep_network_1
        self.hiding_network = hiding_network

    def forward(self, cover_image, secret_image_1):
        encoded_secret_image_1 = self.prep_network1(secret_image_1)

        hidden_image = self.hiding_network(encoded_secret_image_1,
                                           cover_image
                                           )
#         hidden_image = (0.01**0.5)*torch.randn(hidden_image.size(),device=device)
        return hidden_image


class RevealNetwork1(nn.Module):
    def __init__(self):
        super(RevealNetwork1, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=50, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=5,
                               kernel_size=(5, 5), stride=1, padding=2)

        self.conv4 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2)

        self.conv7 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv9 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2)

        self.conv10 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1)
        self.conv11 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv12 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2)

        self.conv13 = nn.Conv2d(
            in_channels=65, out_channels=50, kernel_size=(3, 3), stride=1, padding=1)
        self.conv14 = nn.Conv2d(
            in_channels=65, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv15 = nn.Conv2d(
            in_channels=65, out_channels=5, kernel_size=(5, 5), stride=1, padding=2)

        self.final_layer = nn.Conv2d(
            in_channels=65, out_channels=3, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, hidden_image):

        output_1 = F.relu(self.conv1(hidden_image))
        output_2 = F.relu(self.conv2(hidden_image))
        output_3 = F.relu(self.conv3(hidden_image))
        concat_1 = torch.cat([output_1, output_2, output_3], dim=1)

        output_4 = F.relu(self.conv4(concat_1))
        output_5 = F.relu(self.conv5(concat_1))
        output_6 = F.relu(self.conv6(concat_1))
        concat_2 = torch.cat([output_4, output_5, output_6], dim=1)

        output_7 = F.relu(self.conv7(concat_2))
        output_8 = F.relu(self.conv8(concat_2))
        output_9 = F.relu(self.conv9(concat_2))
        concat_3 = torch.cat([output_7, output_8, output_9], dim=1)

        output_10 = F.relu(self.conv10(concat_3))
        output_11 = F.relu(self.conv11(concat_3))
        output_12 = F.relu(self.conv12(concat_3))
        concat_4 = torch.cat([output_10, output_11, output_12], dim=1)

        output_13 = F.relu(self.conv13(concat_4))
        output_14 = F.relu(self.conv14(concat_4))
        output_15 = F.relu(self.conv15(concat_4))
        concat_5 = torch.cat([output_13, output_14, output_15], dim=1)

        output_revealed_image = F.relu(self.final_layer(concat_5))

        return output_revealed_image


class Decoder(nn.Module):
    def __init__(self, reveal_network_1):
        super().__init__()
        self.reveal_network_1 = reveal_network_1

    def forward(self, hidden_image):
        reveal_image_1 = self.reveal_network_1(hidden_image)
        return reveal_image_1


class SteganoModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(SteganoModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, cover_image, secret_image_1, hidden_image, mode):
        if mode == 'full':
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = False
            hidden_image = self.encoder(cover_image, secret_image_1)
            reveal_image_1 = self.decoder(hidden_image)
            return hidden_image, reveal_image_1
        elif mode == 'encoder':
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            hidden_image = self.encoder(cover_image, secret_image_1)
            return hidden_image
        elif mode == 'decoder':
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = True

            reveal_image1 = self.decoder(hidden_image)
            return reveal_image1


prep_1 = PrepNetwork1()
hiding_network = HidingNetwork()

encoder = Encoder(prep_1, hiding_network)

reveal_1 = RevealNetwork1()


decoder = Decoder(reveal_1)

model = SteganoModel(encoder, decoder)
model.to(device)


def predict(model, cover, secret, mode, reveal_size=None):

    cover_image = cover
    cover_image = cover_image.to(device)
    secret_image_1 = secret
    secret_image_1 = secret_image_1.to(device)

    model.eval()
    if mode == 'decoder':

        reveal_image_1 = model(cover_image, cover_image, cover_image, mode)

        # dot_graph = torchviz.make_dot(model(cover_image,cover_image,cover_image,mode))
        # dot_graph.render("decoder.dot")
    elif mode == 'encoder':
        hidden_image = model(cover_image, secret_image_1, secret_image_1, mode)
        # dot_graph = torchviz.make_dot(model(cover_image,secret_image_1,secret_image_1,mode))
        # dot_graph.render("encoder.dot")
    elif mode == "full":

        hidden_image, reveal_image_1 = model(
            cover_image, secret_image_1, secret_image_1, mode)
        # dot_graph = torchviz.make_dot(model(cover_image,secret_image_1,secret_image_1,mode))
        # dot_graph.render("full.dot")

        cover_image = cover_image * 255
        cover_image = cover_image.to(torch.device('cpu'))
        cover_image = cover_image.detach().to(torch.long)
        secret_image_1 = secret_image_1 * 255
        secret_image_1 = secret_image_1.to(torch.device('cpu'))
        secret_image_1 = secret_image_1.detach().to(torch.long)
    if mode == 'encoder' or mode == 'full':
        hidden_image[hidden_image > 1] = 1
        hidden_image = hidden_image * 255
        hidden_image = hidden_image.to(torch.device('cpu'))
        hidden_image = hidden_image.detach().to(torch.long)
        h = hidden_image[0].permute(1, 2, 0).numpy()
        h = h.astype(np.uint8)

        if mode == 'encoder':
            return h
    if mode == 'decoder' or mode == 'full':
        reveal_image_1[reveal_image_1 > 1] = 1
        reveal_image_1 = reveal_image_1 * 255
        reveal_image_1 = reveal_image_1.to(torch.device('cpu'))
        reveal_image_1 = reveal_image_1.detach().to(torch.long)
        transform = torchvision.transforms.Resize(reveal_size)
        reveal_image_1 = transform(reveal_image_1)
        r = reveal_image_1[0].permute(1, 2, 0).numpy()
        r = r.astype(np.uint8)

        if mode == 'decoder':
            return r
    return {
        'cover_image_grid': cover_image[0].permute(1, 2, 0).numpy(),
        'secret_image_1_grid': secret_image_1[0].permute(1, 2, 0).numpy(),
        'hidden_image_grid': h,
        'reveal_image_1_grid': r,
    }
