import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, input_channel, activation="relu"):
        super(SelfAttention, self).__init__()
        self.chanel_in = input_channel
        self.activation = activation

        self.query_conv = nn.Conv2d(input_channel, input_channel // 8, 1)
        self.key_conv = nn.Conv2d(input_channel, input_channel // 8, 1)
        self.value_conv = nn.Conv2d(input_channel, input_channel, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        attention_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N) # Q
        attention_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)  # K
        energy = torch.bmm(attention_query, attention_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        attention_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(attention_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out


class generator(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=32, nb=6):
        super(generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = nb
        self.conv1 = nn.Conv2d(input_nc, ngf, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block(ngf * 4, 3, 1, 1))
            self.resnet_blocks[i].weight_init(0, 0.02)

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv1 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm = nn.InstanceNorm2d(ngf * 2)
        self.deconv2 = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm = nn.InstanceNorm2d(ngf)
        self.deconv3 = nn.Conv2d(ngf, output_nc, 7, 1, 0)

        self.attn = SelfAttention(128, 'relu')

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.attn(x)
        x = self.resnet_blocks(x)
        x = F.relu(self.deconv1_norm(self.deconv1(x)))
        x = F.relu(self.deconv2_norm(self.deconv2(x)))
        x = F.pad(x, (3, 3, 3, 3), 'reflect')
        o = F.tanh(self.deconv3(x))

        return o


class discriminator(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ndf=64):
        super(discriminator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ndf = ndf
        self.conv1 = nn.Conv2d(input_nc, ndf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)
        self.conv4_norm = nn.InstanceNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, output_nc, 4, 1, 1)

        self.attn = SelfAttention(128, 'relu')

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_norm(self.conv2(x)), 0.2)
        x = self.attn(x)
        x = F.leaky_relu(self.conv3_norm(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_norm(self.conv4(x)), 0.2)
        x = self.conv5(x)

        return x


# resnet block with reflect padding
class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input + x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
