import os
import time
import pickle
import argparse
import itertools
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable


import network
import util
from utils.gan_logger import logger

# parser = argparse.ArgumentParser()
#
# parser.add_argument('--dataset', required=False, default='horse2zebra',  help='') # 数据集
# parser.add_argument('--train_subfolder', required=False, default='train',  help='')
# parser.add_argument('--test_subfolder', required=False, default='test',  help='')
# parser.add_argument('--input_ngc', type=int, default=3, help='input channel for generator')     # 生成器输入 channel
# parser.add_argument('--output_ngc', type=int, default=3, help='output channel for generator')   # 生成器输出 channel
# parser.add_argument('--input_ndc', type=int, default=3, help='input channel for discriminator')  # 鉴别器输入 channel
# parser.add_argument('--output_ndc', type=int, default=1, help='output channel for discriminator')   # 鉴别器输出 channel
# parser.add_argument('--batch_size', type=int, default=1, help='batch size')                         # batch size
# parser.add_argument('--ngf', type=int, default=32)                                                  # 生成器网络 out_channels
# parser.add_argument('--ndf', type=int, default=64)                                                  # 鉴别器网络 out_channels
# parser.add_argument('--nb', type=int, default=9, help='the number of resnet block layer for generator')     # 残块数
# parser.add_argument('--input_size', type=int, default=256, help='input size')                               # 图片输入大小 256 * 256
# parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')              #
# parser.add_argument('--crop', type=bool, default=True, help='random crop True or False')                # 根据这个参数判断是否 crop
# parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True or False')            # 数组左右翻转
# parser.add_argument('--train_epoch', type=int, default=200, help='train epochs num')                    # 训练多少次
# parser.add_argument('--decay_epoch', type=int, default=100, help='learning rate decay start epoch num')     # 多少轮之后开始衰减学习率
# parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')          # 鉴别器学习率
# parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')          # 生成器学习率
# parser.add_argument('--lambdaA', type=float, default=10, help='lambdaA for cycle loss')                 # 计算 cycle loss func 用的
# parser.add_argument('--lambdaB', type=float, default=10, help='lambdaB for cycle loss')                  # 计算 cycle loss func 用的
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')                # Adam 优化器用的
# parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')                  # Adam 优化器用的
# parser.add_argument('--save_root', required=False, default='results', help='results save path')            # 结果保存路径
# opt = parser.parse_args()


class Myoption():
    def __init__(self):
        # self.dataset = "horse2zebra"
        self.dataset = "mnist"
        self.train_subfolder = "train"
        self.test_subfolder = "test"
        self.input_ngc = 3
        self.output_ngc = 3
        self.input_ndc = 3
        self.output_ndc = 1
        self.batch_size = 32
        self.ngf = 32
        self.ndf = 64
        self.nb = 9
        # hz
        # self.input_size = 256
        # self.resize_scale = 286
        # self.train_epoch = 200
        # self.decay_epoch = 100

        # mnist
        self.input_size = 28
        self.resize_scale = 32
        self.train_epoch = 200
        self.decay_epoch = 120

        self.crop = True
        self.fliplr = True
        self.lrD = 0.0002
        self.lrG = 0.0002
        self.lambdaA = 10
        self.lambdaB = 10
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.save_root = "results"
        self.output_path = "output"


opt = Myoption()


# print('------------ Options -------------')
# for k, v in sorted(vars(opt).items()):
#     print('%s: %s' % (str(k), str(v)))
# print('-------------- End ----------------')
#
# results save path

""" 
Create path
"""

#
# USE_CUDA = torch.cuda.is_available()
# device = torch.device("cuda:0" if USE_CUDA else "cpu")
# model = torch.nn.DataParallel(model, device_ids=[0,1])#我这里只有两个显卡，主显卡0一定要有，如果想要换成别的显卡为主线卡，可以另外添加一行代码
# model['rep'].to(device)

# root = opt.output_path  opt.dataset + '_' + opt.save_root + '/' # horse2zebra_results/
res_path = os.path.join(opt.output_path, f"{opt.dataset}_{opt.save_root}") # output/horse2zebra_results/
model = opt.dataset + '_'

if not os.path.isdir(res_path):
    os.mkdir(res_path)

if not os.path.isdir(res_path + 'test_results'):
    os.mkdir(res_path + 'test_results')

if not os.path.isdir(res_path + 'test_results/AtoB'):
    os.mkdir(res_path + 'test_results/AtoB')

if not os.path.isdir(res_path + 'test_results/BtoA'):
    os.mkdir(res_path + 'test_results/BtoA')

# data_loader
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

"""
train_loader_A path: data/horse2zebra/trainA
train_loader_B path: data/horse2zebra/trainB
test_loader_A path: data/horse2zebra/testA
test_loader_B path: data/horse2zebra/testB
"""

dataset_path = os.path.join("data", opt.dataset)
train_loader_A = util.data_load(dataset_path, f"{opt.train_subfolder}A", transform, opt.batch_size, shuffle=True)
train_loader_B = util.data_load(dataset_path, f"{opt.train_subfolder}B", transform, opt.batch_size, shuffle=True)
test_loader_A = util.data_load(dataset_path, f"{opt.test_subfolder}A", transform, opt.batch_size, shuffle=False)
test_loader_B = util.data_load(dataset_path, f"{opt.test_subfolder}B", transform, opt.batch_size, shuffle=False)


# network
G_A = network.generator(opt.input_ngc, opt.output_ngc, opt.ngf, opt.nb)
G_B = network.generator(opt.input_ngc, opt.output_ngc, opt.ngf, opt.nb)
D_A = network.discriminator(opt.input_ndc, opt.output_ndc, opt.ndf)
D_B = network.discriminator(opt.input_ndc, opt.output_ndc, opt.ndf)

G_A.weight_init(mean=0.0, std=0.02)
G_B.weight_init(mean=0.0, std=0.02)
D_A.weight_init(mean=0.0, std=0.02)
D_B.weight_init(mean=0.0, std=0.02)

G_A.cuda()
G_B.cuda()
D_A.cuda()
D_B.cuda()
G_A.train()
G_B.train()
D_A.train()
D_B.train()

print('---------- Networks initialized -------------')
util.print_network(G_A)
util.print_network(G_B)
util.print_network(D_A)
util.print_network(D_B)
print('-----------------------------------------------')

# loss
BCE_loss = nn.BCELoss().cuda()
MSE_loss = nn.MSELoss().cuda()
L1_loss = nn.L1Loss().cuda()

# Adam optimizer
G_optimizer = optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
D_A_optimizer = optim.Adam(D_A.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
D_B_optimizer = optim.Adam(D_B.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))

# image store
# fakeA_store = util.image_store(50)
# fakeB_store = util.image_store(50)
fakeA_store = util.ImagePool(50)
fakeB_store = util.ImagePool(50)

train_hist = {}
train_hist['D_A_losses'] = []
train_hist['D_B_losses'] = []
train_hist['G_A_losses'] = []
train_hist['G_B_losses'] = []
train_hist['A_cycle_losses'] = []
train_hist['B_cycle_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()
epoch_time = time.time()


for epoch in range(opt.train_epoch):
    D_A_losses = []
    D_B_losses = []
    G_A_losses = []
    G_B_losses = []
    A_cycle_losses = []
    B_cycle_losses = []
    epoch_start_time = time.time()
    num_iter = 0
    idx = 0
    ts = time.time()

    if (epoch+1) > opt.decay_epoch:
        D_A_optimizer.param_groups[0]['lr'] -= opt.lrD / (opt.train_epoch - opt.decay_epoch)
        D_B_optimizer.param_groups[0]['lr'] -= opt.lrD / (opt.train_epoch - opt.decay_epoch)
        G_optimizer.param_groups[0]['lr'] -= opt.lrG / (opt.train_epoch - opt.decay_epoch)

    # Training
    for (realA, _), (realB, _) in zip(train_loader_A, train_loader_B):
        if opt.resize_scale:
            realA = util.imgs_resize(realA, opt.resize_scale)
            realB = util.imgs_resize(realB, opt.resize_scale)

        if opt.crop:
            realA = util.random_crop(realA, opt.input_size)
            realB = util.random_crop(realB, opt.input_size)

        if opt.fliplr:
            realA = util.random_fliplr(realA)
            realB = util.random_fliplr(realB)

        realA, realB = Variable(realA.cuda()), Variable(realB.cuda())

        # train generator G
        G_optimizer.zero_grad()

        # generate real A to fake B; D_A(G_A(A))
        fakeB = G_A(realA)
        D_A_result = D_A(fakeB)
        G_A_loss = MSE_loss(D_A_result, Variable(torch.ones(D_A_result.size()).cuda()))

        # reconstruct fake B to rec A; G_B(G_A(A))
        recA = G_B(fakeB)
        A_cycle_loss = L1_loss(recA, realA) * opt.lambdaA

        # generate real B to fake A; D_A(G_B(B))
        fakeA = G_B(realB)
        D_B_result = D_B(fakeA)
        G_B_loss = MSE_loss(D_B_result, Variable(torch.ones(D_B_result.size()).cuda()))

        # reconstruct fake A to rec B G_A(G_B(B))
        recB = G_A(fakeA)
        B_cycle_loss = L1_loss(recB, realB) * opt.lambdaB
        G_loss = G_A_loss + G_B_loss + A_cycle_loss + B_cycle_loss
        G_loss.backward()
        G_optimizer.step()

        train_hist['G_A_losses'].append(G_A_loss.item())
        train_hist['G_B_losses'].append(G_B_loss.item())
        train_hist['A_cycle_losses'].append(A_cycle_loss.item())
        train_hist['B_cycle_losses'].append(B_cycle_loss.item())
        G_A_losses.append(G_A_loss.item())
        G_B_losses.append(G_B_loss.item())
        A_cycle_losses.append(A_cycle_loss.item())
        B_cycle_losses.append(B_cycle_loss.item())

        # train discriminator D_A
        D_A_optimizer.zero_grad()

        D_A_real = D_A(realB)
        D_A_real_loss = MSE_loss(D_A_real, Variable(torch.ones(D_A_real.size()).cuda()))

        # fakeB = fakeB_store.query(fakeB.data)
        fakeB = fakeB_store.query(fakeB)
        D_A_fake = D_A(fakeB)
        D_A_fake_loss = MSE_loss(D_A_fake, Variable(torch.zeros(D_A_fake.size()).cuda()))

        D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
        D_A_loss.backward()
        D_A_optimizer.step()

        train_hist['D_A_losses'].append(D_A_loss.item())
        D_A_losses.append(D_A_loss.item())

        # train discriminator D_B
        D_B_optimizer.zero_grad()

        D_B_real = D_B(realA)
        D_B_real_loss = MSE_loss(D_B_real, Variable(torch.ones(D_B_real.size()).cuda()))

        # fakeA = fakeA_store.query(fakeA.data)
        fakeA = fakeA_store.query(fakeA)
        D_B_fake = D_B(fakeA)
        D_B_fake_loss = MSE_loss(D_B_fake, Variable(torch.zeros(D_B_fake.size()).cuda()))

        D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5
        D_B_loss.backward()
        D_B_optimizer.step()

        train_hist['D_B_losses'].append(D_B_loss.item())
        D_B_losses.append(D_B_loss.item())

        num_iter += 1
        print(f"============== Idx:{num_iter} done! Cost time: {time.time() - ts} Timestamp: {ts} ====================")
        ts = time.time()

    # Logging single epoch info
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    logger.info(f"[{epoch+1}/{opt.train_epoch}] Cost time: {round(per_epoch_ptime, 2)}"
                f"loss_D_A: {round(torch.mean(torch.FloatTensor(D_A_losses)), 3)} loss_D_B: {round(torch.mean(torch.FloatTensor(D_B_losses)), 2)}"
                f"loss_G_A: {round(torch.mean(torch.FloatTensor(G_A_losses)), 3)} loss_G_B: {round(torch.mean(torch.FloatTensor(G_B_losses)), 3)}"
                f"loss_A_cycle: {round(torch.mean(torch.FloatTensor(A_cycle_losses)), 3)} loss_B_cycle: {round(torch.mean(torch.FloatTensor(B_cycle_losses)), 3)}")

    # Test network training result
    idxA = 0
    train_path_AtoB = os.path.join(f"{opt.dataset}_results", "test_results", "AtoB")
    for realA, _ in train_loader_A:

        if idxA > 9:
            break
        path = os.path.join(train_path_AtoB, f"{str(idxA)}_input.png")
        plt.imsave(path, (realA[0].numpy().transpose(1, 2, 0) + 1) / 2)
        realA = Variable(realA.cuda(), volatile=True)
        genB = G_A(realA)
        path = os.path.join(train_path_AtoB, f"{str(idxA)}_output.png")
        plt.imsave(path, (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        recA = G_B(genB)
        path = os.path.join(train_path_AtoB, f"{str(idxA)}_recon.png")
        plt.imsave(path, (recA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        idxA += 1

    idxB = 0
    train_path_BtoA = os.path.join(f"{opt.dataset}_results", "test_results", "BtoA")
    for realB, _ in train_loader_B:
        if idxB > 9:
            break

        path = os.path.join(train_path_BtoA, f"{str(idxB)}_input.png")
        plt.imsave(path, (realB[0].numpy().transpose(1, 2, 0) + 1) / 2)
        realB = Variable(realB.cuda(), volatile=True)
        genA = G_B(realB)
        path = os.path.join(train_path_BtoA, f"{str(idxB)}_output.png")
        plt.imsave(path, (genA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        recB = G_A(genA)
        path = os.path.join(train_path_BtoA, f"{str(idxB)}_recon.png")
        plt.imsave(path, (recB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        idxB += 1

    print(f"============== Epoch:{epoch} done! Cost time: {time.time() - epoch_time} Timestamp: {epoch_time} ====================\n\n")
    epoch_time = time.time()


# Logging tranning info
end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
logger.info("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.train_epoch, total_ptime))
logger.info("Training finish!... save training results")


# Save Model
torch.save(G_A.state_dict(), os.path.join(res_path, f"{model}generatorA_param.pkl"))
torch.save(G_B.state_dict(), os.path.join(res_path, f"{model}generatorB_param.pkl"))
torch.save(D_A.state_dict(), os.path.join(res_path, f"{model}discriminatorA_param.pkl"))
torch.save(D_B.state_dict(), os.path.join(res_path, f"{model}discriminatorB_param.pkl"))

with open(os.path.join(res_path, f"{model}train_hist.pkl"), 'wb') as f:
    pickle.dump(train_hist, f)

util.show_train_hist(train_hist, save=True, path=os.path.join(res_path, f"{model}train_hist.png"))
