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


class MyOption:
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
        self.input_size = 256
        self.resize_scale = 286
        self.train_epoch = 200
        self.decay_epoch = 100

        # mnist
        # self.input_size = 28
        # self.resize_scale = 32
        # self.train_epoch = 200
        # self.decay_epoch = 120

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


opt = MyOption()


# device = torch.device("cuda:0,1,2" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[Running on device]: {str(device).upper()}")


# root = opt.output_path  opt.dataset + '_' + opt.save_root + '/' # horse2zebra_results/
res_path = os.path.join(opt.output_path, f"{opt.dataset}_{opt.save_root}") # output/horse2zebra_results/
print(res_path)
model = opt.dataset + '_'

if not os.path.isdir(res_path):
    os.mkdir(res_path)


if not os.path.isdir(os.path.join(res_path, "test_results")):
    os.mkdir(os.path.join(res_path, "test_results"))

if not os.path.isdir(os.path.join(res_path, "test_results", "AtoB")):
    os.mkdir(os.path.join(res_path, "test_results", "AtoB"))

if not os.path.isdir(os.path.join(res_path, "test_results", "BtoA")):
    os.mkdir(os.path.join(res_path, "test_results", "BtoA"))

# data_loader
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

"""
train_loader_A path: data/horse2zebra/trainA
train_loader_B path: data/horse2zebra/trainB
test_loader_A path: data/horse2zebra/testA
test_loader_B path: data/horse2zebra/testBdl
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

G_A.to(device)
G_B.to(device)
D_A.to(device)
D_B.to(device)
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
BCE_loss = nn.BCELoss().to(device)
MSE_loss = nn.MSELoss().to(device)
L1_loss = nn.L1Loss().to(device)

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

        realA, realB = Variable(realA.to(device)), Variable(realB.to(device))

        # train generator G
        G_optimizer.zero_grad()

        # generate real A to fake B; D_A(G_A(A))
        fakeB = G_A(realA)
        D_A_result = D_A(fakeB)
        G_A_loss = MSE_loss(D_A_result, Variable(torch.ones(D_A_result.size()).to(device)))

        # reconstruct fake B to rec A; G_B(G_A(A))
        recA = G_B(fakeB)
        A_cycle_loss = L1_loss(recA, realA) * opt.lambdaA

        # generate real B to fake A; D_A(G_B(B))
        fakeA = G_B(realB)
        D_B_result = D_B(fakeA)
        G_B_loss = MSE_loss(D_B_result, Variable(torch.ones(D_B_result.size()).to(device)))

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
        D_A_real_loss = MSE_loss(D_A_real, Variable(torch.ones(D_A_real.size()).to(device)))

        # fakeB = fakeB_store.query(fakeB.data)
        fakeB = fakeB_store.query(fakeB)
        D_A_fake = D_A(fakeB)
        D_A_fake_loss = MSE_loss(D_A_fake, Variable(torch.zeros(D_A_fake.size()).to(device)))

        D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5
        D_A_loss.backward()
        D_A_optimizer.step()

        train_hist['D_A_losses'].append(D_A_loss.item())
        D_A_losses.append(D_A_loss.item())

        # train discriminator D_B
        D_B_optimizer.zero_grad()

        D_B_real = D_B(realA)
        D_B_real_loss = MSE_loss(D_B_real, Variable(torch.ones(D_B_real.size()).to(device)))

        # fakeA = fakeA_store.query(fakeA.data)
        fakeA = fakeA_store.query(fakeA)
        D_B_fake = D_B(fakeA)
        D_B_fake_loss = MSE_loss(D_B_fake, Variable(torch.zeros(D_B_fake.size()).to(device)))

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

    logger.info('[%d/%d] - ptime: %.2f, loss_D_A: %.3f, loss_D_B: %.3f, loss_G_A: %.3f, loss_G_B: %.3f, loss_A_cycle: %.3f, loss_B_cycle: %.3f' % (
        (epoch + 1), opt.train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_A_losses)),
        torch.mean(torch.FloatTensor(D_B_losses)), torch.mean(torch.FloatTensor(G_A_losses)),
        torch.mean(torch.FloatTensor(G_B_losses)), torch.mean(torch.FloatTensor(A_cycle_losses)),
        torch.mean(torch.FloatTensor(B_cycle_losses))))

    # Test network training result
    idxA = 0
    train_path_AtoB = os.path.join(res_path, "train_results", "AtoB")
    if epoch % 20 == 0:
        for realA, _ in train_loader_A:
            if idxA > 1:
                break
            path = os.path.join(train_path_AtoB, f"{epoch}_{idxA}_input.png")
            plt.imsave(path, (realA[0].numpy().transpose(1, 2, 0) + 1) / 2)
            realA = Variable(realA.to(device), volatile=True)
            genB = G_A(realA)
            path = os.path.join(train_path_AtoB, f"{epoch}_{idxA}_output.png")
            plt.imsave(path, (genB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            recA = G_B(genB)
            path = os.path.join(train_path_AtoB, f"{epoch}_{idxA}_recon.png")
            plt.imsave(path, (recA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            idxA += 1

        idxB = 0
        train_path_BtoA = os.path.join(res_path, "train_results", "BtoA")
        for realB, _ in train_loader_B:
            if idxB > 1:
                break

            path = os.path.join(train_path_BtoA, f"{epoch}_{idxB}_input.png")
            plt.imsave(path, (realB[0].numpy().transpose(1, 2, 0) + 1) / 2)
            realB = Variable(realB.to(device), volatile=True)
            genA = G_B(realB)
            path = os.path.join(train_path_BtoA, f"{epoch}_{idxB}_output.png")
            plt.imsave(path, (genA[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            recB = G_A(genA)
            path = os.path.join(train_path_BtoA, f"{epoch}_{idxB}_recon.png")
            plt.imsave(path, (recB[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            idxB += 1

    print(f"============== Epoch:{epoch} done! Cost time: {time.time() - epoch_time} Timestamp: {epoch_time} ====================\n\n")
    epoch_time = time.time()

    if epoch % 20 == 0:
        #         G_A_state = {'model': G_A.state_dict(), 'optimizer': G_optimizer.state_dict(), 'epoch': epoch}
        #         torch.save(G_A_state, os.path.join(res_path, f"{model}generatorA_param{epoch}.pkl"))

        #         G_B_state = {'model': G_B.state_dict(), 'optimizer': G_optimizer.state_dict(), 'epoch': epoch}
        #         torch.save(G_B_state, os.path.join(res_path, f"{model}generatorB_param{epoch}.pkl"))

        #         D_A_state = {'model': D_A.state_dict(), 'optimizer': D_A_optimizer.state_dict(), 'epoch': epoch}
        #         torch.save(D_A_state, os.path.join(res_path, f"{model}discriminatorA_param{epoch}.pkl"))

        #         D_B_state = {'model': D_B.state_dict(), 'optimizer': D_B_optimizer.state_dict(), 'epoch': epoch}
        #         torch.save(D_B_state, os.path.join(res_path, f"{model}discriminatorB_param{epoch}.pkl"))

        torch.save(G_A.state_dict(), os.path.join(res_path, f"{model}generatorA_param{epoch}.pkl"))
        torch.save(G_B.state_dict(), os.path.join(res_path, f"{model}generatorB_param{epoch}.pkl"))
        torch.save(D_A.state_dict(), os.path.join(res_path, f"{model}discriminatorA_param{epoch}.pkl"))
        torch.save(D_B.state_dict(), os.path.join(res_path, f"{model}discriminatorB_param{epoch}.pkl"))



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
