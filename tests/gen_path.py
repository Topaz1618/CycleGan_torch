import os

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



print(os.path.join("data", opt.dataset), f"{opt.train_subfolder}A", opt.batch_size)

print(opt.dataset + '_results/test_results/AtoB/' + str(1) + '_input.png')

print(os.path.join(f"{opt.dataset}_results", "test_results", "AtoB", f"{str(1)}_input.png"))

res_path = os.path.join(opt.output_path, f"{opt.dataset}_{opt.save_root}")
model = opt.dataset + '_'
print(os.path.join(res_path, f"{model}generatorB_param.pkl"))