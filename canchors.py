import os
import sys
import random
import numpy 
import multiprocessing 
import torch

import util 
import models 

class Model:
    def __init__(self, n_anchors, n_sv_types, z_dim, manual_seed=3060):
        if manual_seed is None:
            manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        
        def init_weights(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_uniform(m.weight, 1e-2)
                m.bias.data.fill_(0.01)

        D1 = models.Discriminator(input_dim=n_anchors, out_dim=1, out_dim2=n_sv_types)
        D2 = models.Discriminator(input_dim=n_anchors, out_dim=1, out_dim2=n_sv_types)

        G1 = models.Generator(z_dim=z_dim, input_dim=n_anchors)
        G2 = models.Generator(z_dim=z_dim, input_dim=n_anchors)
        E1 = models.Encoder(input_dim=n_anchors, z_dim=z_dim)
 
        init_weights(E1)
        init_weights(G1)
        init_weights(G2)
        init_weights(D1)
        init_weights(D2)

        self.E1 = E1
        self.G1 = G1
        self.G2 = G2
        self.D1 = D1
        self.D2 = D2

    def loadModel(self):
        model_E1_state = torch.load("./model/E1.pth")
        model_G1_state = torch.load("./model/G1.pth")
        model_G2_state = torch.load("./model/G2.pth")
        model_D1_state = torch.load("./model/D1.pth")
        model_D2_state = torch.load("./model/D2.pth")
        self.E1.load_state_dict(model_E1_state)
        self.G1.load_state_dict(model_G1_state)
        self.G2.load_state_dict(model_G2_state)
        self.D1.load_state_dict(model_D1_state)
        self.D2.load_state_dict(model_D2_state)

    def train(self, train_data, niter, lr_e, lr_g, lr_d, batch_size,
              lambda_adv, lambda_recon, lambda_encoding, betas,
              model_path='./model'
             ):
        # load data
        anchors1, sv_type1, anchors2, sv_type2 = train_data 
        anchors1_tensor = torch.Tensor(numpy.array(anchors1))
        anchors2_tensor = torch.Tensor(numpy.array(anchors2))
        sv_type1_tensor = torch.Tensor(numpy.array(sv_type1))
        sv_type2_tensor = torch.Tensor(numpy.array(sv_type2))


        dataset1 = torch.utils.data.TensorDataset(anchors1_tensor, sv_type1_tensor)
        dataset2 = torch.utils.data.TensorDataset(anchors2_tensor, sv_type2_tensor)

        dataset1_loader = torch.utils.data.DataLoader(dataset=dataset1,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     drop_last=True)

        dataset2_loader = torch.utils.data.DataLoader(dataset=dataset2,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     drop_last=True)

        dataset1_loader_it = iter(dataset1_loader)
        dataset2_loader_it = iter(dataset2_loader)
        # loss function
        recon_criterion = torch.nn.MSELoss()
        encoding_criterion = torch.nn.MSELoss()
        dis_criterion = torch.nn.BCELoss()
        aux_criterion = torch.nn.NLLLoss()
        # Optimizer
        optimizerD1 = torch.optim.Adam(self.D1.parameters(), lr=lr_d, betas=betas)
        optimizerD2 = torch.optim.Adam(self.D2.parameters(), lr=lr_d, betas=betas)
        optimizerG1 = torch.optim.Adam(self.G1.parameters(), lr=lr_g, betas=betas)
        optimizerG2 = torch.optim.Adam(self.G2.parameters(), lr=lr_g, betas=betas)
        #optimizerE1 = torch.optim.Adam(self.E1.parameters(), lr=lr_e, betas=betas)
        optimizerE1 = torch.optim.Adam(self.E1.parameters(), lr=lr_e)

        ones = torch.ones(batch_size, 1)
        zeros = torch.zeros(batch_size, 1)


        # Training
        for iteration in range(1, niter + 1):
            if iteration % 10000 == 0:
                for param_group in optimizerD1.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in optimizerD2.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in optimizerG1.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in optimizerG2.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                for param_group in optimizerE1.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9

            # Train discriminator
            D1_loss_item = 0.0
            D2_loss_item = 0.0

            try:
                real1, real_sv_type1 = next(dataset1_loader_it)
                real2, real_sv_type2 = next(dataset2_loader_it)
            except StopIteration:
                dataset1_loader_it, dataset2_loader_it = iter(dataset1_loader), iter(dataset2_loader)
                real1, real_sv_type1 = next(dataset1_loader_it)
                real2, real_sv_type2 = next(dataset2_loader_it)

            self.D1.zero_grad()
            self.D2.zero_grad()

            D1_real1_flag, D1_real1_sv_score = self.D1(real1) # real A
            D2_real2_flag, D2_real2_sv_score = self.D2(real2) # real B
            E1_real1 = self.E1(real1)
            G2E1_real1 = self.G2(E1_real1)

            E1_real2 = self.E1(real2)
            G1E1_real2 = self.G1(E1_real2)

            D1G1E1_real2_flag, D1G1E1_real2_sv_score = self.D1(G1E1_real2.detach()) # false A
            D2G2E1_real1_flag, D2G2E1_real1_sv_score = self.D2(G2E1_real1.detach()) # false B

            dis_D1_real = dis_criterion(D1_real1_flag, ones)
            aux_D1_real = aux_criterion(D1_real1_sv_score, real_sv_type1.reshape(1,64)[0].type(torch.LongTensor))
            D1_real = dis_D1_real + aux_D1_real
            dis_D1_fake = dis_criterion(D1G1E1_real2_flag, zeros)
            aux_D1_fake = aux_criterion(D1G1E1_real2_sv_score, real_sv_type2.reshape(1,64)[0].type(torch.LongTensor)) 
            D1_fake = dis_D1_fake + aux_D1_fake

            dis_D2_real = dis_criterion(D2_real2_flag, ones)
            aux_D2_real = aux_criterion(D2_real2_sv_score, real_sv_type2.reshape(1,64)[0].type(torch.LongTensor))

            D2_real = dis_D2_real + aux_D2_real
            dis_D2_fake = dis_criterion(D2G2E1_real1_flag, zeros) 
            aux_D2_fake = aux_criterion(D2G2E1_real1_sv_score, real_sv_type1.reshape(1,64)[0].type(torch.LongTensor)) 
            D2_fake = dis_D2_fake + aux_D2_fake

            D1_loss = D1_real + D1_fake
            D2_loss = D2_real + D2_fake

            D1_loss.backward()
            D2_loss.backward()
            optimizerD1.step()
            optimizerD2.step()

            D1_loss_item += D1_loss.item()
            D2_loss_item += D2_loss.item()

            # Train encoder and decoder
            try:
                real1, real_sv_type1 = next(dataset1_loader_it)
                real2, real_sv_type2 = next(dataset2_loader_it)
            except StopIteration:
                dataset1_loader_it, dataset2_loader_it = iter(dataset1_loader), iter(dataset2_loader)
                real1, real_sv_type1 = next(dataset1_loader_it)
                real2, real_sv_type2 = next(dataset2_loader_it)

            self.G1.zero_grad()
            self.G2.zero_grad()
            self.E1.zero_grad()

            E1_real1 = self.E1(real1)
            G1E1_real1 = self.G1(E1_real1)
            G2E1_real1 = self.G2(E1_real1)

            E1G1E1_real1 = self.E1(G1E1_real1)
            E1G2E1_real1 = self.E1(G2E1_real1)
            G1E1G2E1_real1 = self.G1(E1G2E1_real1)

            E1_real2 = self.E1(real2)
            G1E1_real2 = self.G1(E1_real2)
            G2E1_real2 = self.G2(E1_real2)
            E1G1E1_real2 = self.E1(G1E1_real2)
            E1G2E1_real2 = self.E1(G2E1_real2)
            G2E1G1E1_real2 = self.G2(E1G1E1_real2)

            D1G1E1_real1, D1G1E1_real1_sv_score = self.D1(G1E1_real1)
            D2G2E1_real1, D2G2E1_real1_sv_score = self.D2(G2E1_real1)
            D1G1E1_real2, D1G1E1_real2_sv_score = self.D1(G1E1_real2)
            D2G2E1_real2, D2G2E1_real2_sv_score = self.D2(G2E1_real2)
            D1G1E1G2E1_real1, D1G1E1G2E1_real1_sv_score = self.D1(G1E1G2E1_real1)
            D2G2E1G1E1_real2, D2G2E1G1E1_real2_sv_score = self.D2(G2E1G1E1_real2)

            # adversarial loss
            G11_adv_loss = dis_criterion(D1G1E1_real1, ones) + aux_criterion(D1G1E1_real1_sv_score, real_sv_type1.reshape(1,64)[0].type(torch.LongTensor))
            G21_adv_loss = dis_criterion(D1G1E1_real2, ones) + aux_criterion(D1G1E1_real2_sv_score, real_sv_type2.reshape(1,64)[0].type(torch.LongTensor))
            G121_adv_loss = dis_criterion(D1G1E1G2E1_real1, ones) + aux_criterion(D1G1E1G2E1_real1_sv_score, real_sv_type1.reshape(1,64)[0].type(torch.LongTensor))
        
            G22_adv_loss = dis_criterion(D2G2E1_real2, ones) + aux_criterion(D2G2E1_real2_sv_score, real_sv_type2.reshape(1,64)[0].type(torch.LongTensor))
            G12_adv_loss = dis_criterion(D2G2E1_real1, ones) + aux_criterion(D2G2E1_real1_sv_score, real_sv_type1.reshape(1,64)[0].type(torch.LongTensor))
            G212_adv_loss = dis_criterion(D2G2E1G1E1_real2, ones) + aux_criterion(D2G2E1G1E1_real2_sv_score, real_sv_type2.reshape(1,64)[0].type(torch.LongTensor))
        

            G1_adv_loss = G11_adv_loss + G21_adv_loss + G121_adv_loss
            G2_adv_loss = G22_adv_loss + G12_adv_loss + G212_adv_loss
            adv_loss = (G1_adv_loss + G2_adv_loss) * lambda_adv

            # reconstruction loss
            l_rec_G1E1_real1 = recon_criterion(G1E1_real1, real1)
            l_rec_G2E1_real2 = recon_criterion(G2E1_real2, real2)
            recon_loss = (l_rec_G1E1_real1 + l_rec_G2E1_real2) * lambda_recon

            # encoding loss
            tmp_E1_real1 = E1_real1.detach()
            tmp_E1_real2 = E1_real2.detach()
            l_encoding_G1E1_real1 = encoding_criterion(E1G1E1_real1, tmp_E1_real1)
            l_encoding_G2E1_real2 = encoding_criterion(E1G2E1_real2, tmp_E1_real2)
            l_encoding_G1E1_real2 = encoding_criterion(E1G1E1_real2, tmp_E1_real2)
            l_encoding_G2E1_real1 = encoding_criterion(E1G2E1_real1, tmp_E1_real1)
            encoding_loss = (l_encoding_G1E1_real1 + l_encoding_G2E1_real2 + l_encoding_G1E1_real2 + l_encoding_G2E1_real1) * lambda_encoding

            G_loss = adv_loss + recon_loss + encoding_loss

            G_loss.backward()

            optimizerG1.step()
            optimizerG2.step()
            optimizerE1.step()

            if iteration % 100 == 0:
                print ("iteration", iteration, D2_real)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.E1.state_dict(), os.path.join(model_path, 'E1.pth'))
        torch.save(self.G1.state_dict(), os.path.join(model_path, 'G1.pth'))
        torch.save(self.G2.state_dict(), os.path.join(model_path, 'G2.pth'))
        torch.save(self.D1.state_dict(), os.path.join(model_path, 'D1.pth'))
        torch.save(self.D2.state_dict(), os.path.join(model_path, 'D2.pth'))

    def predict(self, x1, x2):
        xt1 = torch.Tensor(x1)
        xt2 = torch.Tensor(x2)
        xg1 = self.G2(self.E1(xt1))
        torch.set_printoptions(threshold=100000)
        print (xg1[0])
        real1, svs1 = self.D1(xt1)
        real2, svs2 = self.D2(xg1)
        real3, svs3 = self.D2(xt2)
        #Get probabilities
        prob1 = torch.nn.functional.softmax(svs1, dim=-1) 
        prob2 = torch.nn.functional.softmax(svs2, dim=-1)
        prob3 = torch.nn.functional.softmax(svs3, dim=-1)
        __, predicted1 = torch.max(prob1, 1)
        __, predicted2 = torch.max(prob2, 1)
        __, predicted3 = torch.max(prob3, 1)
        f = open("predict", "w")
        for i in prob2:
            f.write (str(i)+"\n")
        f.close()
        print (prob2)
        return predicted1, predicted2, predicted3

def main():

    opt = {
        'train_input_file1':'data1/in_1',
        'train_input_file2':'data1/in_2',
        'train_output_file1':'data1/out',
        'train_output_file2':'data1/out',
        'test_input_file1':'data1/test_in_1',
        'test_input_file2':'data1/test_in_2',
        'test_output_file1':'data1/test_out',
        'test_output_file2':'data1/test_out',
        'manual_seed': 3060,
        'gan_loss': 'wgan'
    }

    config = {
        "batch_size": 64,
        "lambda_adv": 0.001,
        "lambda_encoding": 0.1,
        "lambda_l1_reg": 0,
        "lambda_recon": 1,
        "lr_d": 0.001,
        "lr_e": 0.0001,
        "lr_g": 0.001,
        "niter": 20000,
        "z_dim": 16,
        "betas": (0.5, 0.9)
    }

    train_data = util.loadInputOutputData(opt['train_input_file1'], opt['train_output_file1'],opt['train_input_file2'], opt['train_output_file2'])
    test_data = util.loadInputOutputData(opt['test_input_file1'], opt['test_output_file1'],opt['test_input_file2'], opt['test_output_file2'])

    model = Model(n_anchors=50, n_sv_types=2, z_dim=config['z_dim'])
    model.train(train_data = train_data, niter=config['niter'], lr_e=config['lr_e'], lr_g=config['lr_g'], lr_d=config['lr_d'], 
        batch_size=config['batch_size'], lambda_adv=config['lambda_adv'], lambda_recon=config['lambda_recon'], lambda_encoding=config['lambda_encoding'], betas=config['betas'])

    model.loadModel()

    #print(model.E1.layer1[0].weight)
    util.writeLayerWeightBias(model.E1.layer1, "w", "parms", 5, weight_name = "E1_weights", bias_name = "E1_biases")
    util.writeLayerWeightBias(model.G1.layer1, "a", "parms", 5, weight_name = "G1_weights", bias_name = "G1_biases")
    util.writeLayerWeightBias(model.G2.layer1, "a", "parms", 5, weight_name = "G2_weights", bias_name = "G2_biases")
    util.writeLayerWeightBias(model.D1.layer1, "a", "parms", 5, weight_name = "D1_weights", bias_name = "D1_biases")
    util.writeLayerWeightBias(model.D2.layer1, "a", "parms", 5, weight_name = "D2_weights", bias_name = "D2_biases")


    x1,y1,x2,y2 = test_data
    predicted1, predicted2, predicted3 = model.predict(x1, x2)
    print ("prediction acc = ", torch.sum(predicted1 == torch.Tensor(y1).reshape(1,y1.shape[0])[0]) / predicted1.shape[0])
    print ("prediction acc = ", torch.sum(predicted2 == torch.Tensor(y1).reshape(1,y1.shape[0])[0]) / predicted2.shape[0])
    print ("prediction acc = ", torch.sum(predicted3 == torch.Tensor(y1).reshape(1,y1.shape[0])[0]) / predicted3.shape[0])


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
