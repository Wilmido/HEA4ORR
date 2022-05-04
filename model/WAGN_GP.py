import torch
import numpy as np
from random import shuffle
from model.networks import Generator, Discriminator



def elem2atom(data, num_feature, coord_nums_dict, element_dict):
    num_batch, num_atoms, _ = data.shape 

    envs = []
    new_data = np.zeros((num_batch, num_atoms, num_feature))
    for i in range(num_batch):
        #= random choose a enviorment for hea
        ind = np.random.randint(len(coord_nums_dict))
        coor_nums = list(coord_nums_dict.values())[ind]
        env = list(coord_nums_dict.keys())[ind]
        shuffle(coor_nums)

        #= create a list of sites for atoms while only two sites are active
        at_site_list = [0 for _ in range(num_atoms)]
        at_site_list[0], at_site_list[1] = 1, 1
        shuffle(at_site_list)

        for j in range(num_atoms):
  
            new_data[i][j] = element_dict[np.argmax(data[i][j])]

            if j < len(coor_nums):
                new_data[i][j][2] = coor_nums[j]
            else:
                new_data[i][j][2] = 0

            new_data[i][j][3] = at_site_list[j]

        envs.append(env)

    return new_data, envs



class WGAN_GP(object):
    def __init__(self, num_feature, args):
        self.num_feature = num_feature

        self.z_dim = args.z_dim
        self.G = self.get_cuda(Generator(self.z_dim))
        self.D = self.get_cuda(Discriminator(self.num_feature))
        
        self.lr = args.learning_rate
        self.b1 = 0.5
        self.b2 = 0.999        
        self.C_LAMBDA = args.c_lambda
        self.g_optimizer = torch.optim.Adam(self.G.parameters(),  lr=self.lr, betas=(self.b1, self.b2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(),  lr=self.lr / 4, betas=(self.b1, self.b2))


    def train_discriminator(self, real, coord_nums_dict, element_dict, iters):
        mean_iteration_disc_loss = 0
        for _ in range(iters):
            real =self.get_cuda(real)
            cur_batch_size = len(real)

            self.d_optimizer.zero_grad()
            fake_noise = self.get_cuda(self.get_noise(cur_batch_size))
            
            
            fake = self.G(fake_noise)

            fake = fake.detach().cpu().numpy()
            fake, _ = elem2atom(fake, self.num_feature, coord_nums_dict, element_dict)
            fake = (fake - np.mean(fake, axis=1, keepdims=True)) / np.mean(fake, axis=1, keepdims=True)
            fake = self.get_cuda(torch.tensor(fake, dtype=torch.float32))

            fake_pred = self.D(fake)
            real_pred = self.D(real)

            epsilon = torch.rand(cur_batch_size,  1,  1,  requires_grad=True)
            epsilon = self.get_cuda(epsilon)
            gp = self.gradient_penalty(self.D,  real,  fake,  epsilon)
            disc_loss = self.get_disc_loss(fake_pred, real_pred, gp, self.C_LAMBDA)       
            
            mean_iteration_disc_loss += disc_loss.item() / iters
            disc_loss.backward(retain_graph=True)
            self.d_optimizer.step()

        return mean_iteration_disc_loss

    def train_generator(self, real, coord_nums_dict, element_dict, iters=1):
        cur_batch_size = len(real)
        mean_iteration_gen_loss = 0
        for _ in range(iters):
            self.g_optimizer.zero_grad()
            fake_noise_2 = self.get_noise(cur_batch_size)
            fake_noise_2 = self.get_cuda(fake_noise_2)
            fake_2 = self.G(fake_noise_2)

            fake_2 = fake_2.detach().cpu().numpy()
            fake_2, _ = elem2atom(fake_2, self.num_feature, coord_nums_dict, element_dict)
            fake_2 = (fake_2 - np.mean(fake_2, axis=1, keepdims=True)) / np.mean(fake_2, axis=1, keepdims=True)
            fake_2 = self.get_cuda(torch.tensor(fake_2, dtype=torch.float32))
                        
            fake_pred2 = self.D(fake_2)

            gen_loss = self.get_gen_loss(fake_pred2)
            mean_iteration_gen_loss += gen_loss.item() / iters
            
            gen_loss.backward()
            self.g_optimizer.step()

        return mean_iteration_gen_loss


    def predict(self, coord_nums_dict, element_dict):
        z = self.get_noise()
        pred_atoms =  self.G(self.get_cuda(z))
        pred_atoms = pred_atoms.detach().cpu().numpy()
        result, envs = elem2atom(pred_atoms, self.num_feature, coord_nums_dict, element_dict)
        return result, envs



    def get_noise(self, n_sample=1):
        """
        Function for creating nosie vectors: Given the dimensions (n_sample,  z_dim)
        n_sample : the number of samples to generate,  a scalar
        z_dim: the dimension of the nosie vector, a scalar
        """
        return torch.randn(n_sample,  self.z_dim)


    @staticmethod
    def get_cuda(x):
        if torch.cuda.is_available():
            x = x.cuda()
        return x

   
    @staticmethod
    def gradient_penalty(discriminator, real, fake, epsilon):
        """
        Return the gradient penalty. Calculate the magnitude of gradient 
        and penalize the mean quadratic distance of each magnitude to 1
        Parameters:
            discriminator: the discriminator model
            real: the real data 
            fake: the fake data
            epsilon: a vetocr of uniformly random proportions of real/fake data    
        """
        mixed_data = real * epsilon + fake * (1 - epsilon)

        mixed_score = discriminator(mixed_data)

        gradient = torch.autograd.grad(inputs=mixed_data, outputs=mixed_score,
                        grad_outputs=torch.ones_like(mixed_score), 
                        create_graph=True, retain_graph=True)

        gradient = gradient[0]
        gradient = gradient.reshape(len(gradient), -1)

        gradient_norm = gradient.norm(2, dim=1)

        penalty = torch.mean((gradient_norm - 1) ** 2)      
        return penalty


    @staticmethod
    def get_gen_loss(fake_pred):
        """
        Return the loss of a generator.
        Parameters:
            fake_pred: the discriminator's score of the fake data
        Returns:
            gen_loss: a scalar loss value of the generator
        """
        gen_loss = -1. * torch.mean(fake_pred)
        return gen_loss


    @staticmethod
    def get_disc_loss(fake_pred, real_pred, gp, c_lambda):
        """
        Return the loss of a discriminator given the scores of the fake and real data
        the gradient penalty and gradient penalty weight.
        Pramters:
            fake_pred: the scores of the fake data
            real_pred: the scores of the real data
            gp: the unweighted gradient penalty
            c_lambda: the current weight of the gradient penalty
        Returns:
            disc_loss: a scalar loss value 
        """
        disc_loss = torch.mean(fake_pred) - torch.mean(real_pred) + c_lambda * gp
        return disc_loss


    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')


    def load_model(self, D_model_path, G_model_path):
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))
        


#! WGAN - Training discriminator more iterations than generator