import torch
import torch.nn as nn
# from torchsummary import summary


hidden_dim_1 = 64
hidden_dim_2 = 256
hidden_dim_3 = 128



SharedConv = None
   


def make_nn_block(input_channels, output_channels, kernel_size=1, stride=1, padding=0):

    return nn.Sequential(
        nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        nn.BatchNorm1d(output_channels),                         
        nn.LeakyReLU(0.2, inplace=True),
    )

class MyModel(nn.Module):
    def __init__(self, num_feature=6):
        super(MyModel, self).__init__()
        
        self.c1 = make_nn_block(num_feature, hidden_dim_1 , kernel_size=1)              

        self.c2 = make_nn_block(hidden_dim_1, hidden_dim_2)
        self.c3 = make_nn_block(hidden_dim_2, 2 * hidden_dim_2) 
        self.c4 = make_nn_block(2 * hidden_dim_2, 4 * hidden_dim_2)
        self.fc = nn.Sequential(			 
            # nn.Linear(2 * hidden_dim_2 + hidden_dim_1, hidden_dim_3),  
            nn.Conv1d(hidden_dim_1+ 4 * hidden_dim_2, hidden_dim_3, 1),
            nn.BatchNorm1d(hidden_dim_3),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5),
            nn.Conv1d(hidden_dim_3, 1, 1),)
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        num_atoms = x.size()[2]
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x3 = self.c4(x3)
        hidden_feature = x3.size()[1]
        # x_global = torch.mean(x3, axis=-1, keepdim=True)
        x_global = torch.max(x3, axis=-1, keepdim=True)[0]
        x_global = x_global.view(-1, hidden_feature, 1).repeat(1, 1, num_atoms)
        # x = torch.max(x, axis=-1)[0]
        x = torch.cat((x1, x_global),axis=1)

        x = self.fc(x)
        # result = torch.max(x, dim=-1)[0].reshape(-1, 1)
        result = torch.mean(x, dim=-1).reshape(-1, 1)

        return result



class Rmodel(nn.Module):
    def __init__(self, num_feature=5):
        super(Rmodel, self).__init__()
        global SharedConv
        if SharedConv == None:
            # SharedConv = make_nn_block(num_feature, hidden_dim_1 , kernel_size=3, stride=1, padding=1)
            SharedConv = make_nn_block(num_feature, hidden_dim_1 , kernel_size=1)
        self.c1 = SharedConv               
        self.c2 = make_nn_block(hidden_dim_1, hidden_dim_2,3,1,1)       
        self.fc = nn.Sequential(
            nn.Flatten(),			 
            nn.Linear(hidden_dim_2 + num_feature, hidden_dim_3),  
            nn.BatchNorm1d(hidden_dim_3),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(hidden_dim_3, 1),)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x1 = self.c1(x)
        x1 = self.c2(x1)
        x = torch.cat((x,x1),axis=1)
        x = torch.mean(x, axis=-1)
        # x = torch.max(x, axis=-1)[0]
        result = self.fc(x) 
        return result     

class Generator(nn.Module):
    """
    The output shape is (batch_size, num_atoms, 5)
    """
    def __init__(self, z_dim=10, num_atoms=24, hidden_dim=512):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.num_atoms = num_atoms

        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, hidden_dim // 2),
            self.make_gen_block(hidden_dim // 2, self.num_atoms * 5, True),
        )

    @staticmethod
    def make_gen_block(input_channels, output_channels, final_layer=False):
        """ 
            Function to return a sequnce of operations.Especially, the fianl layer just has a 
            transposed convolution and a activation function.
        """
        if not final_layer:			
            """ return nn.Sequential(
                nn.ConvTranspose1d(input_channels, output_channels, kernel_size, stride),
                #! convtransposed1d output shape: (N-1) * s - 2 * padding + k
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(output_channels),   
            ) """
            return nn.Sequential(
                        nn.Linear(input_channels, output_channels, bias=True),
                        nn.ReLU(inplace=True),
                    )
        else:
            return nn.Sequential(
                        nn.Linear(input_channels, output_channels, bias=True),
                        # nn.Tanh(),
                    )


    def forward(self, x):
        output = self.gen(x)
        output = output.view(-1, self.num_atoms, 5)
        return output
    

class Discriminator(nn.Module):
    """
    The output shape is (batch_size, num_atoms, 1)
    """
    def __init__(self, num_feature, im_chan=1):
        super(Discriminator, self).__init__()
        
        global SharedConv
        if SharedConv == None:    
            # SharedConv = make_nn_block(num_feature, hidden_dim_1 , kernel_size=3, stride=1, padding=1)
            SharedConv = make_nn_block(num_feature, hidden_dim_1 , kernel_size=1)
        self.d1 = SharedConv

        self.d2 = nn.Sequential(
                    nn.Conv1d(hidden_dim_1, 64, 1),
                    nn.InstanceNorm1d(64),                         
                    nn.LeakyReLU(inplace=True),
                )
        
        
        self.d3 = nn.Linear(64, im_chan, bias=True)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        a1 = self.d1(x)
        a2 = self.d2(a1)
        a2 = torch.mean(a2, axis=-1)
        a3 = self.d3(a2)
        return a3



if __name__ == '__main__':
    model = MyModel()
    # summary(model, (1, 24, 5))


    

'''
#? 1.兜兜转转到最后还是借鉴了pointnet的想法，进行了permute
#? 2.将每个点的特征的进行升维，然后对于DFT计算的原子个数进行max操作(相当于maxpool)，这样有个一个好处就是
#? 模型将不再受限于输入计算原子个数的规定。将会有更好的泛化能力。
#? 3.还有一点就是conv1d没有达到kernel size，全部都是1，连padding都没有。正常理解就是只考虑了局部信息

#! shardcovb必须要加上padding，不加完全underfit
#! 使用batch normalization过拟合非常严重。加上了 L2正则化，没有多大的帮助
dropout
目前效果的最好是conv层使用Bn，而全连接层使用dropout
接下来考虑的是，使用共享卷积。第一层不使用BN，在接下来使用。
#! 第一层的BN十分关键，极其影响性能。
记录模型结构
#$ top1
#= covn1d    25，64                     
bn
LeakyReLu
#= conv1d     64， 256     
bn
LReLu 
fc            256， 4
LReLU
dropout 
fc             4， 1
#$ top2是在1的基础上将128改为了256，非常明显的过拟合。感觉提升不大

0.9
933 158 26 lr:1e-5 bdecay:0.035
 

0.9的   163   237    8       epoch 1200                          weight_dacay=0.00004
0.7         200  64               500
0.5    186, 247, 51, 'lr': 1.711971181502245e-05     epoch: 1000
#! 采用early stopping。在过拟合前停止
#! 使用top1的模型在 70:30 和 90:10 数据集上跑出非常棒的结果
#? 说明原作者的模型存在缺陷。

使用mean自然而然的不能提高模型的效能，反而退化到和作者的性能差不多了，甚至更差。
{'fc1_dim': 504, 'fc2_dim': 210, 'fc3_dim': 17, 'lr': 4.5748519056306125e-05}


#! 已经发现是kernel size的问题，

0.5 + kernel size = 1 
. 
 {'fc1_dim': 682, 'fc2_dim': 194, 'fc3_dim': 11, 'lr': 9.286720099093146e-05, 'weight_decay': 0.058379283861116835}.
'''