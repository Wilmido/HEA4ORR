#REF: https://finthon.com/python-genetic-algorithm/


import numpy as np
from operator import itemgetter
from tqdm import tqdm
from utils.torch_utils import get_fitness
from multiprocessing.dummy import Pool as ThreadPool

# 首先我们构建一个叫chrom的类
class Chrom:
    def __init__(self, **data):
        self.__dict__.update(data)
        self.length = len(data['data'])  # length of chromosome

#$ 创建一个GA类，包含算法的所有操作
class GA:
    def __init__(self, args, datas, outputs, max_feature=90):
        '''
        Initialize the pop of GA algorithom and evaluate the pop by computing its' fitness value.
        The data structure of pop is composed of several individuals which has the form like that:
        {'chrom': data,  'fitness':}
        '''
        self.args = args
        self.cxpb = args.CXPB
        self.mutpb = args.MUTPB
        self.ngen = args.NGEN
        self.datas = datas
        self.outputs = outputs
        self.popsize = args.popsize
        self.max_feature = max_feature
        self.history = []
        self.pop = self.initilization_population()
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population

        assert self.popsize % 2 == 0

    def initilization_population(self):
        pop = []
        for _ in tqdm(range(self.popsize)):
            chrominfo = (np.random.randint(0, 2, self.max_feature))  # initialise popluation
            
            #! Force the origin 6 descriptors to 1 at the begining stage
            for i in range(6):
                chrominfo[i]=1
            fitness = self.evaluate(chrominfo)  # evaluate each chromosome
            pop.append({'Chrom': Chrom(data=chrominfo), 'fitness': fitness})  # store the chromosome and its fitness        
        return pop




    def evaluate(self, chrominfo):
        '''
        fitness function to evaluate individual(Chrom)'''
        inputs = chrominfo * self.datas
        inputs = inputs.loc[:, (inputs != 0).any(axis=0)]

        inputs = np.array(inputs)

        fitness = get_fitness(inputs, self.outputs, self.args)
        return fitness


    @staticmethod
    def selectBest(pop):
        '''
        select the best individual(chromosome) from population
        '''
        s_inds = sorted(pop,key=itemgetter('fitness'), reverse=True)   # from large to small, return a pop
        return s_inds[0]
 
    
    @staticmethod
    def selection(individuals, k):
        '''
        select individuals from population. The better individual is,the greater probability to be choosen it has
        '''
        s_inds = sorted(individuals,key=itemgetter('fitness'), reverse=True)
        p =  [np.exp(ind['fitness'].detach().cpu().numpy()) for ind in s_inds]
        weight = p / np.sum(p)
        chosen=[]
        
        for _ in range(k):
            chosen.append(np.random.choice(s_inds,p=weight))
        
        chosen = sorted(chosen, key=itemgetter('fitness'),reverse=False)      #最后按照适应度从小到大的顺序排列。

        return chosen


    @staticmethod
    def crossover(offspring):
        '''Crossover operation        '''
        length = len(offspring[0]['Chrom'].data)
        geninfo1 = offspring[0]['Chrom'].data
        geninfo2 = offspring[1]['Chrom'].data

        if length == 1:
            pos1 = 1
            pos2 = 1
        else:
            pos1 = np.random.randint(1,length)    # select a position in the range of [0,dim-1]
            pos2 = np.random.randint(1,length)
        
        child1 = Chrom(data=[])
        child2 = Chrom(data=[])
        temp1 = []
        temp2 = []
        for i in range(length):
            if min(pos1,pos2) <= i <= max(pos1,pos2):
                temp2.append(geninfo2[i])
                temp1.append(geninfo1[i])
            else:
                temp2.append(geninfo1[i])
                temp1.append(geninfo2[i])
        child1.data = temp1
        child2.data = temp2
        return child1, child2


    @staticmethod
    def mutation(chrom):

        length = len(chrom.data)
        
        if length == 1:
            pos = 0
        else:
            pos = np.random.randint(0, length)
        
        chrom.data[pos] = np.invert(chrom.data[pos])
        return chrom

    def run(self,mylogger):
        popsize = self.popsize
        epochs = self.ngen
        mylogger.logger.info('='*30)
        mylogger.logger.info("======START OF EVOLUTION======")
        mylogger.logger.info('='*30)
        # begin the evolution
        for g in tqdm(range(epochs)):
            # Apply selection baesd on their converted fitness

            selectpop = self.selection(self.pop, popsize)
            child = []

            while len(child) != popsize:  
                # Apply crossover and mutation on the offspring
                offspring = [selectpop.pop() for _ in range(2)]
                
                if np.random.random() < self.cxpb:        # cross
                    crossoff1, crossoff2 = self.crossover(offspring)
                    if np.random.random() < self.mutpb:                # mutation
                        muteoff1 = self.mutation(crossoff1)
                        muteoff2 = self.mutation(crossoff2)
                        fit_muteoff1 = self.evaluate(muteoff1.data)
                        fit_muteoff2 = self.evaluate(muteoff2.data)
                        child.append({'Chrom':muteoff1, 'fitness':fit_muteoff1})
                        child.append({'Chrom':muteoff2, 'fitness':fit_muteoff2})
                    else:
                        fit_crossoff1 = self.evaluate(crossoff1.data)
                        fit_crossoff2 = self.evaluate(crossoff2.data)
                        child.append({'Chrom':crossoff1, 'fitness':fit_crossoff1})
                        child.append({'Chrom':crossoff2, 'fitness':fit_crossoff2})
                else:
                    child.extend(offspring)

            # The population is entirely replaced by the offspring
            self.pop = child

            fits = [ind['fitness'] for ind in self.pop]

            best_ind = self.selectBest(self.pop)

            if best_ind['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = best_ind
            
            self.history.append(best_ind['fitness'].clone().detach().cpu().numpy())

            if (g+1) % 5 == 0:           
                np.save(f'result/{g}.npy', self.history)  
                mylogger.logger.info(f"###########Generation {g+1}###########")                
                mylogger.logger.info(f"Best individual found is {self.bestindividual['Chrom'].data}\n{self.bestindividual['fitness']}")
                mylogger.logger.info(f"Max fitness of current pop: {max(fits)}")
        
        mylogger.logger.info(f"Best individual is {self.bestindividual['Chrom'].data}\n{self.bestindividual['fitness']}")
        mylogger.logger.info("End of evolution")

        
if __name__ == '__main__':
    pass        
                
                