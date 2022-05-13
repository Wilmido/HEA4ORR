# Exploring the representation of high-entropy alloys for screening electrocatalysis of oxygen reduction reaction via feature engineering

This is repository for high entropy alloys(HEAs) experiments for My Graduation Project "基于特征工程探究高熵合金表达在氧还原电催化剂筛选的应用"

## Introduction
A regression model is proposed to predict the *OH adsorption energy of HEAs(high-entropy alloys), which can perfectly handle the problem of "input disorder" and has a excellent performance that the mean absolute error is within 0.038 eV compared with traditional calculations.Moreover, Feature engineering is used to data augment, and shapley value is used for analysing the feature selected by genetic algorithm. It is worth noting that the absorbed atoms’ molar mass and coordination number of atoms constituting the HEAs make great contributions to the prediction of the model. At last, WGAN-GP(Wasserstein GAN using gradient penalty) is used to generate HEAs environments and compositions.

Except for predicting adsorption energy of HEAs, this method can also be used for any other multiatomic systems which are similarly constrained by datasets shortages.

## Dependecies
The prominent packages are:
* SHAP
* numpy
* pandas
* seaborn
* matplotlib
* scikit-learn
* pytorch 1.8.1

To install all the dependencies quickly and easily, you should use __pip__ install `requirements.txt`
```python
pip install -r requirements.txt
```

## Dataset
I build up my dataset based on [neural-network-design-of-HEA](https://github.com/jol-jol/neural-network-design-of-HEA), you can refer this repository for more infomation.

Because of the ownership of the dataset, this repository doesn't provide HEAs dataset! Therefore, you have to collect your own data!

The data structure is shown below.
||Atom|Ru|Rh|Pd|Ir|PT|
|--|--|--|--|--|--|--|
|A|Period|5|5|5|6|6|
||Group|8|9|10|9|10|
|B|Radius|1.338|1.345|1.375|1.357|1.387|
|C|CN|
|D|AtSite|
|E|pauling Negativity|2.20|2.28|2.20|2.20|2.28|
||VEC|8|9|10|9|10|
|F|M|101.07|102.906|106.42|192.2|195.08|
||atomic number|44|45|46|77|78|

where CN is coordination number, AtSite is active sites, and M is molar mass.The left features are descriptors we deisred, which are denoted as 'A, B, C, D, E, F' in above table.

You have to follow the coord_numbers <a href="HEA4ORR\data\coord_nums.csv">coord_nums</a> to fill in the blanks. 

If you use the dataset from __neural-network-design-of-HEA__, you should follow the steps below:

After build up the dataset with 9 features, you should use Pearson correlation coefficient to drop out highly related features to reduce copmutaion cost, run following code:
```
cd utils
python PearsonSelection.py
```
<a href="utils/PearsonSelection.py">`PearsonSelection.py`</a> use Pearson correlation coefficient to drop out highly related features.



The result will be like:
<img src="HEA4ORR/misc/Pearson_value.svg" width = "600" height = "450" alt="Pearson_value" />


## Get Started
The model can handle any numbers of atoms and is defined by the number of features which means it can also have no limitation in input dimension.

To train the model, you can simply use the following command, and you will get a checkpoint:
```
# training a model for downstream tasks
python K_fold.py
```

Obtaining the plot of __MAE__ and __RMSE__ compared with DFT-calculated adsorption energy

```
# training a model for downstream tasks, you need to update the checkpoint path first! 
python main.py
```
![6_500_plot](https://user-images.githubusercontent.com/71449089/163707875-e0862e04-4405-4b16-805e-d58973e49797.svg)


*Pretrained Models*
 ---
You can also just simply use the checkpoint I have provided in <a href="HEA4ORR/checkpoint">checkpoint/6_500epochs_5_model.pth</a>.

## T-SNE
Visualize the data, and the features processed by the model. 
```
python t_SNE.py
```

## Feature engineering
### Data augment
```
python data_augment.py
```
use $x^2$、$x^3$、$\sqrt{x}$、$log(1+x)$ basic functions to nonlinear feature transformation.And use $\frac{1}{x}$ for double feature number. At last, there is 90 features in datasets.

###Genetic algrithm


## Generate HEAs
You can switch the mode to choose whether to train the regression model. The result of loss plot demonstrates that the training process of GAN is not good :(
```
python Joint_training.py
```


## Reference
<div id="refer-anchor-1"></div>
https://github.com/jol-jol/neural-network-design-of-HEA
<div id="refer-anchor-2"></div>
https://github.com/Zeleni9/pytorch-wgan
<div id="refer-anchor-3"></div>
* https://arxiv.org/pdf/1612.00593.pdf