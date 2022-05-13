# Exploring the representation of high-entropy alloys for screening electrocatalysis of oxygen reduction reaction via feature engineering

This is repository for high entropy alloys(HEAs) experiments for My Graduation Project "基于特征工程探究高熵合金表达在氧还原电催化剂筛选的应用"

A regression model is proposed to predict the *OH adsorption energy of HEAs(high-entropy alloys), which can perfectly handle the problem of "input disorder" and has a excellent performance that the mean absolute error is within 0.038 eV compared with traditional calculations.Moreover, Feature engineering is used to data augment, and shapley value is used for analysing the feature selected by genetic algorithm. It is worth noting that the absorbed atoms’ molar mass and coordination number of atoms constituting the HEAs make great contributions to the prediction of the model. At last, WGAN-GP(Wasserstein GAN using gradient penalty) is used to generate HEAs environments and compositions.

Except for predicting adsorption energy of HEAs, this method can also be used for any other multiatomic systems which are similarly constrained by datasets shortages.