import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns	
    
# from scipy.stats import pearsonr

def feature_selection(data):
    data = np.array(data,dtype=float).reshape(-1,9)
    data = pd.DataFrame(data)    
    # data.drop([data.columns[[1, 6, 8]]], axis=1, inplace=True)
    data.drop([1,6,8],axis=1,inplace=True)
    # drop columns  "group","VEC", "AN"
    data = np.array(data,dtype=float)
    return data

if __name__ == '__main__': 
    data = pd.read_csv('../data/my_data.csv')   
    # for col_name, c_data in new_data.iteritems():
    data = np.array(data)
    Adata = pd.DataFrame(data.reshape(-1,9),columns=['period' ,'group' ,r'$r$' ,'CN' ,'At site' ,r'$\chi$', 'VEC' ,'M','1/AN'])

    plt.figure()
    
    corr_values = Adata.corr() # pandas直接调用corr就能计算特征之间的相关系数
    sns.heatmap(corr_values, annot=True, vmax=1, square=True, cmap="Blues",fmt='.2f')
    plt.tight_layout()
    # plt.savefig('..\result\Pearson_value.pdf', format='pdf',bbox_inches='tight', dpi=1200)
    plt.savefig(r'..\result\Pearson_value.svg', bbox_inches='tight', dpi=1200)
    plt.show()
                                                       
							