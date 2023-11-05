import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None  
    
df = pd.read_csv('Bengaluru_House_Data.csv')
df.drop(['area_type','society','balcony','availability'],axis='columns',inplace=True)
# print(df.isnull().sum())

df.dropna(inplace=True)
# print(df.isnull().sum())

# print(df['size'].unique())
#some value were 2bhk some were 3 bedroom so bringing consistency
df['bhk'] = df['size'].apply(lambda x : int(x.split(' ')[0]) )     
df.drop(['size'],axis='columns',inplace=True)

df[~df['total_sqft'].apply(is_float)]
df.total_sqft = df.total_sqft.apply(convert_sqft_to_num)
df = df[df.total_sqft.notnull()]

df['price_per_sqft'] = df['price']*100000/df['total_sqft']
df.to_csv("bhp.csv",index=False)

df.location = df.location.apply(lambda x: x.strip())
location_stats = df['location'].value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats<=10]
df.location = df.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

#OUTLIER DETECTION AND REMOVAL

df = df[~(df.total_sqft/df.bhk<300)]

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft) #standard deviation
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df = remove_pps_outliers(df)

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df,"Rajaji Nagar")

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df = remove_bhk_outliers(df)
df = df[df.bath<df.bhk+2]

#TRAINING MODEL AND SEARCHING BEST MODEL USING GRIDSEARCHCV

#FOR LOCATION COLUMN WE USE ONE HOT ENCODING(pd.get_dummies)
dummies = pd.get_dummies(df.location)
df = pd.concat([df,dummies.drop(['other'],axis='columns')],axis='columns')
df = df.drop(['location','price_per_sqft'],axis='columns')

X = df.drop(['price'],axis='columns')
y = df['price']
from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
print(model.score(x_test,y_test)) #0.930917614322498

#finding the best model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5,test_size=0.2)
scores = cross_val_score(model,X,y,cv=cv)
# print(scores)

#GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso

def find_model(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],  
                'positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

# print(find_model(X,y))
#                model  best_score                                        best_params
# 0  linear_regression    0.933080          {'fit_intercept': True, 'positive': True}
# 1              lasso    0.927368                {'alpha': 2, 'selection': 'cyclic'}
# 2      decision_tree    0.886678  {'criterion': 'friedman_mse', 'splitter': 'ran...

def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]

# print(predict_price('Indira Nagar',1000, 3, 3))

import pickle,json
with open('RealEstatePricePrediction.pickle','wb') as f:
    pickle.dump(model,f)

columns = {
    'data_columns' : [col.lower() for col in X.columns]
}

with open ("columns.json",'w') as f:
    f.write(json.dumps(columns))