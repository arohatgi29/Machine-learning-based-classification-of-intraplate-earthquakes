# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
import pandas as pd
import seaborn as sns

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV,Lasso,LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# Load dataset
df1=pd.read_csv('raw_data.csv')
df1.info()
df1

#Filter dataset and choose features
df2=df1[(df1['Type of event'] == 'tect.')]
df2=df2[['Latitude','Longitude','Continent','Depth','Moment magnitude','Seismic moment','Tectonic association']]
df2.info()
df2

# Check for null values
print('There is duplicate row:',any(df2.duplicated()))
df3=df2.drop_duplicates()
print('There is duplicate row:',any(df3.duplicated()))
nan_index = df3.isna()
print(nan_index.sum(),'\n')
df3.info()
df3

#Linear Regression to find the missing Depth
#separate
df_notnull=df3.loc[df3['Depth'].notnull()]
df_isnull=df3.loc[df3['Depth'].isnull()]

#normalization
stand_scaler1=preprocessing.StandardScaler()
notnull_x=stand_scaler1.fit_transform(df_notnull[['Latitude','Longitude','Continent','Moment magnitude','Seismic moment']])
df_notnull_x=pd.DataFrame(notnull_x,columns=['Latitude','Longitude','Continent','Moment magnitude','Seismic moment'],index=df_notnull.index)

stand_scaler2=preprocessing.StandardScaler()
notnull_y=stand_scaler2.fit_transform(df_notnull[['Depth']])
df_notnull_y=pd.DataFrame(notnull_y,columns=['Depth'],index=df_notnull.index)

stand_scaler3=preprocessing.StandardScaler()
isnull_x=stand_scaler3.fit_transform(df_isnull[['Latitude','Longitude','Continent','Moment magnitude','Seismic moment']])
df_isnull_x=pd.DataFrame(isnull_x,columns=['Latitude','Longitude','Continent','Moment magnitude','Seismic moment'],index=df_isnull.index)

#prepare datasets for training and validation
x_train,x_vali,y_train,y_vali=train_test_split(df_notnull_x,df_notnull_y,test_size=0.25,random_state=1234)

#df_notnull[['Latitude','Longitude','Continent','Moment magnitude','Seismic moment']].head()
#df_notnull[['Depth']].head()
#df_isnull[['Latitude','Longitude','Continent','Moment magnitude','Seismic moment']].head()

# Build linear regression models (Linear Regression, Ridge Regression and Lasso Regression) to predict 'Depth'.
#store model parameters
coef=pd.DataFrame(index=x_train.columns)
print('Please wait for a few seconds...','\n')

#Linear Regression
linear=LinearRegression()
linear.fit(x_train,y_train)
coef['Linear Regression']=linear.coef_[0]
print(linear.coef_,'\n')

#Ridge Regression
Lambdas=np.logspace(-5,2,200)
ridge_cv=RidgeCV(alphas=Lambdas,scoring='neg_mean_squared_error',cv=10)
ridge_cv.fit(x_train,y_train)
ridge_best_lambda=ridge_cv.alpha_
ridge=Ridge(alpha=ridge_best_lambda)
ridge.fit(x_train,y_train)
coef['Ridge Regression']=ridge.coef_[0]
print('ridge_best_lambda:',ridge_best_lambda,'\n')
print(ridge.coef_,'\n')

#Lasso Regression
lasso_cv=LassoCV(alphas=Lambdas,cv=10,max_iter=10000)
lasso_cv.fit(x_train,y_train)
lasso_best_lambda=lasso_cv.alpha_
lasso=Lasso(alpha=lasso_best_lambda,max_iter=10000)
lasso.fit(x_train,y_train)
coef['Lasso Regression']=lasso.coef_
print('lasso_best_lambda:',lasso_best_lambda,'\n')
print(lasso.coef_,'\n')

print(coef,'\n\n','Done')

#Linear Regression
train_MSE1=mean_squared_error(linear.predict(x_train),y_train)
vali_MSE1=mean_squared_error(linear.predict(x_vali),y_vali)
print('MSE -- Linear Regression -- Training:',train_MSE1)
print('MSE -- Linear Regression -- Validation:',vali_MSE1)

#Ridge Regression
train_MSE2=mean_squared_error(ridge.predict(x_train),y_train)
vali_MSE2=mean_squared_error(ridge.predict(x_vali),y_vali)
print('MSE -- Ridge Regression -- Training:',train_MSE2)
print('MSE -- Ridge Regression -- Validation:',vali_MSE2)

#Lasso Regression
train_MSE3=mean_squared_error(lasso.predict(x_train),y_train)
vali_MSE3=mean_squared_error(lasso.predict(x_vali),y_vali)
print('MSE -- Lasso Regression -- Training:',train_MSE3)
print('MSE -- Lasso Regression -- Validation:',vali_MSE3)

#predict
isnull_y=ridge.predict(df_isnull_x)

#reverse normalization
df_isnull_y_r=stand_scaler2.inverse_transform(isnull_y)
df_isnull_y_r=pd.DataFrame(df_isnull_y_r,columns=['Depth'],index=df_isnull.index)
print("There is no negative value in 'Depth':",sum(df_isnull_y_r['Depth']>=0)==df_isnull_y_r.shape[0])

df_isnull_x_r=stand_scaler3.inverse_transform(isnull_x)
df_isnull_x_r=pd.DataFrame(df_isnull_x_r,columns=['Latitude','Longitude','Continent','Moment magnitude','Seismic moment'],index=df_isnull.index)

df_notnull_y_r=stand_scaler2.inverse_transform(notnull_y)
df_notnull_y_r=pd.DataFrame(df_notnull_y_r,columns=['Depth'],index=df_notnull.index)

df_notnull_x_r=stand_scaler1.inverse_transform(notnull_x)
df_notnull_x_r=pd.DataFrame(df_notnull_x_r,columns=['Latitude','Longitude','Continent','Moment magnitude','Seismic moment'],index=df_notnull.index)

#df_notnull_x_r.head()
#df_notnull_y_r.head()
#df_isnull_x_r.head()

df_isnull[['Depth']]=df_isnull_y_r[['Depth']]
df_isnull[['Latitude','Longitude','Continent','Moment magnitude','Seismic moment']]=df_isnull_x_r[['Latitude','Longitude','Continent','Moment magnitude','Seismic moment']]
df_notnull[['Depth']]=df_notnull_y_r[['Depth']]
df_notnull[['Latitude','Longitude','Continent','Moment magnitude','Seismic moment']]=df_notnull_x_r[['Latitude','Longitude','Continent','Moment magnitude','Seismic moment']]
df=pd.concat([df_notnull,df_isnull],axis=0).sort_index()
df.info()
df

nan_index = df.isna()
print(nan_index.sum(),'\n')
print('There is duplicate row:',any(df.duplicated()))

df.to_csv('data_after_preprocessing.csv', index=False, header=True)
