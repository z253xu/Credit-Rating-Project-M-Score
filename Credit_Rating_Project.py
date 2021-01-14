file_path = r'/Users/zihanxu/'
import pandas as pd
import statsmodels.api as sm
import numpy as np

def new_report(year):
    df = pd.read_csv(file_path+'Downloads/report_'+str(year)+'1231.csv', index_col=False)
    if ('TOT_CUR_ASSETS' in df) and ('TOT_CUR_LIAB' in df):
        df['CUR_RATIO'] = df['TOT_CUR_ASSETS'] / df['TOT_CUR_LIAB']
    if ('TOT_CUR_ASSETS' in df) and ('INVENTORIES' in df) and ('TOT_CUR_LIAB' in df):
        df['QUICK_RATIO'] = (df['TOT_CUR_ASSETS']-df['INVENTORIES']) / df['TOT_CUR_LIAB']
    if ('TOT_LIAB' in df) and ('TOT_ASSETS' in df):
        df['DEBT_TO_ASSETS_RATIO'] = (100.*(df['TOT_LIAB'] / df['TOT_ASSETS'])).round(1).astype(str)+ '%'
    if ('NON_CUR_LIAB_DUE_WITHIN_1Y' in df) and ('NOTES_PAYABLE' in df):
        df['CASH_DEBT_RATIO'] = df['NON_CUR_LIAB_DUE_WITHIN_1Y'] + df['NOTES_PAYABLE']
    if ('NET_CASH_FLOWS_OPER_ACT' in df) and ('TOT_LIAB' in df):
        df['CASH_FLOW_RATIO'] = df['NET_CASH_FLOWS_OPER_ACT'] / df['TOT_LIAB']
    if ('NET_CASH_FLOWS_OPER_ACT' in df) and ('TOT_ASSETS' in df):
        df['CASH_REC_FOR_ALL_ASSETS'] = df['NET_CASH_FLOWS_OPER_ACT'] / df['TOT_ASSETS']
    df.to_csv(str(year)+'_new.csv', index=False)

def report_generator(start_year, end_year):
    while start_year <= end_year:
        new_report(start_year)
        start_year+=1
        
report_generator(2011,2019)

df1 = pd.read_csv(file_path+'2011_new.csv', index_col=False)
df1.rename(columns={'Unnamed: 0':'S_INFO_COMPCODE'}, inplace=True)
df1['YEAR'] = 2011
for i in range (2012,2019):
    df2 = pd.read_csv(file_path+str(i)+'_new.csv', index_col=False)
    df2.rename(columns={'Unnamed: 0':'S_INFO_COMPCODE'}, inplace=True)
    df2['YEAR'] = i
    dfnew = pd.concat([df1,df2], axis=0, sort=False)
    df1 = dfnew
    
df1.sort_values(['S_INFO_COMPCODE', 'YEAR'], ascending=[True, False], inplace=True)

def fillnan(data):
    if data is None:
        return np.nan
    else:
        return data

def nandivide(a,b,tp=1):
    #if tp == 1:
        a = a.apply(fillnan).apply(float)
        b = b.apply(fillnan).apply(float)
        b = b.apply(lambda x:np.nan if x==0 else x)
        return a/b

#Setting for current year
df1['Sales'] = df1['TOT_OPER_REV']
df1['COGS'] = df1['LESS_OPER_COST']
df1['Net Receivables'] = (df1['NOTES_RCV']
                          +df1['ACCT_RCV']
                          +df1['OTH_RCV']
                          +df1['DVD_RCV']
                          +df1['INT_RCV']
                          -df1['NOTES_PAYABLE']
                          -df1['ACCT_PAYABLE']
                          -df1['ADV_FROM_CUST'])
#df1['AQ'] = 1-(df1['TOT_CUR_ASSETS']+df1['FIX_ASSETS'])/df1['TOT_ASSETS']
df1['AQ'] = 1-nandivide((df1['TOT_CUR_ASSETS']+df1['FIX_ASSETS']), df1['TOT_ASSETS'])
df1['Depreciation'] = df1['DEPR_FA_COGA_DPBA']
df1['PPE'] = df1['FIX_ASSETS']
df1['SG&A'] = (df1['LESS_FIN_EXP']
               +df1['LESS_GERL_ADMIN_EXP']
               +df1['LESS_NON_OPER_EXP']
               +df1['LESS_OPER_COST']
               +df1['LESS_SELLING_DIST_EXP'])
#df1['LVG'] = df1['TOT_LIAB']/df1['TOT_ASSETS']
df1['LVG'] = nandivide(df1['TOT_LIAB'], df1['TOT_ASSETS'])

#Setting for previous year
df1['pre_sales'] = df1.groupby('S_INFO_COMPCODE')['Sales'].shift()
df1['pre_sales'] = df1.groupby('S_INFO_COMPCODE')['Sales'].shift()
df1['pre_cogs'] = df1.groupby('S_INFO_COMPCODE')['COGS'].shift()
df1['pre_NR'] = df1.groupby('S_INFO_COMPCODE')['Net Receivables'].shift()
df1['pre_TCA'] = df1.groupby('S_INFO_COMPCODE')['TOT_CUR_ASSETS'].shift()
df1['pre_PPE'] = df1.groupby('S_INFO_COMPCODE')['PPE'].shift()
df1['pre_TA'] = df1.groupby('S_INFO_COMPCODE')['TOT_ASSETS'].shift()
#df1['pre_AQ'] = 1-(df1['pre_TCA']+df1['pre_PPE'])/df1['pre_TA']
df1['pre_AQ'] = 1-nandivide((df1['pre_TCA']+df1['pre_PPE']), df1['pre_TA'])
df1['pre_dep'] = df1.groupby('S_INFO_COMPCODE')['Depreciation'].shift()
df1['pre_SG&A'] = df1.groupby('S_INFO_COMPCODE')['SG&A'].shift()
df1['pre_LVG'] = df1.groupby('S_INFO_COMPCODE')['LVG'].shift()

#df1['DSRI'] = (df1['Net Receivables']/df1['Sales'])/(df1['pre_NR']/df1['pre_sales'])
df1['DSRI'] = nandivide(nandivide(df1['Net Receivables'], df1['Sales']), nandivide(df1['pre_NR'], df1['pre_sales']))
#df1['GMI'] = (df1['pre_sales']-df1['pre_cogs'])/df1['pre_sales']/((df1['Sales']-df1['COGS'])/df1['Sales'])
df1['GMI'] = nandivide(nandivide((df1['pre_sales']-df1['pre_cogs']), df1['pre_sales']),
                       nandivide((df1['Sales']-df1['COGS']),df1['Sales']))
#df1['AQI'] = df1['AQ']/df1['pre_AQ']
df1['AQI'] = nandivide(df1['AQ'], df1['pre_AQ'])
#df1['SGI'] = df1['Sales']/df1['pre_sales']
df1['SGI'] = nandivide(df1['Sales'], df1['pre_sales'])
#df1['DEPI'] = (df1['pre_dep']/(df1['pre_dep']+df1['pre_PPE']))/(df1['Depreciation']/(df1['Depreciation']+df1['PPE']))
df1['DEPI'] = nandivide(nandivide(df1['pre_dep'], (df1['pre_dep']+df1['pre_PPE'])),
                        nandivide(df1['Depreciation'], (df1['Depreciation']+df1['PPE'])))
#df1['SGAI'] = (df1['SG&A']/df1['Sales'])/(df1['pre_SG&A']/df1['pre_sales'])
df1['SGAI'] = nandivide(nandivide(df1['SG&A'], df1['Sales']),nandivide(df1['pre_SG&A'], df1['pre_sales']))
#df1['LVGI'] = df1['LVG']/df1['pre_LVG']
df1['LVGI'] = nandivide(df1['LVG'], df1['pre_LVG'])
#df1['TATA'] = (df1['OPER_PROFIT']-df1['NET_CASH_FLOWS_OPER_ACT'])/df1['TOT_ASSETS']
df1['TATA'] = nandivide((df1['OPER_PROFIT']-df1['NET_CASH_FLOWS_OPER_ACT']), df1['TOT_ASSETS'])
df1.fillna(0, inplace=True)

#Compute M-Score
df1['MScore'] = (-4.840
                 + 0.920*df1['DSRI']
                 + 0.528*df1['GMI']
                 + 0.404*df1['AQ']
                 + 0.892*df1['SGI']
                 + 0.115*df1['DEPI']
                 - 0.172*df1['SGAI']
                 - 0.327*df1['LVGI']
                 + 4.697*df1['TATA'])

#Label
df3 = pd.read_csv(file_path+'Downloads/ratings.csv', index_col=False)
df3['YEAR'] = (df3['ANN_DT']/10000).astype('int')
good_rating = ['AAA', 'Aa1', 'Aa2', 'AAA-', 'AA+', 'AA']
df3['LABEL'] = (~df3['B_INFO_CREDITRATING'].isin(good_rating)
                & df3['B_INFO_PRECREDITRATING'].isin(good_rating)).astype('int')

df1.loc[:,'LABEL'] = 0

def change_label(comp_id, date):
    tb_labeled = df3['S_INFO_COMPCODE'] == comp_id
    records = df3.loc[tb_labeled]
    records = records.loc[records['YEAR']==date+1]
    if len(records) == 0:
        return 0
    for i in range(len(records)):
        if records['LABEL'].iloc[i] == 1:
            return 1
        else:
            return 0
        
df1['LABEL'] = df1.apply(lambda row:change_label(row['S_INFO_COMPCODE'], row['YEAR']), axis = 1)

del df3['S_INFO_COMPNAME']

df3.to_csv('rating_new.csv', index=False, encoding='GBK')
df1.to_csv('new_report.csv', index=False)

df4 = pd.read_csv(file_path+'new_report.csv', index_col=False)
df4 = df4.dropna(subset=['TOT_ASSETS'])
df4 = df4[['LABEL', 'DSRI', 'GMI', 'AQ', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']]
df4.fillna(0,inplace=True)
df4
df4.isnull().sum()

x = df4['DSRI']
y = df4['LABEL']
results = sm.OLS(y, x).fit()
results.summary() 

x = df4['GMI']
y = df4['LABEL']
results = sm.OLS(y, x).fit()
results.summary()

x = df4['AQ']
y = df4['LABEL']
results = sm.OLS(y, x).fit()
results.summary() 

x = df4['SGI']
y = df4['LABEL']
results = sm.OLS(y, x).fit()
results.summary() 

x = df4['DEPI']
y = df4['LABEL']
results = sm.OLS(y, x).fit()
results.summary() 

x = df4['SGAI']
y = df4['LABEL']
results = sm.OLS(y, x).fit()
results.summary() 

x = df4['LVGI']
y = df4['LABEL']
results = sm.OLS(y, x).fit()
results.summary() 

x = df4['TATA']
y = df4['LABEL']
results = sm.OLS(y, x).fit()
results.summary() 

def confusion_matrix(pred,real):
    x = pd.DataFrame()
    x['label'] = real
    x['predict'] = pred

    TruePositive = sum(x[x['predict']>-1.78]['label'])
    FalsePositive = sum(1-x[x['predict']>-1.78]['label'])
    TrueNegative = sum(1-x[x['predict']<-1.78]['label'])
    FalseNegative = sum(x[x['predict']<-1.78]['label'])

    Precision = TruePositive/(TruePositive+FalsePositive)
    Recall = TruePositive/(TruePositive+FalseNegative)
    Accuracy = (TruePositive+TrueNegative)/(TruePositive+FalsePositive+TrueNegative+FalseNegative)
    F1_Score = (2*Recall*Precision)/(Recall+Precision)
    print('Precision: {}'.format(Precision))
    print('Recall: {}'.format(Recall))
    print('Accuracy: {}'.format(Accuracy))
    print('F1-Score: {}'.format(F1_Score))
    _confusion_matrix = pd.DataFrame({
                    'Negative':{'True':FalseNegative,'False':TrueNegative},
                    'Positive':{'True':TruePositive,'False':FalsePositive}                            
                    })        
    print(_confusion_matrix)
    return _confusion_matrix

#confusion_matrix(df1['MScore'], df1['LABEL'])

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc

def illus_roc_curve(real,pred):
    # Compute ROC curve and ROC area for each class 
    
    fpr,tpr,threshold = roc_curve(real, pred)
    roc_auc = auc(fpr,tpr)
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

#'DSRI' 'GMI', 'AQ', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA'
reg_cols = [
        'DSRI',
        'GMI', 
        'AQ',
        'SGI',
        'DEPI',
        'SGAI',
        'LVGI',
        'TATA',
        ]

logit = sm.Logit(df4['LABEL'], df4[reg_cols])
result = logit.fit (method='bfgs')  
result.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor    

def calculate_vif_(X, thresh=10.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]
df5 = df4[['DSRI', 'GMI', 'AQ', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']]
calculate_vif_(df5)

import pandas as pd
import lightgbm as lgb

df5 = pd.read_csv(file_path+'new_report.csv', index_col=False)
reg_cols = [
        'DSRI',
        'GMI', 
        'AQ',
        'SGI',
        'DEPI',
        'SGAI',
        'LVGI',
        'TATA',
        ]

y = df5['LABEL']
x = df5[reg_cols]
n = int(len(x)*0.5)
x_train = x.iloc[:n]
y_train = y.iloc[:n]
x_test = x.iloc[n:]
y_test = y.iloc[n:]


lgb_train = lgb.Dataset(x_train, y_train, feature_name =reg_cols)
lgb_dev = lgb.Dataset(x_test,y_test, reference = lgb_train)

params = {
    'task':'train',
    'boosting_type':'gbdt',
    'metric': {'l2','fair'},
    'num_leaves':20,
    'num_threads':8,
    'learning_rate':0.02,
    'feature_fraction':0.3,
    'bagging_fraction':0.8
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_dev,
               early_stopping_rounds=10)

x = x.loc[:,reg_cols]
y_pred = gbm.predict(x, num_iteration = gbm.best_iteration)

illus_roc_curve(y,y_pred)
