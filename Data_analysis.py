
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn import metrics
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


train = pd.read_csv(r'C:\Users\alex_\Desktop\Kaggle\Tabular playground - sep\train.csv')
test = pd.read_csv(r'C:\Users\alex_\Desktop\Kaggle\Tabular playground - sep\test.csv')


# Initial glance at data. The only issue I find is that there is missing data among most of the features though not the target "claim"
print(train.shape)
print(train.columns)
print(train.info())
print(train.describe())
print(train.isnull().sum())
print(train.nunique())
print(train.dtypes)
print(train['claim'].value_counts())


# I split the data into an X and Y. Now I am free to manage the data in X however I want.

X = train.drop('claim', axis=1) 
y = train['claim']


######################### Mapping and plotting missing data ##################################

# First I create a summary of the missing data in the df "missing_data" 
total = X.isnull().sum().sort_values(ascending=False)
percent_1 = X.isnull().sum()/X_train.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', 'percents'])
missing_data.head(10)

#      Total  percents
# f31   15678       1.6
# f46   15633       1.6
# f24   15630       1.6
# f83   15627       1.6
# f68   15619       1.6
# f103  15600       1.6
# f95   15599       1.6
# f12   15593       1.6
# f116  15589       1.6
# f64   15578       1.6



# Plotting NaNs - also making a dataframe out of the missing so that its slightly easier to plot them
total_df = total.to_frame(name='missing')
total_df['columns'] = total_df.index

plt.style.use('ggplot')
plt.bar(total_df["columns"], total_df['missing'], color='green')
plt.show()

######################## Applying imputation and standardisation ########################

# As all the features are 'ints', we can impute such values to the missing as means, modes, etc. In data where there are a lot of outliers, the mode is preferred as its not 
# sensitive to them, whereas the mean is better suited for data without outliers. In this dataset there are very few cases of outliers so I used the mean.

# Imputation - here we can also use df.fillna(df.mean()), however I wanted to explore the SimpleImputer() function.
my_imputer = SimpleImputer(strategy='mean')
my_imputer.fit(X)
X_train_imputed = my_imputer.transform(X)
X_train_imp = pd.DataFrame(X_train_imputed, columns = X.columns)

print(X_train_imp.isnull().sum())

# Scaling. As we will see soon, some of the features operate at a completely different scale from the other features. Depending on the machine learning algorithm, that
# might make a large difference. 

scaled_data = X_train_imp.iloc[:, 1:]

scalar = StandardScaler()
scalar.fit(scaled_data)
X_scaled = scalar.transform(scaled_data)
X_scaled = pd.DataFrame(X_scaled, columns = scaled_data.columns)
X_scaled.insert(0, 'id', X_train_imp['id'])


#################################### plots ################

plt.boxplot(X_train_imp.iloc[:, 1:])
plt.show()

plt.boxplot(X_scaled.iloc[:, 1:])
plt.show()

max = train_imp.iloc[:, 1:].max().sort_values(ascending=False)
max.plot(kind='bar')
plt.show()

max = X_scaled.iloc[:, 1:].max().sort_values(ascending=False)
max.plot(kind='bar')
plt.show()



################################ Building models ################

del X_train


X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)
X_train_imp, X_test_imp, y_train2, y_test2 = train_test_split(X_train_imp, y, test_size=0.2, random_state=42, shuffle=True)

xgboost_scale = XGBClassifier(random_state=42, use_label_encoder=False)
xgboost_scale.fit(X_train_scaled, y_train)
predictions_scale = xgboost_scale.predict(X_test_scaled)

xgboost_imp = XGBClassifier(random_state=42, use_label_encoder=False)
xgboost_imp.fit(X_train_imp, y_train2)
predictions_imp = xgboost_imp.predict(X_test_imp)


XGB_scaled_performance = [accuracy_score(y_test, predictions_scale), 
                      roc_auc_score(y_test, predictions_scale),
                      f1_score(y_test, predictions_scale)]

XGB_imputed_performance = [accuracy_score(y_test2, predictions_imp), 
                      roc_auc_score(y_test2, predictions_imp),
                      f1_score(y_test2, predictions_imp)]



metrics = ["Accuracy", "AUC", "F1"]

model_performance = pd.DataFrame({'XGB_imp':XGB_imputed_performance})
model_performance["XGB_scaled"] = pd.DataFrame(XGB_scaled_performance)
model_performance.index = metrics

print(model_performance)




# Exploring the test data as well

print(test.shape)
print(test.columns)
print(test.info())
print(test.describe())
print(test.isnull().sum())
print(test.nunique())
print(test.dtypes)



# Imputation
my_imputer = SimpleImputer(strategy='mean')
my_imputer.fit(test)
test_imputed = my_imputer.transform(test)
test_imputed = pd.DataFrame(test_imputed, columns = test.columns)
print(test_imputed.isnull().sum())


scalar = StandardScaler()
scalar.fit(test_imputed)
test_scaled = scalar.transform(test_imputed)
test_scaled = pd.DataFrame(test_scaled, columns = test_imputed.columns)

predictions = xgboost.predict_proba(test_scaled)[:,1]

subm = test[['id']]
subm['claim'] = predictions

print(subm.head())




