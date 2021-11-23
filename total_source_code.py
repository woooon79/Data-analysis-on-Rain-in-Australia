import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

df = pd.read_csv("weatherAUS.csv")

# divide data into two categories by type
Categorical = ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
Numeric = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am',
           'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
           'Temp9am', 'Temp3pm']
rel_corr = []


# dirty data check
def datacheck(data):
    print(data.isna().sum())

    for x in Categorical:
        print("\n ***{} The number of data in each column *** \n {}\n".format(x, data[x].value_counts()))

    for x in Numeric:
        print("\n *** {} Data information in each columns ***".format(x))
        print("\n min : {}".format(data[x].min()))
        print("max : {}".format(data[x].max()))
        print("mean : {}".format(data[x].mean()))
        print("median : {}\n".format(data[x].median()))

    data2 = data.drop("Date", axis=1)
    cols = data2.columns
    num_cols = data2._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(30, 20))
    ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]
    i = 0
    for col in cat_cols:
        dict_data = data[col].value_counts().to_dict()
        ax_list[i].bar(list(dict_data.keys()), list(dict_data.values()))

        if col == "Location":
            ax_list[i].set_xticklabels(list(dict_data.keys()), rotation=90)
            ax_list[i].tick_params(axis='x', which='major', labelsize=13)

        ax_list[i].set_title(col)
        i += 1

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
    plt.show()

    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(
        4,
        4,
        figsize=(
            30,
            30))
    ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16]
    i = 0
    for col in num_cols:
        data[col].astype("float").hist(bins=40, ax=ax_list[i])
        ax_list[i].set_title(col)
        i += 1
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
    plt.show()


# Remove the missing value and visualize associations as heatmap, and visualize columns that are highly associated with the target.
def heatmap(data):
    data2 = df.drop("Date", axis=1)

    # Configure a dataset with none of the missing values
    df_msno = data2.dropna(axis=0, how='any')
    for x in Categorical[1:]:
        globals()['{}_Label'.format(x)] = LabelEncoder()
        globals()['{}_Label'.format(x)].fit(df_msno[x])
        df_msno[x] = globals()['{}_Label'.format(x)].transform(df_msno[x])

    corr = df_msno.corr()
    top_corr = corr.index
    # First heatmap
    plt.figure(figsize=(9, 9))
    sns.heatmap(df_msno[top_corr].corr(), annot=True, cmap="RdYlGn")
    plt.show()
    # Second heatmap
    rel_corr = ['Rainfall', 'Humidity9am', 'Humidity3pm', 'Cloud9am', 'Cloud3pm', 'RainToday', 'RainTomorrow']
    plt.figure(figsize=(9, 9))
    sns.heatmap(df_msno[rel_corr].corr(), annot=True, cmap="RdYlGn")
    plt.show()


# data cleaning
def cleaning(data):
    rel_corr = ['Rainfall', 'Humidity9am', 'Humidity3pm', 'Cloud9am', 'Cloud3pm', 'RainToday', 'RainTomorrow']
    data2 = data.drop("Date", axis=1)
    df_msno = data2.dropna(axis=0, how='any')
    # Remove rows where "RainTomorrow" value is nan because the target value should not contain nan values
    data = data[data['RainTomorrow'].isna() == False]

    # More than 50 precipitation => heavy rain
    data.drop(data[data["Rainfall"] > 50].index, inplace=True)

    # Replace a day with a rainfall of higher than 1 with a rainy day

    data.reset_index(drop=True, inplace=True)
    for x in range(len(data)):
        if data['RainToday'].isna()[x] == True and data['Rainfall'][x] <= 1:
            data['RainToday'][x] = 'No'
        elif data['RainToday'].isna()[x] == True and data['Rainfall'][x] > 1:
            data['RainToday'][x] = 'Yes'

    # Drop on days when rainfall value does not exist,
    # set to 1.1 if rainfall value does not exist on rainy days

    data.reset_index(drop=True, inplace=True)
    for x in range(len(data)):
        if data['Rainfall'].isna()[x] == True and data['RainToday'][x] == 'Yes':
            data['Rainfall'].isna()[x] = 1.1
        elif data['Rainfall'].isna()[x] == True and data['RainToday'][x] == 'No':
            data['Rainfall'].isna()[x] = 0
        elif data['Rainfall'].isna()[x] == True and data['RainToday'].isna()[x] == True:
            data.drop(x, inplace=True)

    data.reset_index(drop=True, inplace=True)
    data = data[rel_corr]
    data["Humidity9am"].fillna(df_msno["Humidity9am"].mean(), inplace=True)
    data["Humidity3pm"].fillna(df_msno["Humidity3pm"].mean(), inplace=True)
    data["Cloud9am"].fillna(df_msno["Cloud9am"].median(), inplace=True)
    data["Cloud3pm"].fillna(df_msno["Cloud3pm"].median(), inplace=True)

    # Save clean data before encoding and scaling
    data.to_csv("before_encoder_scale.csv", index=False)


'''
find_best_model
INPUT : config ={'dataset':pd.Dataframe,
                 'models': list of models }

OUTPUT : best accuarcy , best model

You can compare models with accuracy. In this process we experience with KNN,SVM,Naive bayes algorithms.
Also we apply each algorithm with best hyperparameter.

KNN : find best 'n_neighbors' hyperparameter value with 'for' loop. 
SVM : Using gridsearchCV to find best 'c' hyperparameter value
'''


def find_best_model(config):
    data = config['dataset']
    data = data.sample(500)  # sampling
    data = data.astype({'RainTomorrow': 'int'})

    # Set features and target variable
    # Target variable : RainTomorrow
    y = data['RainTomorrow']

    X = data.drop(['RainTomorrow'], axis=1)

    # Model building. Split dataset to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    best_accuracy = 0
    best_model = ''
    for model in config['models']:
        classifier = model
        if model == 'KNN':

            # Set range to kind best k(hyperparameter) and find best k
            k_list = range(1, 100, 2)
            train_acc = []
            test_acc = []
            for k in k_list:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                prediction = knn.predict(X_test)
                train_acc.append(knn.score(X_train, y_train))
                test_acc.append((prediction == y_test).mean())

            best_k_i = test_acc.index(max(test_acc))
            best_k = k_list[best_k_i]


            # Predict y_test with best k value
            classifier = KNeighborsClassifier(n_neighbors=best_k)

            # knn = KNeighborsClassifier(n_neighbors=best_k)
            # knn.fit(X_train, y_train)
            # knn_pred = knn.predict(X_test)
            #
            # # calcualte metrics
            # KNN_acc = float(metrics.accuracy_score(y_test, knn_pred).round(3))
            # KNN_mse = float(round(metrics.mean_squared_error(y_test, knn_pred), 3))
            # KNN_jac = float(metrics.jaccard_score(y_test, knn_pred, average='weighted').round(3))
            # KNN_f1s = float(metrics.f1_score(y_test, knn_pred, average='weighted', zero_division=1).round(3))
            #
            # # classification_report
            # print('======================= [ KNN ] =======================\n')
            # print(classification_report(y_test, knn_pred, zero_division=1))
            # print('-------------------------------------------------------\n')
            #
            # # confusion matrix
            # confusionMX = confusion_matrix(y_test, knn_pred)
            # sns.heatmap(pd.DataFrame(confusionMX), annot=True, cmap='YlOrRd', fmt='g')
            # plt.title("KNN Confusion Matrix")
            # plt.ylabel('Actual class')
            # plt.xlabel('Predicted class')
            # plt.show()



        elif model == 'SVM':

            param_gid = {'C': [0.0001, 0.001, 0.1, 0, 1, 10, 100]}
            grid_search = GridSearchCV(SVC(kernel='linear'), param_gid, return_train_score=True)

            grid_search.fit(X_train, y_train)
            classifier = SVC(kernel='linear', C=grid_search.best_params_['C'])

            # svc = SVC(kernel='linear', C=grid_search.best_params_['C'])
            # model = svc.fit(X_train, y_train)
            #
            # # Predicted value
            # SVC_pred = model.predict(X_test)
            #
            # # calcualte metrics
            # SVC_acc = float(metrics.accuracy_score(y_test, SVC_pred).round(3))
            # SVC_mse = float(round(metrics.mean_squared_error(y_test, SVC_pred), 3))
            # SVC_jac = float(metrics.jaccard_score(y_test, SVC_pred, average='weighted').round(3))
            # SVC_f1s = float(metrics.f1_score(y_test, SVC_pred, average='weighted', zero_division=1).round(3))
            #
            # # classification_report
            # print('======================= [ SVM ] =======================\n')
            # print(classification_report(y_test, SVC_pred, zero_division=1))
            # print('-------------------------------------------------------\n')
            #
            # # confusion matrix
            # confusionMX = confusion_matrix(y_test, SVC_pred)
            # sns.heatmap(pd.DataFrame(confusionMX), annot=True, cmap='YlOrRd', fmt='g')
            # plt.title("SVC Confusion Matrix")
            # plt.ylabel('Actual class')
            # plt.xlabel('Predicted class')
            # plt.show()


        elif model == 'Naive_bayes':
            classifier = GaussianNB()

            # # Predicted value
            # NB_pred = classifier.predict(X_test)
            # 
            # # calcualte metrics
            # NB_acc = float(metrics.accuracy_score(y_test, NB_pred).round(3))
            # NB_mse = float(round(metrics.mean_squared_error(y_test, NB_pred), 3))
            # NB_jac = float(metrics.jaccard_score(y_test, NB_pred, average='weighted').round(3))
            # NB_f1s = float(metrics.f1_score(y_test, NB_pred, average='weighted', zero_division=1).round(3))
            # 
            # # classification_report
            # print('=================== [ NB_gaussian ] ===================\n')
            # print(classification_report(y_test, NB_pred, zero_division=1))
            # print('-------------------------------------------------------\n')
            # 
            # # confusion matrix
            # confusionMX = confusion_matrix(y_test, NB_pred)
            # sns.heatmap(pd.DataFrame(confusionMX), annot=True, cmap='YlOrRd', fmt='g')
            # plt.title("NB_gaussian Confusion Matrix")
            # plt.ylabel('Actual class')
            # plt.xlabel('Predicted class')
            # plt.show()

        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)
        accuracy = float(metrics.accuracy_score(y_test, prediction).round(3))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = classifier

    return best_accuracy, best_model


'''
tuner
INPUT : config ={        
        'dataframe': pd.DataFrame,
        'encode_cols':columns for encoding  ,
        'scale_cols':columns for encoding,
        'scalers' : scalers list,
        'encoders' : encoders list,
        'models':models list}
OUTPUT : (all model combination list (pd.DataFrame) , best_model (dictionary) )

It is like gridsearch. Inspect all combination of each scalers,encoders with models.

'''


def tuner(config):
    comb = pd.DataFrame(columns=['Scaler', 'Encoder', 'Algorithm', 'Score'])
    best_accuracy = 0;
    new_columns = ''
    for encoder in config['encoders']:
        temp_df = config['dataframe'].copy()
        encoding_df = pd.DataFrame()
        # encoding the categorical columns
        if encoder == 'OneHotEncoder':
            encoding_df = pd.DataFrame(pd.get_dummies(temp_df[config['encode_cols']]))
        else:
            for col in config['encode_cols']:
                encoder.fit(temp_df[col])
                encoding_df[col] = encoder.transform(temp_df[col])

        for scaler in config['scalers']:
            scaled_df = scaler.fit_transform(temp_df[config['scale_cols']])
            scaled_df = pd.DataFrame(scaled_df, columns=config['scale_cols'])
            temp_df = pd.concat([scaled_df, encoding_df], axis=1)

            accuracy, model = find_best_model({'dataset': temp_df, 'models': config['models']})

            comb = comb.append({'Scaler': scaler,
                                'Encoder': encoder,
                                'Algorithm': model,
                                'Score': accuracy
                                }, ignore_index=True)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = {'Scaler': scaler,
                              'Encoder': encoder,
                              'Algorithm': model,
                              'Score': accuracy,
                              'dataset': temp_df}

    return comb, best_model


# find best model with grid search for scaling and encoding
'''
find_best_comb
INPUT : (dataframe , columns to encoding (list) , columns to scaling (list), scalers (list) , encoders (list) ,models (list) )
OUTPUT : (all model combination list (pd.DataFrame) , best_model (dictionary) )

define 'config' dictionary to set list of scalers,encoders and classifiers.
Call 'tuner()' function to find best model
'''


def find_best_comb(df, encode_col, scale_col, scalers=None, encoders=None, models=None):
    if scalers == None:
        scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]

    if encoders == None:
        encoders = ['OneHotEncoder', LabelEncoder()]

    if models == None:
        models = ['KNN', 'SVM', 'Naive_bayes']

    # Set parameters to find (Find best combination of scaler and encoder)
    config = {
        'dataframe': df,
        'encode_cols': encode_col,
        'scale_cols': scale_col,
        'scalers': scalers,
        'encoders': encoders,
        'models': models
    }

    return tuner(config)


def drowing_scaled(df, scaled_df):
    encoder = LabelEncoder()
    encoder.fit(df['RainToday'])
    df['RainToday'] = encoder.transform(df["RainToday"])
    sns.set_style("darkgrid")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
    ax1.set_title('BeforeScaling')
    sns.kdeplot(df['Rainfall'], ax=ax1, label='Rainfall')
    sns.kdeplot(df['Humidity9am'], ax=ax1, label='Humidty9am')
    sns.kdeplot(df['Humidity3pm'], ax=ax1, label='Humidty3pm')
    sns.kdeplot(df['Cloud9am'], ax=ax1, label='Cloud9am')
    sns.kdeplot(df['Cloud3pm'], ax=ax1, label='Cloud3pm')
    sns.kdeplot(df['RainToday'], ax=ax1, label='RainToday')

    ax2.set_title('AfterScaling')
    sns.kdeplot(scaled_df['Rainfall'], ax=ax2, label='Rainfall')
    sns.kdeplot(scaled_df['Humidity9am'], ax=ax2, label='Humidty9am')
    sns.kdeplot(scaled_df['Humidity3pm'], ax=ax2, label='Humidty3pm')
    sns.kdeplot(scaled_df['Cloud9am'], ax=ax2, label='Cloud9am')
    sns.kdeplot(scaled_df['Cloud3pm'], ax=ax2, label='Cloud3pm')
    sns.kdeplot(scaled_df['RainToday'], ax=ax2, label='RainToday')
    plt.legend()
    plt.show()


def main():
    # Load initial dataset
    datacheck(df)
    heatmap(df)
    cleaning(df)
    bes_df = pd.read_csv("preprocessing.csv")

    model_combination_list, best_model = find_best_comb(bes_df, ['RainToday', 'RainTomorrow'],
                                                        ['Rainfall', 'Humidity9am', 'Humidity3pm', 'Cloud9am',
                                                         'Cloud3pm'])
    pd.set_option('display.max_row', 500)
    pd.set_option('display.max_columns', 100)

    print(model_combination_list)
