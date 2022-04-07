import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import plotly.express as px

def numpy_to_tensor(np_arr):
    return torch.from_numpy(np_arr)

def preprocessing(X, y, type="Scaler", plot=False, TEST_SIZE=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    if type == "Scaler":
        scaler = MinMaxScaler()
        X_train_transformed = scaler.fit_transform(X_train)
        X_test_transformed = scaler.transform(X_test)
        y_train_transformed = scaler.fit_transform(y_train)
        y_test_transformed = scaler.transform(y_test)
        #y_train_transformed = y_train.values  # no-op
        #y_test_transformed = y_test.values  # no-op


    elif type == "PCA":
        num_components=12
        pca = PCA(n_components=num_components)
        X_train_transformed = pca.fit_transform(X_train)
        X_test_transformed = pca.transform(X_test)
        y_train_transformed = y_train.values  # no-op
        y_test_transformed = y_test.values  # no-op

        if plot==True:
            evr = pca.explained_variance_ratio_
            plt.figure()
            title = 'CumSum of PCA ({0}) Components'.format(num_components)
            print('CumSum of PCA Components' + str(np.cumsum(evr)))
            plt.plot(np.cumsum(evr))
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')
            plt.title(title)
            #plt.savefig(title)
            plt.show()
            plt.clf()
            plt.figure()
            title = 'Explained variance ratio ({0} components)'.format(num_components)
            plt.bar(range(0, len(evr)), evr, alpha=0.5, align='center', label='Individual explained variance')
            plt.title(title)
            plt.legend(loc='best')
            #plt.savefig(title)
            plt.show()

    else:
        X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed = X_train, X_test, y_train, y_test


    return X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed



def get_data(TEST_SIZE=0.2, split=True, plot=False, plot_distributions=False, return_constants=False):
    path = os.path.join(os.getcwd(), "data")
    fp = os.path.join(path, "Untitled spreadsheet - Terres_Li_Sample_Data_Set.csv")
    fp1 = os.path.join(path, "v2_Terres_Li_Sample_Data_Set - Terres_Li_Sample_Data_Set_Appended.csv")
    df = pd.read_csv(fp)
    df1 = pd.read_csv(fp1)
    df2 = pd.concat([df, df1])

    constants = {
        'mu_0': df['8.85E-12'][0],
        'm_p': df['8.85E-12'][1],
        'm_pe': df['8.85E-12'][2],
        'e': df['8.85E-12'][3],
        'k_b': df['8.85E-12'][4],
        'c': df['8.85E-12'][5]
    }
    #timestamps, X, derived_x, y = separate_columns(df)
    timestamps, X, derived_x, y = separate_columns(df2)

    if plot_distributions==True:
        for i, key in enumerate(X.keys()):
            fig = px.histogram(X[key], title="{0} Raw Distribution".format(key))
            fig.show()

    #X = preprocessing(data=X, type="PCA", plot=plot)
    #X = preprocessing(data=X, type="None")
    #y = preprocessing(data=y, type="None")
    X_train, X_test, y_train, y_test = preprocessing(X, y, "Scaler", TEST_SIZE=0.25)

    X_train = numpy_to_tensor(X_train)
    X_test = numpy_to_tensor(X_test)
    y_train = numpy_to_tensor(y_train)
    y_test = numpy_to_tensor(y_test)

    if split == True:
        return timestamps, X_train, X_test, derived_x, y_train, y_test

    elif return_constants == True and split == False:
        return timestamps, X, derived_x, y, constants

    else:
        return timestamps, X, derived_x, y



def separate_columns(df):
    # columns 0, 1
    #   Timestamps
    #
    # columns 2 - 20
    #   The independent variables.
    #   These are used as input
    #
    # columns 21 - 36
    #   The derived variables
    #   Not sure what we use these for
    #
    # Columns 37 - 45
    #   No idea
    #
    # column 46 is the dependent variable
    #   This is used as y_true
    #
    # column 47 - 50
    #   No idea
    timestamps = df[["Tstart", "Tend"]].astype('str')
    independent = df[["Bx", "By", "Bz",
                      "Bmag_avg", "Ni_avg", "Vx",
                      "Vy", "Vz", "VSW",
                      "Vth_mag_Ion", "Vth_para_Ion", "Vth_perp_Ion",
                      "Ne", "Ue", "Te",
                      "Ti", "Ti_para", "Ti_perp"]].astype(np.float)

    empty = df[[" "]]
    derived = df[["VA", "Beta", "Beta_para",
                    "Beta_perp", "Omega_i", "omega_pi",
                    "omega_pe", "rho_i", "rho_s",
                    "rho_c", "d_i", "d_e",
                    "sigma_i", "Lperp", "lambda_d"]].astype(np.float)

    dependent = df[["l_brk"]].astype(np.float)

    for key in independent.keys():
        if independent[key].isnull().values.any():
            print("column: {0} has {1} nan".format(key, df[key].isnull().sum()))



    return timestamps, independent, derived, dependent

if __name__ == '__main__':
    timestamps, x, derived_x, y = get_data(split=False)
    print(x)
