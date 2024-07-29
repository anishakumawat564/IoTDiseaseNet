import os
from sklearn import preprocessing
import numpy as np
import pandas as pd
from numpy import matlib
import random as rn
from BMO import BMO
from CHOA import CHOA
from COA import COA
from FHO import FHO
from Glob_Vars import Glob_Vars
from Model_ALSTM import Model_Attention_LSTM
from Model_CNN import Model_CNN
from Model_TCN import Model_TCN
from Model_VAE_ALSTM_TCN_BYSN import Model_VAE_ALSTM_TCN_BYSN
from Objective_Function import Objfun
from PROPOSED import PROPOSED
from Plot_Results import plot_results_learnperc, plot_convergence, plot_results_kfold


# Read Dataset1
an = 0
if an == 1:
    Directory = './Dataset/Dataset1/'
    dir1 = os.listdir(Directory)
    file1 = Directory + dir1[0]
    read = pd.read_csv(file1)
    data = np.asarray(read)
    Target = data[:, -1].astype('int')
    datas = np.delete(data, -1, 1)
    np.save('Data1.npy', datas)
    np.save('Target1.npy', Target)

# Read Dataset2
an = 0
if an == 1:
    Directory = './Dataset/Dataset2/'
    dir1 = os.listdir(Directory)
    file1 = Directory + dir1[0]
    read = pd.read_csv(file1)
    data = np.asarray(read)
    Target = data[:, 17]
    dat = data[:, 0]
    uni = np.unique(dat)
    for k in range(len(uni)):
        ind = np.where(dat == uni[k])
        dat[ind[0]] = k + 1

    datas = np.delete(data, 17, 1).astype('float')
    np.save('Data2.npy', datas)
    np.save('Target2.npy', Target)

# Read Dataset3
an = 0
if an == 1:
    Directory = './Dataset/Dataset3/'
    dir1 = os.listdir(Directory)
    file1 = Directory + dir1[0]
    read = pd.read_csv(file1)
    data = np.asarray(read)
    Target = data[:, -1].astype('int')
    datas = np.delete(data, -1, 1)
    np.save('Data3.npy', datas)
    np.save('Target3.npy', Target)

# Read Dataset4
an = 0
if an == 1:
    Directory = './Dataset/Dataset4/'
    dir1 = os.listdir(Directory)
    file1 = Directory + dir1[0]
    read = pd.read_csv(file1)
    data = np.asarray(read)
    data1 = data[:, -1]
    data2 = data[:, 0]
    uni = np.unique(data1)
    uni1 = np.unique(data2)
    for k in range(len(uni)):
        ind = np.where(data1 == uni[k])
        data1[ind[0]] = k
    for j in range(len(uni1)):
        ind1 = np.where(data2 == uni1[j])
        data2[ind1[0]] = j
    Target = data[:, -1]
    datas = np.delete(data, -1, 1)
    np.save('Data4.npy', datas)
    np.save('Target4.npy', Target)

# Data Cleaning
an = 0
if an == 1:
    pre = []
    for i in range(4):
        Data = np.load('Data' + str(i + 1) + '.npy', allow_pickle=True)
        ## Preprocessing ##
        pd.isnull('data1')  # locates missing data
        df = pd.DataFrame(Data)
        # Replace with 0 values. Accepts regex.
        df.replace(np.NAN, 0, inplace=True)
        # Replace with zero values
        df.fillna(value=0, inplace=True)
        df.drop_duplicates()  # removes the duplicates
        rawData = np.array(df)
        # Data normalizing
        scaler = preprocessing.MinMaxScaler()
        for j in range(rawData.shape[1]):
            print(i, np.unique(rawData[:, j]))
        normalized = scaler.fit_transform(rawData)
        pre.append(normalized)
    np.save('Data_Cleaned.npy', pre)

# Optimizaion for feature selection
an = 0
if an == 1:
    Bestsol_Feat = []
    fitness = []
    for i in range(4):
        Feat = np.load('Data_Cleaned.npy', allow_pickle=True)[i]
        Targets = np.load('Target' + str(i + 1) + '.npy', allow_pickle=True)
        Targets = np.reshape(Targets, [-1, 1])
        Glob_Vars.Data = Feat
        Glob_Vars.Target = Targets
        Npop = 10
        Chlen = 17
        xmin = matlib.repmat(np.concatenate(([np.zeros(10), 5, 50, 5, 50, 2, 0.01, 5]), axis=None), Npop, 1)
        xmax = matlib.repmat(
            np.concatenate(([Feat.shape[1] - 1 * np.ones(10), 255, 100, 255, 100, 5, 0.99, 255]), axis=None), Npop, 1)

        fname = Objfun

        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.asarray(rn.uniform(xmin[p1, p2], xmax[p1, p2]))
        Max_iter = 25

        print("COA...")
        [bestfit1, fitness1, bestsol1, time1] = COA(initsol, fname, xmin, xmax, Max_iter)

        print("BMO...")
        [bestfit2, fitness2, bestsol2, time2] = BMO(initsol, fname, xmin, xmax, Max_iter)

        print("CHOA...")
        [bestfit4, fitness4, bestsol4, time3] = CHOA(initsol, fname, xmin, xmax, Max_iter)

        print("FHO...")
        [bestfit3, fitness3, bestsol3, time4] = FHO(initsol, fname, xmin, xmax, Max_iter)

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

        Bestsol_Feat.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
        fitness.append([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])
    np.save('BestSol_Feat.npy', Bestsol_Feat)
    np.save('Fitness.npy', fitness)

## Classification
an = 0
if an == 1:
    EVAL = []
    for a in range(4):
        DATASETS = ['HEART DISEASE', 'PARKINSON', 'DIABETES', 'LUNG CANCER']
        data = [1,2,3,4]
        Feat = np.load('Data_Cleaned.npy', allow_pickle=True)[a]
        sol = np.load('BestSol_Feat.npy', allow_pickle=True)[a]
        Targets = np.load('Target' + str(a + 1) + '.npy', allow_pickle=True)
        Targets = np.reshape(Targets, [-1, 1])
        Eval_all = []
        for i in range(len(data)):
            Eval = np.zeros((10, 14))
            per = round(Targets.shape[0] * data[i])
            for j in range(sol.shape[0]):
                Selected_Feature = Feat[:, (sol[j][:10] - 1).astype(int)]
                learnper = round(Selected_Feature.shape[0] * 0.75)
                train_data = Selected_Feature[per:, :]
                train_target = Targets[per:, :]
                test_data = Selected_Feature[:per, :]
                test_target = Targets[:per, :]
                Eval = Model_VAE_ALSTM_TCN_BYSN(train_data, train_target, test_data, test_target, sol[j][10:].astype('int'))
            Train_Data1 = Feat[:per, :]
            Test_Data1 = Feat[per:, :]
            Train_Target = Targets[per:, :]
            Test_Target = Targets[:per, :]
            Eval[5, :] = Model_CNN(Train_Data1, Train_Target, Test_Data1, Test_Target)
            Eval[6, :] = Model_TCN(Train_Data1, Train_Target, Test_Data1, Test_Target)
            Eval[7, :] = Model_Attention_LSTM(Train_Data1, Train_Target, Test_Data1, Test_Target)
            Eval[8, :] = Model_VAE_ALSTM_TCN_BYSN(Train_Data1, Train_Target, Test_Data1, Test_Target)
            Eval[9, :] = Eval[4, :]
            Eval_all.append(Eval)
        EVAL.append(Eval_all)
    np.save('Eval_all.npy', np.asarray(EVAL))

## K-Fold
an = 0
if an == 1:
    k_fold = 10
    Eval_all = []
    for k in range(4):
        Feat = np.load('Data_Cleaned.npy', allow_pickle=True)[k]
        sol = np.load('BestSol_Feat.npy', allow_pickle=True)[k]
        Targets = np.load('Target' + str(k + 1) + '.npy', allow_pickle=True)
        Ev = []
        for i in range(k_fold):
            Eval = np.zeros((10, 14))
            for j in range(sol.shape[0]):
                Total_Index = np.arange(Feat.shape[0])
                Test_index = np.arange(((i - 1) * (Feat.shape[0] / k_fold)) + 1, i * (Feat.shape[0] / k_fold))
                Train_Index = np.setdiff1d(Total_Index, Test_index)
                Train_Data = Feat[Train_Index, :]
                Train_Target = Targets[Train_Index, :]
                Test_Data = Feat[Test_index, :]
                Test_Target = Targets[Test_index, :]
                Eval[i, :] = Model_VAE_ALSTM_TCN_BYSN(Train_Data, Train_Target, Test_Data, Test_Target, sol[j][10:].astype('int'))
            Total_Index = np.arange(Feat.shape[0])
            Test_index = np.arange(((i - 1) * (Feat.shape[0] / k_fold)) + 1, i * (Feat.shape[0] / k_fold))
            Train_Index = np.setdiff1d(Total_Index, Test_index)
            Train_Data1 = Feat[:Train_Index, :]
            Train_Target = Targets[:Test_index, :]
            Test_Data1 = Feat[Train_Index:, :]
            Test_Target = Targets[Train_Index:, :]
            Eval[5, :] = Model_CNN(Train_Data1, Train_Target, Test_Data1, Test_Target)
            Eval[6, :] = Model_TCN(Train_Data1, Train_Target, Test_Data1, Test_Target)
            Eval[7, :] = Model_Attention_LSTM(Train_Data1, Train_Target, Test_Data1, Test_Target)
            Eval[8, :] = Model_VAE_ALSTM_TCN_BYSN(Train_Data1, Train_Target, Test_Data1, Test_Target)
            Eval[9, :] = Eval[4, :]
            Ev.append(Eval)
        Eval_all.append(Ev)
    np.save('Eval_Fold.npy', Eval_all)

plot_results_learnperc()
plot_results_kfold()
plot_convergence()
