import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def plot_results_learnperc():
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Eval = np.load('Eval_all.npy', allow_pickle=True)
    # learnper = [35, 55, 65, 75, 85]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[0], Eval.shape[2]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[2]):
                if Graph_Term[j] == 9:
                    Graph[k, l] = Eval[k, 3, l, Graph_Term[j] + 4]
                else:
                    Graph[k, l] = Eval[k, 3, l, Graph_Term[j] + 4] * 100

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(4)
        ax.bar(X + 0.00, Graph[:, 0], color='r', width=0.10, label="COA-AMSCNet")
        ax.bar(X + 0.10, Graph[:, 1], color='#cc9f3f', width=0.10, label="BMO-AMSCNet")
        ax.bar(X + 0.20, Graph[:, 2], color='b', width=0.10, label="CHOA-AMSCNet")
        ax.bar(X + 0.30, Graph[:, 3], color='m', width=0.10, label="FHO-AMSCNet")
        ax.bar(X + 0.40, Graph[:, 4], color='c', width=0.10, label="AO-FHO-AMSCNet")

        plt.xticks(X + 0.25, ('HEART DISEASE', 'PARKINSONS', 'DIABETES', 'LUNG CANCER'))
        plt.xlabel('DATASETS')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc=1)
        path1 = "./Results/Alg_bar_%s.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(4)
        ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="CNN")
        ax.bar(X + 0.10, Graph[:, 6], color='#cc9f3f', width=0.10, label="TCN")
        ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="ALSTM")
        ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="VAE+ALSTM+TCN+BYSN")
        ax.bar(X + 0.40, Graph[:, 9], color='c', width=0.10, label="AO-FHO-AMSCNet")

        plt.xticks(X + 0.25, ('HEART DISEASE', 'PARKINSONS', 'DIABETES', 'LUNG CANCER'))
        plt.xlabel('DATASETS')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc=1)
        path1 = "./Results/Met_bar_%s.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()


def plot_results_kfold():
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0]

    Dataset = ['DATASET-1', 'DATASET-2', 'DATASET-3', 'DATASET-4']
    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    learnper = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Algorithm = ['TERMS', 'COA', 'BMO', 'CHOA', 'FHO', 'PROPOSED']
    Classifier = ['TERMS', 'CNN', 'TCN', 'ALSTM', 'VAE_ALSTM_TCN_BYSN', 'PROPOSED']
    for i in range(eval.shape[0]):

        value = eval[i, 0, :, 4:]
        value[:, :-1] = value[:, :-1] * 100
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('-------------------------------------------------- ', Dataset[i],
              'Algorithm Comparison --------------------------------------------------')
        print(Table)

        Table = PrettyTable()

        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- ', Dataset[i],
              'Classifier Comparison --------------------------------------------------')
        print(Table)

    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            # Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if Graph_Term[j] == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100

            plt.plot(learnper, Graph[:, 0], color='#ff5b00', linewidth=3, marker='o', markerfacecolor='m',markersize=12,label="COA-AMSCNet")
            plt.plot(learnper, Graph[:, 1], color='#89fe05', linewidth=3, marker='o', markerfacecolor='y',markersize=12,label="BMO-AMSCNet")
            plt.plot(learnper, Graph[:, 2], color='#990eea', linewidth=3, marker='o', markerfacecolor='k',markersize=12,label="CHOA-AMSCNet")
            plt.plot(learnper, Graph[:, 3], color='#ff000d', linewidth=3, marker='o', markerfacecolor='b',markersize=12,label="FHO-AMSCNet")
            plt.plot(learnper, Graph[:, 4], color='#ffdf22', linewidth=3, marker='o', markerfacecolor='r',markersize=12,label="AO-FHO-AMSCNet")
            plt.xlabel('K - Fold')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=4)
            path1 = "./Results/Dataset_%s_%s_line_2.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(10)
            ax.bar(X + 0.00, Graph[:, 5], color='#fedf08', width=0.10, label="CNN")
            ax.bar(X + 0.10, Graph[:, 6], color='#cc9f3f', width=0.10, label="TCN")
            ax.bar(X + 0.20, Graph[:, 7], color='#ca0147', width=0.10, label="ALSTM")
            ax.bar(X + 0.30, Graph[:, 8], color='#0d75f8', width=0.10, label="VAE+ALSTM+TCN+BYSN")
            ax.bar(X + 0.40, Graph[:, 9], color='#00fbb0', width=0.10, label="AO-FHO-AMSCNet")
            plt.xticks(X + 0.25, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
            plt.xlabel('K - Fold')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_%s_%s_bar_2.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def plot_convergence():
    convergence = np.load('Fitness.npy', allow_pickle=True)
    Algorithm1 = ['TERMS', 'COA', 'BMO', 'CHOA', 'FHO', 'PROPOSED']
    Terms1 = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']

    for i in range(len(convergence)):
        plt.plot(convergence[i, 0, :], color='r', linewidth=3,
                 label="COA-AMSCNet")
        plt.plot(convergence[i, 1, :], color='g', linewidth=3,
                 label="BMO-AMSCNet")
        plt.plot(convergence[i, 2, :], color='b', linewidth=3,
                 label="CHOA-AMSCNet")
        plt.plot(convergence[i, 3, :], color='y', linewidth=3,
                 label="FHO-AMSCNet")
        plt.plot(convergence[i, 4, :], color='k', linewidth=3,
                 label="AO-FHO-AMSCNet")
        plt.xlabel('Iterations')
        plt.ylabel('Cost Function')
        plt.legend(loc="best")
        path1 = "./Results/Convergence_%s.png" % (str(i + 1))
        plt.savefig(path1)
        plt.show()

        a1 = np.zeros((5, 25))
        print('Statistical Analysis')
        Table = PrettyTable()
        # Table.add_column('TERMS', Terms1[0:])
        a1[0, :] = convergence[i, 0, :]
        a1[1, :] = convergence[i, 1, :]
        a1[2, :] = convergence[i, 2, :]
        a1[3, :] = convergence[i, 3, :]
        a1[4, :] = convergence[i, 4, :]

        Table.add_column(Algorithm1[0], Terms1)
        for k in range(len(Algorithm1) - 1):
            Table.add_column(Algorithm1[k + 1], np.asarray(statistical_analysis(a1[k, :])))
        print('-------------------------------------------------- Statistical Analysis ',
              '--------------------------------------------------')
        print(Table)
        print()



def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a

if __name__ == "__main__":
    plot_results_learnperc()
    plot_results_kfold()
    plot_convergence()
