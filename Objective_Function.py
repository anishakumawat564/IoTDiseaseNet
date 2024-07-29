import numpy as np
from Glob_Vars import Glob_Vars
from Model_VAE_ALSTM_TCN_BYSN import Model_VAE_ALSTM_TCN_BYSN

def Objfun(Soln):
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        Selected_Feature = Glob_Vars.Data[:, np.round(sol[0:10]).astype('int')]
        learnper = round(Selected_Feature.shape[0] * 0.75)
        train_data = Selected_Feature[learnper:, :]
        train_target = Glob_Vars.Target[learnper:, :]
        test_data = Selected_Feature[:learnper, :]
        test_target =  Glob_Vars.Target[:learnper, :]
        Eval = Model_VAE_ALSTM_TCN_BYSN(train_data, train_target, test_data, test_target, sol[10:].astype('int'))
        Fitn[i] = 1 / Eval[4]
    return Fitn


