import ROOT
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from root_numpy import root2array, rec2array, tree2array
import xgboost as xgb
import numpy as np
from sklearn.cross_validation import train_test_split



branch_names = """Jet_genjetPt_nu, Jet_pt, nPVs, Jet_eta, Jet_mt, Jet_leadTrackPt, Jet_leptonPtRel_new, Jet_leptonPt, Jet_leptonDeltaR,                                              
  Jet_neHEF, Jet_neEmEF, Jet_PFMET, Jet_METDPhi, Jet_JetDR""".split(",")

branch_names = [c.strip() for c in branch_names]
#branch_names = (b.replace(" ", "_") for b in branch_names)                                                                                                                          
#branch_names = list(b.replace("-", "_") for b in branch_names)                                                                                                                      
data = root2array("/afs/cern.ch/work/p/pmendiol/TTbar_TMVA/minitree_ZH/minitree_ZH_trailing_12_7.root", "jet", branch_names)

data = rec2array(data)


#a = data[:,0]/data[:,1] #target = Jet_genjetPt_nu/Jet_pt                                                                                                                            

a= np.divide(data[:,0],data[:,1]) #target                                                                                                                                            


print(a[1:5])


X = data[:,1:] #training variables       
y = a #target                                                                                                                                                                        

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test, label=y_test)


#NEEDS PARAMETER TUNING!

params = {'base_score':1.0, 'colsample_bylevel':1, 'colsample_bytree':1,'gamma':0,'learning_rate':0.2, 'max_delta_step':0,
'max_depth':10,'min_child_weight':0.02, 'missing':None, 'n_estimators':700, 'nthread':2,'objective':'reg:linear',
'reg_alpha':0, 'reg_lambda':1.0,'scale_pos_weight':1, 'seed':0, 'silent':True, 'subsample':1.0}

train = xgb.train(params, dtrain)

train.dump_model("dump.raw.txt")

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test, label=y_test)

params = {'base_score':1.0, 'colsample_bylevel':1, 'colsample_bytree':1,'gamma':0,'learning_rate':0.2, 'max_delta_step':0,
'max_depth':10,'min_child_weight':0.02, 'missing':None, 'n_estimators':700, 'nthread':2,'objective':'reg:linear',
'reg_alpha':0, 'reg_lambda':1.0,'scale_pos_weight':1, 'seed':0, 'silent':True, 'subsample':1.0}

train = xgb.train(params, dtrain)

train.dump_model("dump.raw.txt")

predict = train.predict(dtest)

print(predict)


xgb.plot_importance(train)
plt.savefig('importantfeature.pdf')
