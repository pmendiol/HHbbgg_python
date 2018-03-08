import ROOT
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from root_numpy import root2array, rec2array, tree2array
import xgboost as xgb
import numpy as np
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import datetime as datetime 
from scipy.stats import norm
import matplotlib.mlab as mlab

leading_trailing = "trailing"

directory = "/afs/cern.ch/work/p/pmendiol/Training_Macros/CW_TrainingTree/HHbbgg/Hbb/minitree_H2b2g_"+leading_trailing+"_CHF_v2.root"
what_tree = "jet"

branch_names_cw = """jet1jetGenJetPtR,jet2jetGenJetPtR,jjMass""".split(",")
branch_names_cw = [c.strip() for c in branch_names_cw]

branch_names_orig_xgboost = """Jet_genjetPt_nu,Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,                                       Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,Jet_vtx3deL_new,Jet_PFMET,                                                Jet_METDPhi,Jet_JetDR,Jet_neEmEF,Jet_neHEF""".split(",")

branch_names_CHFonly = """Jet_genjetPt_nu,Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,                                            
Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,Jet_vtx3deL_new,Jet_PFMET,
Jet_METDPhi,Jet_JetDR,Jet_CHF""".split(",")

branch_names_CHFplusneEmEF = """Jet_genjetPt_nu,Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,                                      Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,Jet_vtx3deL_new,Jet_PFMET,
Jet_METDPhi,Jet_JetDR,Jet_CHFplusneEmEF""".split(",")

branch_names_totHEF = """Jet_genjetPt_nu,Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,                                             
Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,Jet_vtx3deL_new,Jet_PFMET,
Jet_METDPhi,Jet_JetDR,Jet_totHEF""".split(",")


branch_names_orig_xgboost = [c.strip() for c in branch_names_orig_xgboost]
branch_names_CHFonly = [c.strip() for c in branch_names_CHFonly]
branch_names_CHFplusneEmEF = [c.strip() for c in branch_names_CHFplusneEmEF]
branch_names_totHEF = [c.strip() for c in branch_names_totHEF]

###########################################################################################

data_cw = root2array("/data7/cyeh/Summer16_BjReg/TestTree/2b2g/minitree*.root", "jet", branch_names_cw)
data_cw = rec2array(data_cw)

data_cw_reg = root2array("/data7/cyeh/Summer16_BjReg/TestTree/2b2g/minitree*.root","jet_15plus3_js_2_17", branch_names_cw)
data_cw_reg = rec2array(data_cw_reg)

data_orig_xgboost = root2array("/afs/cern.ch/work/p/pmendiol/Training_Macros/CW_TrainingTree/HHbbgg/Hbb/minitree_H2b2g_"+leading_trailing+"_CHF_v2.root", "jet", branch_names_orig_xgboost)

data_CHFonly = root2array(directory, what_tree, branch_names_CHFonly)
data_CHFplusneEmEF = root2array(directory, what_tree, branch_names_CHFplusneEmEF)
data_totHEF = root2array(directory, what_tree, branch_names_totHEF )



data_orig_xgboost = rec2array(data_orig_xgboost)
data_CHFonly = rec2array(data_CHFonly)
data_CHFplusneEmEF = rec2array(data_CHFplusneEmEF)
data_totHEF = rec2array(data_totHEF)

#############LOAD MODEL##############################

model_CHFonly = joblib.load("/afs/cern.ch/work/p/pmendiol/Training_Samples_CHF/No_Random_Seed/HHBBgg_CHFonly_"+leading_trailing+".pkl")
model_CHFplusneEmEF = joblib.load("/afs/cern.ch/work/p/pmendiol/Training_Samples_CHF/No_Random_Seed/HHBBgg_CHFplusneEmEF_"+leading_trailing+".pkl")
model_totHEF = joblib.load("/afs/cern.ch/work/p/pmendiol/Training_Samples_CHF/No_Random_Seed/HHBBgg_totHEF_"+leading_trailing+".pkl")

model_orig_xgboost = joblib.load("/afs/cern.ch/work/p/pmendiol/Training_Samples_CHF/No_Random_Seed/HHBBgg_orig_"+leading_trailing+".pkl")

#####################################################

X_CHFonly = data_CHFonly[:,1:-1]
X_CHFplusneEmEF = data_CHFplusneEmEF[:,1:-1]
X_totHEF = data_totHEF[:,1:-1]
X_orig_xgboost = data_orig_xgboost[:,1:-1]

y_CHFonly = np.divide(data_CHFonly[:,0],data_CHFonly[:,1])
y_CHFplusneEmEF = np.divide(data_CHFplusneEmEF[:,0],data_CHFplusneEmEF[:,1])
y_totHEF = np.divide(data_totHEF[:,0],data_totHEF[:,1])

#####################################################

y_pred_CHFonly = model_CHFonly.predict(X_CHFonly)
y_pred_CHFplusneEmEF = model_CHFplusneEmEF.predict(X_CHFplusneEmEF)
y_pred_totHEF = model_totHEF.predict(X_totHEF)

y_pred_orig_xgboost = model_orig_xgboost.predict(X_orig_xgboost)

#####################################################

if leading_trailing == 'leading':
   jet_reso_cw = data_cw[:,0]
   jet_reso_cw_reg = data_cw_reg[:,0]

else:
   jet_reso_cw = data_cw[:,1]
   jet_reso_cw_reg = data_cw_reg[:,1]
   
#####################################################

reso_CHFonly = (y_pred_CHFonly*data_CHFonly[:,1] - data_CHFonly[:,0]) / (data_CHFonly[:,0])
reso_CHFonly_noreg = (data_CHFonly[:,1] - data_CHFonly[:,0]) / (data_CHFonly[:,0])

reso_CHFplusneEmEF = (y_pred_CHFplusneEmEF*data_CHFplusneEmEF[:,1] - data_CHFplusneEmEF[:,0]) / (data_CHFplusneEmEF[:,0])
reso_CHFplusneEmEF_noreg = (data_CHFplusneEmEF[:,1] - data_CHFplusneEmEF[:,0]) / (data_CHFplusneEmEF[:,0])

reso_totHEF = (y_pred_totHEF*data_totHEF[:,1] - data_totHEF[:,0]) / (data_totHEF[:,0])
reso_totHEF_noreg = (data_totHEF[:,1] - data_totHEF[:,0]) / (data_totHEF[:,0])

reso_orig_xgboost = (y_pred_orig_xgboost*data_orig_xgboost[:,1] - data_orig_xgboost[:,0])/ (data_orig_xgboost[:,0])


#####################################################

#hist_params = {'normed'=1,'bins'=100,'range'=[-0.7,0.7],'histtype'='stepfilled','fill'=False}

print(reso_CHFonly)
print(reso_CHFplusneEmEF)
print(reso_totHEF)
print(jet_reso_cw)

plt.hist(reso_CHFonly,fill = False, histtype = 'stepfilled', normed = 1, bins = 100, range = [-0.7,0.7], edgecolor = 'red',label='CHFonly')
plt.hist(reso_CHFplusneEmEF, fill = False, histtype = 'stepfilled', normed = 1, bins = 100, range = [-0.7,0.7],edgecolor = 'blue', label='CHFplusneEmEF')
plt.hist(reso_totHEF, fill = False, histtype = 'stepfilled', normed = 1, bins = 100, range = [-0.7,0.7], edgecolor = 'green', label = 'totHEF')
plt.hist(jet_reso_cw_reg, fill = False, histtype = 'stepfilled', normed = 1, bins = 100, range = [-0.7,0.7], edgecolor = 'orange', label = 'reso_CW')
plt.hist(jet_reso_cw, fill = False, histtype = 'stepfilled', normed = 1, bins = 100, range = [-0.7,0.7], edgecolor = 'black', label = 'reso_NoReg')

plt.hist(reso_orig_xgboost, fill = False, histtype = 'stepfilled', normed = 1, bins = 100, range = [-0.7,0.7], edgecolor = 'yellow', label = 'orig_xgboost')
                                                                                       
plt.axvline(0, color='black', linestyle='dashed', linewidth=1)

plt.xlabel("(pt_reco - pt_gen)/pt_gen")

plt.ylabel("A. U.")

plt.legend(fancybox=True,loc='best',fontsize=8, shadow=True)

plt.savefig("Resolution_"+leading_trailing+".pdf", dpi=1000)
plt.savefig("Resolution_"+leading_trailing+".png", dpi=1000)
