import ROOT
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from root_numpy import root2array, rec2array, tree2array, array2root, array2tree
import xgboost as xgb
import numpy as np
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import datetime as datetime 
from scipy.stats import norm
import matplotlib.mlab as mlab


lead_trail = ["Leading", "Trailing"]

for a in range(len(lead_trail)):

	#================DATA IMPORT=============================

	directory = "/afs/cern.ch/work/p/pmendiol/CHF/NoRegression/2b2g_NoRegression_"+lead_trail[a]+"/*.root"
	what_tree = "jet"
	
	directory_cw = "/data7/cyeh/Summer16_BjReg/TestTree/2b2g/minitree*.root"
	what_tree_cw = "jet_15plus3_js_2_17"
	
	directory_noreg = "/afs/cern.ch/work/p/pmendiol/CHF/NoRegression/2b2g_NoRegression_"+lead_trail[a]+"/*.root"
	what_tree_noreg = "jet"
	
	#=================BRANCH NAMES===========================
	
	branch_names_xgb = """Jet_genjetPt_nu,Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,
	Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,
	Jet_vtx3deL_new,Jet_PFMET,Jet_METDPhi,Jet_JetDR,Jet_neHEF,Jet_neEmEF""".split(",")
	
	branch_names_chfonly = """Jet_genjetPt_nu,Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,
	Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,
	Jet_vtx3deL_new,Jet_PFMET,Jet_METDPhi,Jet_JetDR,Jet_CHF""".split(",")
	
	branch_names_chfp = """Jet_genjetPt_nu,Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,
	Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,
	Jet_vtx3deL_new,Jet_PFMET,Jet_METDPhi,Jet_JetDR,Jet_CHFplusneEmEF""".split(",")
	
	branch_names_tot = """Jet_genjetPt_nu,Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,
	Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,
	Jet_vtx3deL_new,Jet_PFMET,Jet_METDPhi,Jet_JetDR,Jet_totHEF""".split(",")
	
	branch_names_cw = """jet1jetGenJetPtR,jet2jetGenJetPtR,jjMass,jet1Pt,jet2Pt,jet1jetGenJetPt,jet2jetGenJetPt""".split(",")
	
	branch_names_noreg = """Jet_genjetPt_nu,Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,
	Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,
	Jet_vtx3deL_new,Jet_PFMET,Jet_METDPhi,Jet_JetDR,Jet_neHEF,Jet_neEmEF""".split(",")
	
	branch_names_xgb 	 = [c.strip() for c in branch_names_xgb]
	branch_names_chfonly = [c.strip() for c in branch_names_chfonly]
	branch_names_chfp 	 = [c.strip() for c in branch_names_chfp]
	branch_names_tot	 = [c.strip() for c in branch_names_tot]
	branch_names_cw		 = [c.strip() for c in branch_names_cw]
	branch_names_noreg   = [c.strip() for c in branch_names_noreg]
	
	#=================ROOT2ARRAY==============================
	
	data_xgb	= root2array(directory, what_tree, branch_names_xgb)
	data_chf 	= root2array(directory, what_tree, branch_names_chfonly)
	data_chfp 	= root2array(directory, what_tree, branch_names_chfp)
	data_tot	= root2array(directory, what_tree, branch_names_tot)
	data_cw		= root2array(directory_cw, what_tree_cw, branch_names_cw)
	data_noreg 	= root2array(directory_noreg, what_tree_noreg, branch_names_noreg)
	
	#=================REC2ARRAY===============================
	
	data_xgb	= rec2array(data_xgb)
	data_chf 	= rec2array(data_chf)
	data_chfp 	= rec2array(data_chfp)
	data_tot	= rec2array(data_tot)
	data_cw		= rec2array(data_cw)
	data_noreg 	= rec2array(data_noreg)
	
	#================LOAD WEIGHT FILES========================
	
	model_xgb  	= joblib.load("/afs/cern.ch/work/p/pmendiol/CHF/Weights/.pkl")
	model_chf  	= joblib.load("/afs/cern.ch/work/p/pmendiol/CHF/Weights/.pkl")
	model_chfp  = joblib.load("/afs/cern.ch/work/p/pmendiol/CHF/Weights/.pkl")
	model_tot  	= joblib.load("/afs/cern.ch/work/p/pmendiol/CHF/Weights/.pkl")
	
	#===============FEATURES=================================
	
	X_xgb	= data_xgb[:,1:-1]
	X_chf 	= data_chf[:,1:-1]
	X_chfp	= data_chfp[:,1:-1]
	X_tot	= data_tot[:,1:-1]
	
	#===============PREDICTIONS==============================
	
	pred_xgb	= model_xgb.predict(X_xgb)
	pred_chf	= model_chf.predict(X_chf)
	pred_chfp	= model_chfp.predict(X_chfp)
	pred_tot	= model_tot.predict(X_tot)
	
	#==============UNREGRESSED pTs==========================
	
	Pt_xgb	 = data_xgb[:,1]
	Pt_chf	 = data_chf[:,1]
	Pt_chfp	 = data_chfp[:,1]
	Pt_tot	 = data_tot[:,1]
	Pt_cw	 = data_cw[:,1]
	Pt_noreg = data_noreg[:,1]
	
	#=============GENJET pTs================================
	
	Pt_xgb_gen	 = data_xgb[:,0]
	Pt_chf_gen	 = data_chf[:,0]
	Pt_chfp_gen	 = data_chfp[:,0]
	Pt_tot_gen	 = data_tot[:,0]
	Pt_cw_gen	 = data_cw[:,0]
	Pt_noreg_gen = data_noreg[:,0]
		
	#==============REGRESSED pTs============================
	
	Pt_xgb_reg	 = pred_xgb*Pt_xgb
	Pt_chf_reg	 = pred_chf*Pt_chf
	Pt_chfp_reg	 = pred_chfp*Pt_chfp
	Pt_tot_reg	 = pred_tot*Pt_tot
	
	if lead_trail[a] == "Leading":
		Pt_cw_reg	= data_cw[:,3]
	else:
		Pt_cw_reg	= data_cw[:,4]
	
	#=============pT RESOLUTION============================
	
	PtR_xgb	  = (Pt_xgb_reg - Pt_xgb_gen)/Pt_xgb_gen
	PtR_chf	  = (Pt_chf_reg - Pt_chf_gen)/Pt_chf_gen
	PtR_chfp  = (Pt_chfp_reg - Pt_chf_gen)/Pt_chfp_gen
	PtR_tot   = (Pt_tot_reg - Pt_tot_gen)/Pt_tot_gen
	PtR_noreg = data_noreg[:,0]
	
	if lead_trail[a] == "Leading":
		PtR_cw   = data_cw[:,0]
	else:
		PtR_cw	 = data_cw[:,1]
	
	#===========VARIABLE NAMING============================
	
	Pt_xgb 		= np.array(Pt_xgb, dtype=[("Pt_xgb", np.float64)])
	Pt_chf 		= np.array(Pt_chf, dtype=[("Pt_chf", np.float64)])
	Pt_chfp		= np.array(Pt_chfp, dtype=[("Pt_chfp", np.float64)])
	Pt_tot		= np.array(Pt_tot, dtype=[("Pt_tot", np.float64)])
	Pt_cw		= np.array(Pt_cw, dtype=[("Pt_cw", np.float64)])
	Pt_noreg	= np.array(Pt_noreg, dtype=[("Pt_noreg", np.float64)])
	
	Pt_xgb_gen   = np.array(Pt_xgb_gen, dtype=[("Pt_xgb_gen", np.float64)])
	Pt_chf_gen	 = np.array(Pt_chf_gen, dtype=[("Pt_chf", np.float64)])
	Pt_chfp_gen	 = np.array(Pt_chfp_gen, dtype=[("Pt_chfp", np.float64)])
	Pt_tot_gen	 = np.array(Pt_tot_gen, dtype=[("Pt_tot", np.float64)])
	Pt_cw_gen	 = np.array(Pt_cw_gen, dtype=[("Pt_cw", np.float64)])
	Pt_noreg_gen = np.array(Pt_noreg_gen, dtype=[("Pt_noreg", np.float64)])
	
	Pt_xgb_reg 		= np.array(Pt_xgb_reg, dtype=[("Pt_xgb_reg", np.float64)])
	Pt_chf_reg 		= np.array(Pt_chf_reg, dtype=[("Pt_chf_reg", np.float64)])
	Pt_chfp_reg		= np.array(Pt_chfp_reg, dtype=[("Pt_chfp_reg", np.float64)])
	Pt_tot_reg		= np.array(Pt_tot_reg, dtype=[("Pt_tot_reg", np.float64)])
	Pt_cw_reg		= np.array(Pt_cw_reg, dtype=[("Pt_cw_reg", np.float64)])
	
	PtR_xgb 		= np.array(PtR_xgb, dtype=[("PtR_xgb", np.float64)])
	PtR_chf 		= np.array(PtR_chf, dtype=[("PtR_chf", np.float64)])
	PtR_chfp		= np.array(PtR_chfp, dtype=[("PtR_chfp", np.float64)])
	PtR_tot			= np.array(PtR_tot, dtype=[("PtR_tot", np.float64)])
	PtR_cw			= np.array(PtR_cw, dtype=[("PtR_cw", np.float64)])
	PtR_noreg		= np.array(PtR_noreg, dtype=[("PtR_noreg", np.float64)])
	
	#==============ARRAY2ROOT============================
	
	array2root(Pt_xgb, 'XGB_models.root', 'XGB', mode = 'recreate')
	array2root(Pt_chf, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_chfp, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_tot, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_cw, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_noreg, 'XGB_models.root', 'XGB', mode = 'update')
	
	array2root(Pt_xgb_gen, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_chf_gen, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_chfp_gen, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_tot_gen, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_cw_gen, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_noreg_gen, 'XGB_models.root', 'XGB', mode = 'update')
	
	array2root(Pt_xgb_reg, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_chf_reg, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_chfp_reg, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_tot_reg, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(Pt_cw_reg, 'XGB_models.root', 'XGB', mode = 'update')
	
	array2root(PtR_xgb, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(PtR_chf, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(PtR_chfp, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(PtR_tot, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(PtR_cw, 'XGB_models.root', 'XGB', mode = 'update')
	array2root(PtR_noreg, 'XGB_models.root', 'XGB', mode = 'update')
	