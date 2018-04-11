import ROOT
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from root_numpy import root2array, rec2array, tree2array, array2root, array2tree
import xgboost as xgb
import numpy as np
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.externals import joblib
from ROOT import TLorentzVector


type = ["G", "R", "N"]

for a in range(len(type)):  

	if type[a] == "G":
		point = ["m250","m260","m270","m280","m300","m320","m340","m350","m400","m450","m500","m550","m600","m650","m700","m750","m800","m900","m1000"]
	elif type[a] == "R":
		point = ["m250","m260","m270","m280","m300","m320","m340","m350","m400","m450","m500","m550","m600","m650","m700","m750","m800","m900"]
	else:
		point = ["node0","node1","node2","node3","node4","node5","node6","node7","node8","node9","node10","node11","node12","node12","node13"]

	for b in range(len(point)):

		#================DATA IMPORT=============================
		directory = ROOT.TFile("/data3/plpmendiola/2017HHbbgg/2b2g_NoRegression/minitree_13TeV_"+type[a]+"_"+point[b]+".root")
		what_tree = directory.Get("jet")

		directory_cw = ROOT.TFile("/data3/plpmendiola/2017HHbbgg/2b2g/minitree_13TeV_"+type[a]+"_"+point[b]+".root")
		what_tree_cw = directory.Get("jet_15plus3_js_2_17")

		#directory_noreg = "/data3/plpmendiola/2017HHbbgg/2b2g_NoRegression/minitree_13TeV_"+type[a]+"_"+point[b]+".root"
		#what_tree_noreg = "jet"

		#=================BRANCH NAMES===========================

		#LEADING=================================================
		branch_names_xgb1 = """jet1jetGenJetPt,jet1Pt,nPVs,jet1Eta,Jet1_mt,Jet1_leadTrackPt,Jet1_leptonPtRel,
		Jet1_leptonPt,Jet1_leptonDeltaR,Jet1_vtxPt,Jet1_vtxMass,Jet1_vtx3dL,Jet1_vtxNtrk,Jet1_vtx3deL,pfMET,
		Jet1_METDPhi,jjDR,Jet1_neHEF,Jet1_neEmEF,jet1En,jet1Phi,jet1Eta""".split(",")

		branch_names_chfonly1 = """jet1jetGenJetPt,jet1Pt,nPVs,jet1Eta,Jet1_mt,Jet1_leadTrackPt,Jet1_leptonPtRel,
		Jet1_leptonPt,Jet1_leptonDeltaR,Jet1_vtxPt,Jet1_vtxMass,Jet1_vtx3dL,Jet1_vtxNtrk,Jet1_vtx3deL,pfMET,Jet1_METDPhi,
		jjDR,Jet1_CHF,jet1En,jet1Phi,jet1Eta""".split(",")
		
		branch_names_chfp1 = """jet1jetGenJetPt,jet1Pt,nPVs,jet1Eta,Jet1_mt,Jet1_leadTrackPt,Jet1_leptonPtRel,
		Jet1_leptonPt,Jet1_leptonDeltaR,Jet1_vtxPt,Jet1_vtxMass,Jet1_vtx3dL,Jet1_vtxNtrk,Jet1_vtx3deL,pfMET,Jet1_METDPhi,
		jjDR,Jet1_CHFplusneEmEF,jet1En,jet1Phi,jet1Eta""".split(",") 

		branch_names_tot1 = """jet1jetGenJetPt,jet1Pt,nPVs,jet1Eta,Jet1_mt,Jet1_leadTrackPt,Jet1_leptonPtRel,
		Jet1_leptonPt,Jet1_leptonDeltaR,Jet1_vtxPt,Jet1_vtxMass,Jet1_vtx3dL,Jet1_vtxNtrk,Jet1_vtx3deL,pfMET,Jet1_METDPhi,
		jjDR,Jet1_totHEF,jet1En,jet1Phi,jet1Eta""".split(",")

		branch_names_noreg1 = """jet1jetGenJetPt,jet1Pt,nPVs,jet1Eta,Jet1_mt,Jet1_leadTrackPt,Jet1_leptonPtRel,
		Jet1_leptonPt,Jet1_leptonDeltaR,Jet1_vtxPt,Jet1_vtxMass,Jet1_vtx3dL,Jet1_vtxNtrk,Jet1_vtx3deL,pfMET,Jet1_METDPhi,
		jjDR,Jet1_neHEF,Jet1_neEmEF,jet1En,jet1Phi,jet1Eta""".split(",")
 
		#TRAILING==================================================

		branch_names_xgb2 = """jet2jetGenJetPt,jet2Pt,nPVs,jet2Eta,Jet2_mt,Jet2_leadTrackPt,Jet2_leptonPtRel,
		Jet2_leptonPt,Jet2_leptonDeltaR,Jet2_vtxPt,Jet2_vtxMass,Jet2_vtx3dL,Jet2_vtxNtrk,Jet2_vtx3deL,pfMET,Jet2_METDPhi,
		jjDR,Jet2_neHEF,Jet2_neEmEF,jet2En,jet2Phi,jet2Eta""".split(",")

		branch_names_chfonly2 = """jet2jetGenJetPt,jet2Pt,nPVs,jet2Eta,Jet2_mt,Jet2_leadTrackPt,Jet2_leptonPtRel,
		Jet2_leptonPt,Jet2_leptonDeltaR,Jet2_vtxPt,Jet2_vtxMass,Jet2_vtx3dL,Jet2_vtxNtrk,Jet2_vtx3deL,pfMET,Jet2_METDPhi,
		jjDR,Jet2_CHF,jet2En,jet2Phi,jet2Eta""".split(",")
		
		branch_names_chfp2 = """jet2jetGenJetPt,jet2Pt,nPVs,jet2Eta,Jet2_mt,Jet2_leadTrackPt,Jet2_leptonPtRel,
		Jet2_leptonPt,Jet2_leptonDeltaR,Jet2_vtxPt,Jet2_vtxMass,Jet2_vtx3dL,Jet2_vtxNtrk,Jet2_vtx3deL,pfMET,Jet2_METDPhi,
		jjDR,Jet2_CHFplusneEmEF,jet2En,jet2Phi,jet2Eta""".split(",") 

		branch_names_tot2 = """jet2jetGenJetPt,jet2Pt,nPVs,jet2Eta,Jet2_mt,Jet2_leadTrackPt,Jet2_leptonPtRel,
		Jet2_leptonPt,Jet2_leptonDeltaR,Jet2_vtxPt,Jet2_vtxMass,Jet2_vtx3dL,Jet2_vtxNtrk,Jet2_vtx3deL,pfMET,Jet2_METDPhi,
		jjDR,Jet2_totHEF,jet2En,jet2Phi,jet2Eta""".split(",")		
		
		branch_names_noreg2 = """jet2jetGenJetPt,jet2Pt,nPVs,jet2Eta,Jet2_mt,Jet2_leadTrackPt,Jet2_leptonPtRel,
		Jet2_leptonPt,Jet2_leptonDeltaR,Jet2_vtxPt,Jet2_vtxMass,Jet2_vtx3dL,Jet2_vtxNtrk,Jet2_vtx3deL,pfMET,Jet2_METDPhi,
		jjDR,Jet2_neHEF,Jet2_neEmEF,jet2En,jet2Phi,jet2Eta""".split(",")
		
		branch_names_cw = """jet1Pt,jet1Eta,jet1Phi,jet1bQPt,jet1genPt,jet1jetGenJetEn,jet1jetGenJetEta,jet1jetGenJetPhi,jet1jetGenJetPt,
		jet1jetGenJetPtR,jet1jetGenJetEn_ori,jet1jetGenJetEta_ori,jet1jetGenJetPhi_ori,jet1jetGenJetPt_ori,jet1jetGenJetPtR_ori,
		jet2Pt,jet2Eta,jet2Phi,jet2bQPt,jet2genPt,jet2jetGenJetEta,jet2jetGenJetPhi,jet2jetGenJetEn,jet2jetGenJetPt,
		jet2jetGenJetPtR,jet2jetGenJetEta_ori,jet2jetGenJetPhi_ori,jet2jetGenJetEn_ori,jet2jetGenJetPt_ori,jet2jetGenJetPtR_ori,
		jjMass,jjMassbQ,jjMassGenjet,jjMassjetGenjet,jjMassjetGenjet_ori,jjDR,Jet1_corr,Jet1_mt,Jet1_leadTrackPt,Jet1_leptonPtRel,
		Jet1_leptonPtRel_new,Jet1_leptonPt,Jet1_leptonDeltaR,Jet1_neHEF,Jet1_neEmEF,Jet1_chMult,Jet1_vtxPt,Jet1_vtxMass,Jet1_vtx3dL,
		Jet1_vtxNtrk,Jet1_vtx3deL_new,Jet1_METDPhi,Jet1_totHEF,Jet2_corr,Jet2_mt,Jet2_leadTrackPt,Jet2_leptonPtRel,Jet2_leptonPtRel_new,
		Jet2_leptonPt,Jet2_leptonDeltaR,Jet2_neHEF,Jet2_neEmEF,Jet2_chMult,Jet2_vtxPt,Jet2_vtxMass,Jet2_vtx3dL,Jet2_vtxNtrk,Jet2_vtx3deL,
		Jet2_vtx3deL_new,Jet2_METDPhi,Jet2_totHEF,nPVs,nGoodPVs,genMETPhi,pfMETPhi,genMET,pfMET,pfMET_T1JERUp,pfMET_T1JERDo,
		pfMET_T1JESUp,pfMET_T1JESDo,rho""".split(",")
		
		branch_names_xgb1 = [c.strip() for c in branch_names_xgb1]
		branch_names_chfonly1 = [c.strip() for c in branch_names_chfonly1]
		branch_names_chfp1 	 = [c.strip() for c in branch_names_chfp1]
		branch_names_tot1	 = [c.strip() for c in branch_names_tot1]
		branch_names_noreg1   = [c.strip() for c in branch_names_noreg1]
		
		branch_names_xgb2 	 = [c.strip() for c in branch_names_xgb2]
		branch_names_chfonly2 = [c.strip() for c in branch_names_chfonly2]
		branch_names_chfp2 	 = [c.strip() for c in branch_names_chfp2]
		branch_names_tot2	 = [c.strip() for c in branch_names_tot2] 
		branch_names_noreg2   = [c.strip() for c in branch_names_noreg2]

		branch_names_cw		 = [c.strip() for c in branch_names_cw] 


		#=================ROOT2ARRAY==============================
		'''
		data_xgb1	= root2array(directory, what_tree, branch_names_xgb1)
		data_chf1 	= root2array(directory, what_tree, branch_names_chfonly1)
		data_chfp1 	= root2array(directory, what_tree, branch_names_chfp1)
		data_tot1	= root2array(directory, what_tree, branch_names_tot1)
		data_noreg1 = root2array(directory, what_tree, branch_names_noreg1)
		
		data_xgb2	= root2array(directory, what_tree, branch_names_xgb2)
		data_chf2 	= root2array(directory, what_tree, branch_names_chfonly2)
		data_chfp2 	= root2array(directory, what_tree, branch_names_chfp2)
		data_tot2	= root2array(directory, what_tree, branch_names_tot2)
		data_noreg2 = root2array(directory, what_tree, branch_names_noreg2)
		
		data_cw		= root2array(directory_cw, what_tree_cw, branch_names_cw) 
		
		'''

		
		data_xgb1	= tree2array(what_tree, branch_names_xgb1)
		data_chf1	= tree2array(what_tree, branch_names_chfonly1)
		data_chfp1 	= tree2array(what_tree, branch_names_chfp1)
		data_tot1	= tree2array(what_tree, branch_names_tot1)
		data_cw		= tree2array(what_tree_cw, branch_names_cw1)
		data_noreg1 = tree2array(what_tree, branch_names_noreg1) 
		
		data_xgb2	= tree2array(what_tree, branch_names_xgb2)
		data_chf2	= tree2array(what_tree, branch_names_chfonly2)
		data_chfp2 	= tree2array(what_tree, branch_names_chfp2)
		data_tot2	= tree2array(what_tree, branch_names_tot2)
		data_cw		= tree2array(what_tree_cw, branch_names_cw2)
		data_noreg2 = tree2array(what_tree, branch_names_noreg2) 

		#================LOAD WEIGHT FILES========================

		model_xgb1  	= joblib.load("/data3/plpmendiola/2017HHbbgg/weights/xgb_model_Leading.pkl")
		model_chf1  	= joblib.load("/data3/plpmendiola/2017HHbbgg/weights/CHFonly_model_Leading.pkl")
		model_chfp1  = joblib.load("/data3/plpmendiola/2017HHbbgg/weights/CHFplusneEmEF_model_Leading.pkl")
		model_tot1  	= joblib.load("/data3/plpmendiola/2017HHbbgg/weights/totHEF_model_Leading.pkl")

		model_xgb2  	= joblib.load("/data3/plpmendiola/2017HHbbgg/weights/xgb_model_Trailing.pkl")
		model_chf2  	= joblib.load("/data3/plpmendiola/2017HHbbgg/weights/CHFonly_model_Trailing.pkl")
		model_chfp2  = joblib.load("/data3/plpmendiola/2017HHbbgg/weights/CHFplusneEmEF_model_Trailing.pkl")
		model_tot2  	= joblib.load("/data3/plpmendiola/2017HHbbgg/weights/totHEF_model_Trailing.pkl")

		#===============FEATURES=================================

		X_xgb1	= data_xgb1[:,1:-4] 
		X_chf1 	= data_chf1[:,1:-4]
		X_chfp1	= data_chfp1[:1:-4]
		X_tot1	= data_tot1[:,1:-4]
		
		print(X_xgb1)
		
		X_xgb2	= data_xgb2[:,1:-4]
		X_chf2 	= data_chf2[:,1:-4]
		X_chfp2	= data_chfp2[:,1:-4]
		X_tot2	= data_tot2[:,1:-4]

		#===============PREDICTIONS==============================

		pred_xgb1	= model_xgb1.predict(X_xgb1)
		pred_chf1	= model_chf1.predict(X_chf1)
		pred_chfp1	= model_chfp1.predict(X_chfp1)
		pred_tot1	= model_tot1.predict(X_tot1)
		
		pred_xgb2	= model_xgb2.predict(X_xgb2)
		pred_chf2	= model_chf2.predict(X_chf2)
		pred_chfp2	= model_chfp2.predict(X_chfp2)
		pred_tot2	= model_tot2.predict(X_tot2)

		#==============UNREGRESSED pTs==========================

		Pt_xgb1	 = data_xgb1['jet1Pt']
		Pt_chf1	 = data_chf1['jet1Pt']
		Pt_chfp1 = data_chfp1['jet1Pt']
		Pt_tot1	 = data_tot1['jet1Pt']
		Pt_cw1	 = data_cw['jet1Pt'] #ALREADY REGRESSED
		Pt_noreg1 = data_noreg1['jet1Pt']
		
		Pt_xgb2	 = data_xgb2['jet2Pt']
		Pt_chf2	 = data_chf2['jet2Pt']
		Pt_chfp2	 = data_chfp2['jet2Pt']
		Pt_tot2	 = data_tot2['jet2Pt']
		Pt_cw2	 = data_cw['jet2Pt'] #ALREADY REGRESSED
		Pt_noreg2 = data_noreg2['jet2Pt']

		#==============UNREGRESSED ENERGIES=====================
		
		En_xgb1 = data_xgb1['jet1En']
		En_chf1 = data_chf1['jet1En']
		En_chfp1= data_chfp1['jet1En']
		En_tot1 = data_tot1['jet1En']
		En_noreg1 = data_noreg1['jet1En']
		
		En_xgb2 = data_xgb2['jet2En']
		En_chf2 = data_chf2['jet2En']
		En_chfp2= data_chfp2['jet2En']
		En_tot2 = data_tot2['jet2En']
		En_noreg2 = data_noreg2['jet2En']

		#=============GENJET pTs================================

		Pt_noreg_gen1 = data_noreg['jet1jetGenJetPt']
		Pt_xgb_gen1	 = data_xgb['jet1jetGenJetPt']
		Pt_chf_gen1	 = data_chf['jet1jetGenJetPt']
		Pt_chfp_gen1 = data_chfp['jet1jetGenJetPt']
		Pt_tot_gen1	 = data_tot['jet1jetGenJetPt']
		Pt_cw_gen1	 = data_cw['jet1jetGenJetPt']
		

		Pt_noreg_gen2 = data_noreg['jet2jetGenJetPt']
		Pt_xgb_gen2	 = data_xgb['jet2jetGenJetPt']
		Pt_chf_gen2	 = data_chf['jet2jetGenJetPt']
		Pt_chfp_gen2	 = data_chfp['jet2jetGenJetPt']
		Pt_tot_gen2	 = data_tot['jet2jetGenJetPt']
		Pt_cw_gen2	 = data_cw['jet2jetGenJetPt']
		
		#==============REGRESSED pTs============================

		Pt_xgb_reg1	 = pred_xgb1*Pt_xgb1
		Pt_chf_reg1	 = pred_chf1*Pt_chf1
		Pt_chfp_reg1 = pred_chfp1*Pt_chfp1
		Pt_tot_reg1	 = pred_tot1*Pt_tot1
		Pt_cw_reg1	 = data_cw['jet1Pt']
		
		Pt_xgb_reg2	 = pred_xgb2*Pt_xgb2
		Pt_chf_reg2	 = pred_chf2*Pt_chf2
		Pt_chfp_reg2 = pred_chfp2*Pt_chfp2
		Pt_tot_reg2	 = pred_tot2*Pt_tot2
		Pt_cw_reg2	 = data_cw['jet2Pt']
		
		#==============REGRESSED ENERGIES=======================
		
		En_xgb_reg1 = pred_xgb1*En_xgb1
		En_chf_reg1 = pred_chf1*En_chf1
		En_chfp_reg1= pred_chfp1*En_chfp1
		En_tot_reg1 = pred_tot1*En_tot1
		
		En_xgb_reg2 = pred_xgb2*En_xgb2
		En_chf_reg2 = pred_chf2*En_chf2
		En_chfp_reg2= pred_chfp2*En_chfp2
		En_tot_reg2 = pred_tot2*En_tot2
		
		#==============DIJET INVARIANT MASS=======================
		
		XGB_vec1 = TLorentzVector(Pt_xgb_reg1,data_xgb1['jet1Eta'],data_xgb1['jet1Phi'],En_xgb_reg1)
		XGB_vec2 = TLorentzVector(Pt_xgb_reg2,data_xgb2['jet2Eta'],data_xgb2['jet2Phi'],En_xgb_reg2)
		XGB_vec = XGB_vec1 + XGB_vec2
		XGB_Mjj = XGB_vec.M()
		
		CHF_vec1 = TLorentzVector(Pt_chf_reg1,data_chf1['jet1Eta'],data_chf1['jet1Phi'],En_chf_reg1)
		CHF_vec2 = TLorentzVector(Pt_chf_reg2,data_chf2['jet2Eta'],data_chf2['jet2Phi'],En_chf_reg2)
		CHF_vec = CHF_vec1 + CHF_vec2
		CHF_Mjj = CHF_vec.M() 
		
		CHFp_vec1 = TLorentzVector(Pt_chfp_reg1,data_chfp1['jet1Eta'],data_chfp1['jet1Phi'],En_chfp_reg1)
		CHFp_vec2 = TLorentzVector(Pt_chfp_reg2,data_chfp2['jet2Eta'],data_chfp2['jet2Phi'],En_chfp_reg2)
		CHFp_vec  = CHFp_vec1 + CHFp_vec2
		CHFp_Mjj  = CHFp_vec.M()
		
		tot_vec1  = TLorentzVector(Pt_tot_reg1,data_tot1['jet1Eta'],data_tot1['jet1Phi'],En_tot_reg1)
		tot_vec2  = TLorentzVector(Pt_tot_reg2,data_tot2['jet2Eta'],data_tot2['jet2Phi'],En_tot_reg2)
		tot_vec   = tot_vec1 + tot_vec2
		tot_Mjj   = tot_vec.M()
		
		NoReg_vec1 = TLorentzVector(Pt_noreg1,data_noreg1['jet1Eta'],data_noreg1['jet1Phi'],En_noreg1)
		NoReg_vec2 = TLorentzVector(Pt_noreg2,data_noreg2['jet2Eta'],data_noreg2['jet2Phi'],En_noreg2)
		NoReg_vec  = NoReg_vec1 + NoReg_vec2
		NoReg_Mjj  = NoReg_vec.M()
		
		CW_Mjj	  = data_cw['jjMass']
		
		#=============pT RESOLUTION=============================

		PtR_xgb1   = (Pt_xgb_reg1 - Pt_noreg_gen1)/Pt_noreg_gen1
		PtR_chf1   = (Pt_chf_reg1 - Pt_noreg_gen1)/Pt_noreg_gen1
		PtR_chfp1  = (Pt_chfp_reg1 - Pt_noreg_gen1)/Pt_noreg_gen1
		PtR_tot1   = (Pt_tot_reg1 - Pt_noreg_gen1)/Pt_noreg_gen1
		PtR_noreg1 = (Pt_noreg1	- Pt_noreg_gen1)/Pt_noreg_gen1
		
		PtR_xgb2   = (Pt_xgb_reg2 - Pt_noreg_gen2)/Pt_noreg_gen2
		PtR_chf2   = (Pt_chf_reg2 - Pt_noreg_gen2)/Pt_noreg_gen2
		PtR_chfp2  = (Pt_chfp_reg2 - Pt_noreg_gen2)/Pt_noreg_gen2
		PtR_tot2   = (Pt_tot_reg2 - Pt_noreg_gen2)/Pt_noreg_gen2
		PtR_noreg2 = (Pt_noreg2	- Pt_noreg_gen2)/Pt_noreg_gen2

		PtR_cw1   = data_cw['jet1jetGenJetPtR']
		PtR_cw2	  = data_cw['jet2jetGenJetPtR']

		#===========VARIABLE NAMING============================

		Pt_xgb1 		= np.array(Pt_xgb1, dtype=[("Pt_xgb1", np.float64)])
		Pt_chf1 		= np.array(Pt_chf1, dtype=[("Pt_chf1", np.float64)])
		Pt_chfp1		= np.array(Pt_chfp1, dtype=[("Pt_chfp1", np.float64)])
		Pt_tot1		= np.array(Pt_tot1, dtype=[("Pt_tot1", np.float64)])
		Pt_cw1		= np.array(Pt_cw1, dtype=[("Pt_cw1", np.float64)])
		Pt_noreg1	= np.array(Pt_noreg1, dtype=[("Pt_noreg1", np.float64)])
		
		Pt_xgb_gen1   = np.array(Pt_xgb_gen1, dtype=[("Pt_xgb_gen1", np.float64)])
		Pt_chf_gen1	 = np.array(Pt_chf_gen1, dtype=[("Pt_chf_gen1", np.float64)])
		Pt_chfp_gen1	 = np.array(Pt_chfp_gen1, dtype=[("Pt_chfp_gen1", np.float64)])
		Pt_tot_gen1	 = np.array(Pt_tot_gen1, dtype=[("Pt_tot_gen1", np.float64)])
		Pt_cw_gen1	 = np.array(Pt_cw_gen1, dtype=[("Pt_cw_gen1", np.float64)])
		
		Pt_noreg_gen1 = np.array(Pt_noreg_gen1, dtype=[("Pt_noreg_gen1", np.float64)])

		Pt_xgb_reg1 		= np.array(Pt_xgb_reg1, dtype=[("Pt_xgb_reg1", np.float64)])
		Pt_chf_reg1 		= np.array(Pt_chf_reg1, dtype=[("Pt_chf_reg1", np.float64)])
		Pt_chfp_reg1		= np.array(Pt_chfp_reg1, dtype=[("Pt_chfp_reg1", np.float64)])
		Pt_tot_reg1		= np.array(Pt_tot_reg1, dtype=[("Pt_tot_reg1", np.float64)])
		Pt_cw_reg1		= np.array(Pt_cw_reg1, dtype=[("Pt_cw_reg1", np.float64)])

		PtR_xgb1 		= np.array(PtR_xgb1, dtype=[("PtR_xgb1", np.float64)])
		PtR_chf1 		= np.array(PtR_chf1, dtype=[("PtR_chf1", np.float64)])
		PtR_chfp1		= np.array(PtR_chfp1, dtype=[("PtR_chfp1", np.float64)])
		PtR_tot1			= np.array(PtR_tot1, dtype=[("PtR_tot1", np.float64)])
		PtR_cw1			= np.array(PtR_cw1, dtype=[("PtR_cw1", np.float64)])
		PtR_noreg1		= np.array(PtR_noreg1, dtype=[("PtR_noreg1", np.float64)])
		
		Pt_xgb2 		= np.array(Pt_xgb2, dtype=[("Pt_xgb2", np.float64)])
		Pt_chf2 		= np.array(Pt_chf2, dtype=[("Pt_chf2", np.float64)])
		Pt_chfp2		= np.array(Pt_chfp2, dtype=[("Pt_chfp2", np.float64)])
		Pt_tot2		= np.array(Pt_tot2, dtype=[("Pt_tot2", np.float64)])
		Pt_cw2		= np.array(Pt_cw2, dtype=[("Pt_cw2", np.float64)])
		Pt_noreg2	= np.array(Pt_noreg2, dtype=[("Pt_noreg2", np.float64)])
		
		Pt_xgb_gen2   = np.array(Pt_xgb_gen2, dtype=[("Pt_xgb_gen2", np.float64)])
		Pt_chf_gen2	 = np.array(Pt_chf_gen2, dtype=[("Pt_chf_gen2", np.float64)])
		Pt_chfp_gen2	 = np.array(Pt_chfp_gen2, dtype=[("Pt_chfp_gen2", np.float64)])
		Pt_tot_gen2	 = np.array(Pt_tot_gen2, dtype=[("Pt_tot_gen2", np.float64)])
		Pt_cw_gen2	 = np.array(Pt_cw_gen2, dtype=[("Pt_cw_gen2", np.float64)])
		
		Pt_noreg_gen2 		= np.array(Pt_noreg_gen2, dtype=[("Pt_noreg_gen2", np.float64)])

		Pt_xgb_reg2 		= np.array(Pt_xgb_reg2, dtype=[("Pt_xgb_reg2", np.float64)])
		Pt_chf_reg2 		= np.array(Pt_chf_reg2, dtype=[("Pt_chf_reg2", np.float64)])
		Pt_chfp_reg2		= np.array(Pt_chfp_reg2, dtype=[("Pt_chfp_reg2", np.float64)])
		Pt_tot_reg2		= np.array(Pt_tot_reg2, dtype=[("Pt_tot_reg2", np.float64)])
		Pt_cw_reg2		= np.array(Pt_cw_reg2, dtype=[("Pt_cw_reg2", np.float64)])

		PtR_xgb2 		= np.array(PtR_xgb2, dtype=[("PtR_xgb2", np.float64)])
		PtR_chf2 		= np.array(PtR_chf2, dtype=[("PtR_chf2", np.float64)])
		PtR_chfp2		= np.array(PtR_chfp2, dtype=[("PtR_chfp2", np.float64)])
		PtR_tot2		= np.array(PtR_tot2, dtype=[("PtR_tot2", np.float64)])
		PtR_cw2			= np.array(PtR_cw2, dtype=[("PtR_cw2", np.float64)])
		PtR_noreg2		= np.array(PtR_noreg2, dtype=[("PtR_noreg2", np.float64)])
		
		XGB_Mjj			= np.array(XGB_Mjj, dtype=[("XGB_Mjj", np.float64)])
		CHF_Mjj			= np.array(CHF_Mjj, dtype=[("CHF_Mjj", np.float64)])
		CHFp_Mjj		= np.array(CHFp_Mjj, dtype=[("CHFp_Mjj", np.float64)])
		tot_Mjj			= np.array(tot_Mjj, dtype=[("tot_Mjj", np.float64)])
		CW_Mjj			= np.array(CW_Mjj, dtype=[("CW_Mjj", np.float64)])
		NoReg_Mjj		= np.array(NoReg_Mjj, dtype=[("NoReg_Mjj", np.float64)])
		
		
		#==============ARRAY2ROOT============================ 
		 
		array2root(Pt_xgb, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_chf, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_chfp, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_tot, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_cw, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_noreg, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		'''
		array2root(Pt_xgb_gen, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_chf_gen, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_chfp_gen, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_tot_gen, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_cw_gen, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update') 
		'''
		array2root(Pt_noreg_gen, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')

		array2root(Pt_xgb_reg, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_chf_reg, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_chfp_reg, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_tot_reg, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(Pt_cw_reg, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')

		array2root(PtR_xgb, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(PtR_chf, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(PtR_chfp, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(PtR_tot, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(PtR_cw, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(PtR_noreg, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		
		array2root(XGB_Mjj, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(CHF_Mjj, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(CHFp_Mjj, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(tot_Mjj, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(NoReg_Mjj, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
		array2root(CW_Mjj, "minitree_13TeV_"+type[a]+"_"+point[b]+".root", 'XGB', mode = 'update')
