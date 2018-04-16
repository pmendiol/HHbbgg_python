import ROOT
from root_numpy import root2array, rec2array, tree2array, array2root, array2tree, tree2rec
import numpy as np
from numpy import*
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from ROOT import TLorentzVector
from root_pandas import read_root, to_root


REG_TYPE = ["XGB", "CHF", "CHFp", "tot"]

for reg_type in REG_TYPE:

	TYPE = ["G", "R", "N"]
	
	for type in TYPE:  
	
		if type == "G":
			POINT = ["m250","m260","m270","m280","m300","m320","m340","m350","m400","m450","m500","m550","m600","m650","m700","m750","m800","m900","m1000"]
		elif type == "R":
			POINT = ["m250","m260","m270","m280","m300","m320","m340","m350","m400","m450","m500","m550","m600","m650","m700","m750","m800","m900"]
		elif type =="N"
			POINT = ["node0","node1","node2","node3","node4","node5","node6","node7","node8","node9","node10","node11","node12","node12","node13"]
			
		for point in POINT:
		
			dir = "/data3/plpmendiola/2017HHbbgg/2b2g_NoRegression/minitree_13TeV_"+type+"_"+point+".root"
			tree = "jet"
			
			if reg_type == "XGB":
			
				features_names_1 = """jet1jetGenJetPt,jet1Pt,nPVs,jet1Eta,Jet1_mt,Jet1_leadTrackPt,Jet1_leptonPtRel,
				Jet1_leptonPt,Jet1_leptonDeltaR,Jet1_vtxPt,Jet1_vtxMass,Jet1_vtx3dL,Jet1_vtxNtrk,Jet1_vtx3deL,
				pfMET,Jet1_METDPhi,jjDR,Jet1_neHEF,Jet1_neEmEF""".split(",")
				
				feature_names_2 = """jet2jetGenJetPt,jet2Pt,nPVs,jet2Eta,Jet2_mt,Jet2_leadTrackPt,Jet2_leptonPtRel,
				Jet2_leptonPt,Jet2_leptonDeltaR,Jet2_vtxPt,Jet2_vtxMass,Jet2_vtx3dL,Jet2_vtxNtrk,Jet2_vtx3deL,
				pfMET,Jet2_METDPhi,jjDR,Jet2_neHEF,Jet2_neEmEF""".split(",")
				
				features_names_1 = [c.strip() for c in feature_names_1]
				features_names_2 = [c.strip() for c in feature_names_2]
				
				features_1 = read_root(dir, tree, columns=feature_names_1)
				features_2 = read_root(dir, tree, columns=feature_names_2)

				Model_1 = joblib.load("/data3/plpmendiola/2017HHbbgg/Training_Samples/Weights/"+reg_type+"_Leading.pkl") #BAGUHIN TO
				Model_2 = joblib.load("/data3/plpmendiola/2017HHbbgg/Training_Samples/Weights/"+reg_type+"_Trailing.pkl")
				
				w_1 = Model_1.predict(features_1)
				w_2 = Model_2.predict(features_2)
				
			elif reg_type == "CHF":
			
				features_names_1 = """jet1jetGenJetPt,jet1Pt,nPVs,jet1Eta,Jet1_mt,Jet1_leadTrackPt,Jet1_leptonPtRel,
				Jet1_leptonPt,Jet1_leptonDeltaR,Jet1_vtxPt,Jet1_vtxMass,Jet1_vtx3dL,Jet1_vtxNtrk,Jet1_vtx3deL,
				pfMET,Jet1_METDPhi,jjDR,Jet1_CHF""".split(",")
				
				feature_names_2 = """jet2jetGenJetPt,jet2Pt,nPVs,jet2Eta,Jet2_mt,Jet2_leadTrackPt,Jet2_leptonPtRel,
				Jet2_leptonPt,Jet2_leptonDeltaR,Jet2_vtxPt,Jet2_vtxMass,Jet2_vtx3dL,Jet2_vtxNtrk,Jet2_vtx3deL,
				pfMET,Jet2_METDPhi,jjDR,Jet2_CHF""".split(",")
				
				features_names_1 = [c.strip() for c in feature_names_1]
				features_names_2 = [c.strip() for c in feature_names_2]
				
				features_1 = read_root(dir, tree, columns=feature_names_1)
				features_2 = read_root(dir, tree, columns=feature_names_2)

				Model_1 = joblib.load("/data3/plpmendiola/2017HHbbgg/Training_Samples/Weights/"+reg_type+"_Leading.pkl") #BAGUHIN TO
				Model_2 = joblib.load("/data3/plpmendiola/2017HHbbgg/Training_Samples/Weights/"+reg_type+"_Trailing.pkl")
				
				w_1 = Model_1.predict(features_1)
				w_2 = Model_2.predict(features_2)
			
			elif reg_type == "CHFp":
			
				features_names_1 = """jet1jetGenJetPt,jet1Pt,nPVs,jet1Eta,Jet1_mt,Jet1_leadTrackPt,Jet1_leptonPtRel,
				Jet1_leptonPt,Jet1_leptonDeltaR,Jet1_vtxPt,Jet1_vtxMass,Jet1_vtx3dL,Jet1_vtxNtrk,Jet1_vtx3deL,
				pfMET,Jet1_METDPhi,jjDR,Jet1_CHFplusneEmEF""".split(",")
				
				feature_names_2 = """jet2jetGenJetPt,jet2Pt,nPVs,jet2Eta,Jet2_mt,Jet2_leadTrackPt,Jet2_leptonPtRel,
				Jet2_leptonPt,Jet2_leptonDeltaR,Jet2_vtxPt,Jet2_vtxMass,Jet2_vtx3dL,Jet2_vtxNtrk,Jet2_vtx3deL,
				pfMET,Jet2_METDPhi,jjDR,Jet2_CHFplusneEmEF""".split(",")
				
				features_names_1 = [c.strip() for c in feature_names_1]
				features_names_2 = [c.strip() for c in feature_names_2]
				
				features_1 = read_root(dir, tree, columns=feature_names_1)
				features_2 = read_root(dir, tree, columns=feature_names_2)


				Model_1 = joblib.load("/data3/plpmendiola/2017HHbbgg/Training_Samples/Weights/"+reg_type+"_Leading.pkl") #BAGUHIN TO
				Model_2 = joblib.load("/data3/plpmendiola/2017HHbbgg/Training_Samples/Weights/"+reg_type+"_Trailing.pkl")
				
				w_1 = Model_1.predict(features_1)
				w_2 = Model_2.predict(features_2)
			
			elif reg_type == "tot":
			
				features_names_1 = """jet1jetGenJetPt,jet1Pt,nPVs,jet1Eta,Jet1_mt,Jet1_leadTrackPt,Jet1_leptonPtRel,
				Jet1_leptonPt,Jet1_leptonDeltaR,Jet1_vtxPt,Jet1_vtxMass,Jet1_vtx3dL,Jet1_vtxNtrk,Jet1_vtx3deL,
				pfMET,Jet1_METDPhi,jjDR,Jet1_neHEF,Jet1_totHEF""".split(",")
				
				feature_names_2 = """jet2jetGenJetPt,jet2Pt,nPVs,jet2Eta,Jet2_mt,Jet2_leadTrackPt,Jet2_leptonPtRel,
				Jet2_leptonPt,Jet2_leptonDeltaR,Jet2_vtxPt,Jet2_vtxMass,Jet2_vtx3dL,Jet2_vtxNtrk,Jet2_vtx3deL,
				pfMET, Jet2_METDPhi,jjDR,Jet2_neHEF,Jet2_totHEF""".split(",")
				
				features_names_1 = [c.strip() for c in feature_names_1]
				features_names_2 = [c.strip() for c in feature_names_2]
				
				features_1 = read_root(dir, tree, columns=feature_names_1)
				features_2 = read_root(dir, tree, columns=feature_names_2)

				Model_1 = joblib.load("/data3/plpmendiola/2017HHbbgg/Training_Samples/Weights/"+reg_type+"_Leading.pkl") #BAGUHIN TO
				Model_2 = joblib.load("/data3/plpmendiola/2017HHbbgg/Training_Samples/Weights/"+reg_type+"_Trailing.pkl")
				
				w_1 = Model_1.predict(features_1)
				w_2 = Model_2.predict(features_2)
				
			
			VARS = read_root(dir,tree)
			
			VARS['jet1Pt'] = VARS['jet1Pt']*w_1
			VARS['jet2Pt'] = VARS['jet2Pt']*w_2
			
			VARS['jet1En'] = VARS['jet1En']*w_1
			VARS['jet2En'] = VARS['jet2En']*w_2
			
			VARS['jet1jetGenJetPtR'] = (VARS['jet1Pt'] - VARS['jet1jetGenJetPt'])/VARS['jet1jetGenJetPt']
			VARS['jet2jetGenJetPtR'] = (VARS['jet2Pt']- VARS['jet2jetGenJetPt'])/VARS['jet2jetGenJetPt']
			
			vec1 = TLorentzVector()
			vec2 = TLorentzVector()
			jjMass = []
			
			for i in VARS['jet1Pt'].index:
				vec1.SetPtEtaPhiE(VARS.iloc[i]['jet1Pt'],VARS.iloc[i]['jet1Eta'],VARS.iloc[i]['jet1Phi'],VARS.iloc[i]['jet1En'])
				vec2.SetPtEtaPhiE(VARS.iloc[i]['jet2Pt'],VARS.iloc[i]['jet2Eta'],VARS.iloc[i]['jet2Phi'],VARS.iloc[i]['jet2En'])
				jjMass.append((vec1+vec2).M())
				
			jjMass = pd.DataFrame(jjMass, columns = ['jjMass'])
			VARS['jjMass'] = jjMass
	
			VARS.to_root("minitree_13TeV_"+type+"_"+point+".root", key = reg_type)