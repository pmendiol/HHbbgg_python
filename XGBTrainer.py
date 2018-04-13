import ROOT
from root_numpy import tree2array
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.externals import joblib
import datetime as datetime 
import pandas as pd


lead_trail = ["Leading", "Trailing"]

for a in range(len(lead_trail)):

	directory = "/data3/plpmendiola/2017HHbbgg/Training_Samples/26Mar2018_CHF_"+lead_trail[a]+".root"
	file = ROOT.TFile(directory)
	tree = file.Get("jet")

	CAT = ["XGB", "CHF", "CHFp", "tot"]
	
	for b in range(len(CAT)):
	
		if CAT[b] == "XGB":
			
			features_names = """Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,
			Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,
			Jet_vtx3deL_new,Jet_PFMET,Jet_METDPhi,Jet_JetDR,Jet_neHEF,Jet_neEmEF""".split(",")
		
		elif CAT[b] == "CHF":
			
			features_names = """Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,
			Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,
			Jet_vtx3deL_new,Jet_PFMET,Jet_METDPhi,Jet_JetDR,Jet_CHF""".split(",")
		
		elif CAT[b] == "CHFp":

			features_names = """Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,
			Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,
			Jet_vtx3deL_new,Jet_PFMET,Jet_METDPhi,Jet_JetDR,Jet_CHFplusneEmEF""".split(",")
		
		elif CAT[b] == "CHFp":

			features_names = """Jet_pt,nPVs,Jet_eta,Jet_mt,Jet_leadTrackPt,Jet_leptonPtRel_new,
			Jet_leptonPt,Jet_leptonDeltaR,Jet_vtxPt,Jet_vtxMass,Jet_vtx3dL,Jet_vtxNtrk,
			Jet_vtx3deL_new,Jet_PFMET,Jet_METDPhi,Jet_JetDR,Jet_totHEF""".split(",")
			
		Target_names = """Jet_genjetPt_nu,Jet_pt""".split(",")

		start = datetime.datetime.now()
		
		Target_names = [c.strip() for c in Target_names]
		features_names = [c.strip() for c in features_names]
		
		Target = tree2array(tree,Target_names)
		Features = tree2array(tree,features_names)
		
		Features = pd.DataFrame(Features)
		Target = pd.DataFrame(Target)
		Target['Jet_genjetPt_nu/Jet_pt'] = Target['Jet_genjetPt_nu']/Target['Jet_pt']
		
		X = Features
		y = Target['Jet_genjetPt_nu/Jet_pt']
		
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
		
		if lead_trail[a] == "Leading":
		
			Model = XGBRegressor(base_score=1.0, colsample_bylevel=1, colsample_bytree=1,gamma=0,
			learning_rate=0.01, max_delta_step=0, max_depth=10,min_child_weight=0.02, 
			missing=None, n_estimators=700,objective='reg:linear', reg_alpha=0, reg_lambda=1.0,scale_pos_weight=1, 
			seed=0, silent=True, subsample=1.0)
			
		elif lead_trail[a] == "Trailing":
		
			Model = XGBRegressor(base_score=1.0, colsample_bylevel=1, colsample_bytree=1,gamma=0,
			learning_rate=0.01, max_delta_step=0, max_depth=10,min_child_weight=0.02, 
			missing=None, n_estimators=700,objective='reg:linear', reg_alpha=1, reg_lambda=1.0,scale_pos_weight=1, 
			seed=0, silent=True, subsample=1.0)
		
		Fit = Model.fit(X_train, y_train)
		Predict = Model.predict(X_test)
		
		end = datetime.datetime.now()
		
		joblib.dump(Model, CAT[b]+"_"+lead_trail[a]+".pkl", compress=9)
		
		print 'Training Done. Time Cost for'+CAT[b]+'_'+lead_trail[a]+'(in seconds): %d' % ((end - start).seconds)