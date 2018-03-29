import ROOT
from ROOT import TCut
from root_numpy import root2array, tree2array, array2root
import numpy as np

lead_trail = "Leading"

directory = ""

Filename = ROOT.TFile(directory)
Tree = Filename.Get("XGB")

#Leading
xmin_list = ['25', '75', '100', '125', '150', '200', '250']
xmax_list = ['75', '100','125', '150', '200', '250', '400']

#Trailing 
#xmin_list = ['20', '75', '100', '125', '150', '200'] 
#xmax_list = ['75', '100','125', '150', '200', '250']

for a, b in zip(range(len(xmin_list)), range(len(xmax_list))):

	xmin = xmin_list[a]
	xmax = xmax_list[b]

	branch_names = """Pt_xgb,Pt_chf,Pt_chfp,Pt_tot,Pt_cw,Pt_noreg,Pt_xgb_gen,Pt_chf_gen,
	Pt_chfp_gen,Pt_tot_gen,Pt_cw_gen,Pt_noreg_gen,Pt_xgb_reg,Pt_chf_reg,Pt_chfp_reg,Pt_tot_reg,
	Pt_cw_reg,PtR_xgb,PtR_chf,PtR_chfp,PtR_tot,PtR_cw,PtR_noreg""".split(",")
	
	what_tree = "xgb"
	variables = tree2array(Tree, branch_names, selection = 'Pt_'+what_tree+'_reg >'+xmin and 'Pt_'+what_tree+'_reg <'+xmax)
	PtR_xgb   = (variables['Pt_'+what_tree+'_reg'] - variables['Pt_'+what_tree+'_gen'])/variables['Pt_'+what_tree+'_gen']
	array2root(PtR_xgb, "Bins_"+xmin+"."+xmax+".root", "PtR_"+what_tree, mode="recreate")
	
	
	what_tree = "chf"
	Variables = tree2array(Tree, branch_names, selection = 'Pt_'+what_tree+'_reg >'+xmin and 'Pt_'+what_tree+'_reg <'+xmax)
	PtR_chf	  = (variables['Pt_'+what_tree+'_reg'] - variables['Pt_'+what_tree+'_gen'])/variables['Pt_'+what_tree+'_gen'] 
	array2root(PtR_chf, "Bins_"+xmin+"."+xmax+".root", "PtR_"+what_tree, mode="update")
	
	
	what_tree = "chfp"
	Variables = tree2array(Tree, branch_names, selection = 'Pt_'+what_tree+'_reg >'+xmin and 'Pt_'+what_tree+'_reg <'+xmax)
	PtR_chfp  = (variables['Pt_'+what_tree+'_reg'] - variables['Pt_'+what_tree+'_gen'])/variables['Pt_'+what_tree+'_gen'] 
	array2root(PtR_chfp, "Bins_"+xmin+"."+xmax+".root", "PtR_"+what_tree, mode="update")
	
	
	what_tree = "tot"
	Variables = tree2array(Tree, branch_names, selection = 'Pt_'+what_tree+'_reg >'+xmin and 'Pt_'+what_tree+'_reg <'+xmax)
	PtR_tot   = (variables['Pt_'+what_tree+'_reg'] - variables['Pt_'+what_tree+'_gen'])/variables['Pt_'+what_tree+'_gen'] 
	array2root(PtR_tot, "Bins_"+xmin+"."+xmax+".root", "PtR_"+what_tree, mode="update")
	
	
	what_tree = "cw"
	Variables = tree2array(Tree, branch_names, selection = 'Pt_'+what_tree+'_reg >'+xmin and 'Pt_'+what_tree+'_reg <'+xmax)
	PtR_cw    = (variables['Pt_'+what_tree+'_reg'] - variables['Pt_'+what_tree+'_gen'])/variables['Pt_'+what_tree+'_gen']
	array2root(PtR_cw, "Bins_"+xmin+"."+xmax+".root", "PtR_"+what_tree, mode="update")
	
	what_tree = "noreg"
	Variables = tree2array(Tree, branch_names, selection = 'Pt_noreg >'+xmin and 'Pt_noreg <'+xmax)
	PtR_noreg = (variables['Pt_'+what_tree+'_reg'] - variables['Pt_'+what_tree+'_gen'])/variables['Pt_'+what_tree+'_gen'] 
	array2root(PtR_noreg, "Bins_"+xmin+"."+xmax+".root", "PtR_"+what_tree, mode="update")
		
		
		