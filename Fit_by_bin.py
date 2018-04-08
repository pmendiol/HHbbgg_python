import ROOT
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import RooFit, RooRealVar, RooBukinPdf, RooDataSet, RooArgList, RooArgSet, RooTreeData
from ROOT import TCanvas, TH1F
from ROOT import *
from root_numpy import tree2array, fill_hist


lead_trail = "Leading"

bin_range  = ['25.75', '75.100', '100.125', '125.150', '150.200', '200.250', '250.400'] #Leading

#bin_range  = ['20.75', '75.100', '100.125', '125.150', '150.200', '200.250'] #Trailing

for b in range(len(bin_range_lead)):

	directory = "/afs/cern.ch/work/p/pmendiol/CHF/Binning/Binned_"+lead_trail+"/Bins_"+bin_range[b]+".root"

	filename = ROOT.TFile(directory)

	Tree_CHFonly = filename.Get("PtR_chf")
	Tree_CHFplusneEmEF = filename.Get("PtR_chfp")
	Tree_totHEF = filename.Get("PtR_tot")
	Tree_xgb = filename.Get("PtR_xgb")
	Tree_CW = filename.Get("PtR_cw")
	Tree_noreg = filename.Get("PtR_noreg")

	chfonly = tree2array(Tree_CHFonly)
	chfplusneemef = tree2array(Tree_CHFplusneEmEF)
	tothef = tree2array(Tree_totHEF) 
	xgb = tree2array(Tree_xgb)
	cw_reg = tree2array(Tree_CW)
	cw_noreg = tree2array(Tree_noreg) 

	CHFonly = TH1F('CHFonly', '', 30, -1,1)
	CHFplusneEmEF = TH1F('CHFplusneEmEF', '', 30, -1,1)
	totHEF = TH1F('totHEF', '', 30, -1,1)
	XGB = TH1F('orig_xgboost', '',30, -1,1)
	CW = TH1F('CW_reg', '',30, -1,1)
	NoReg = TH1F('CW_noreg', '',30, -1,1)

	CHFonly_hist=fill_hist(CHFonly, chfonly)
	CHFplusneEmEF_hist = fill_hist(CHFplusneEmEF, chfplusneemef)
	totHEF_hist = fill_hist(totHEF, tothef)
	xgb_hist = fill_hist(XGB, xgb)
	CW_reg_hist = fill_hist(CW, cw_reg)
	CW_noreg_hist = fill_hist(NoReg, cw_noreg)


	hist_str = ['CHFonly', 'CHFplusneEmEF', 'totHEF', 'XGB', 'CW', 'NoReg']
	hist = [CHFonly, CHFplusneEmEF, totHEF, XGB, CW, NoReg]

	for d, k in zip(range(len(hist_str)), range(len(hist))):
	
		ROOT.RooMsgService.instance().setGlobalKillBelow(RooFit.WARNING);
		ROOT.RooMsgService.instance().setGlobalKillBelow(RooFit.FATAL);
		ROOT.RooMsgService.instance().setSilentMode(True);

		x = ROOT.RooRealVar("x", "", -1, 1)
		xp = ROOT.RooRealVar("xp", "",-0.3,0.3)
		sigp = ROOT.RooRealVar("sigp", "",0,0.5)
		xi = ROOT.RooRealVar("xi", "", -0.5,0.5)
		rho1 = ROOT.RooRealVar("rho1", "", -0.5,0.5)
		rho2 = ROOT.RooRealVar("rho2", "", -0.5,0.5)

		xp.setConstant(False) #peak position
		sigp.setConstant(False) #width
		xi.setConstant(False) #asymmetry parameter
		rho1.setConstant(False)
		rho2.setConstant(False)

		Bukin = RooBukinPdf("Bukin_fit", "", x,xp,sigp,xi,rho1,rho2)
		Roo_Hist = RooDataHist("Roo_Hist", "Roo_Hist", RooArgList(x), hist[k])


		c = TCanvas("c", "", 900, 600)

		Bukin.fitTo(Roo_Hist)


		vxp = xp.getVal()
		vsigp = sigp.getVal()
		vxi = xi.getVal()
		vrho1 = rho1.getVal()
		vrho2 = rho2.getVal()

		vxp_err = xp.getError()
		vsigp_err = sigp.getError()
		vxi_err = xi.getError()
		vrho1_err = rho1.getError()
		vrho2_err = rho2.getError()

		out_sigp = vsigp
		out_xp = vxp

		out_xp_error = vxp_err
		out_sigp_error = vsigp_err


		#NOTE: 
		#mean
		#sigma
		#mean_error
		#sigma_error

		Xp = open("Xp_"+hist_str[d]+"_"+lead_trail+".txt", "a")
		Sigp = open("Sigma_"+hist_str[d]+"_"+lead_trail+".txt", "a")
		Xp_err = open("XpErr_"+hist_str[d]+"_"+lead_trail+".txt", "a")
		Sigp_err = open("SigmaErr_"+hist_str[d]+"_"+lead_trail+".txt", "a")

		Xp.write(str(out_xp)+',\n')
		Sigp.write(str(out_sigp)+',\n')
		Xp_err.write(str(vxp_err)+',\n')
		Sigp_err.write(str(vsigp_err)+',\n')

		Xp.close()
		Sigp.close()
		Xp_err.close()
		Sigp_err.close()


		xframe = x.frame()
		Bukin.plotOn(xframe)
		xframe.Draw()
		c.SaveAs("Fit_"+str(bin_range[b])+"_"+hist_str[d]+"_"+lead_trail+".pdf")
		c.SaveAs("Fit_"+str(bin_range[b])+"_"+hist_str[d]+"_"+lead_trail+".png")
