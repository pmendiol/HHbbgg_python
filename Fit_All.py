import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ROOT import gSystem
gSystem.Load('libRooFit')
from ROOT import RooFit, RooRealVar, RooBukinPdf, RooDataSet, RooArgList, RooArgSet, RooTreeData
from ROOT import TCanvas, TH1F
from ROOT import *
from root_numpy import tree2array, fill_hist, array2tree 
import numpy as np

lead_trail = "Trailing"  
 
directory = "/afs/cern.ch/work/p/pmendiol/CHF/Regressed/XGB_models_"+lead_trail+".root"


filename = ROOT.TFile(directory)
Tree = filename.Get("XGB")


chfonly = tree2array(Tree, branches="PtR_chf")
chfplusneemef = tree2array(Tree, branches="PtR_chfp")
tothef = tree2array(Tree, branches="PtR_tot")
xgb = tree2array(Tree, branches="PtR_xgb")
cw_reg = tree2array(Tree, branches="PtR_cw")
cw_noreg = tree2array(Tree, branches="PtR_noreg")

chfonly	= np.array(chfonly, dtype=[("chfonly", np.float64)])
chfplusneemef	= np.array(chfplusneemef, dtype=[("chfplusneemef", np.float64)])
tothef	= np.array(tothef, dtype=[("tothef", np.float64)])
xgb	= np.array(xgb, dtype=[("xgb", np.float64)])
cw_reg	= np.array(cw_reg, dtype=[("cw_reg", np.float64)])
cw_noreg	= np.array(cw_noreg, dtype=[("cw_noreg", np.float64)])



#=============HIST==========================
CHFonly = TH1F('CHFonly', '', 40, -1,1)
CHFplusneEmEF = TH1F('CHFplusneEmEF', '', 40, -1,1)
totHEF = TH1F('totHEF', '', 40, -1,1)
Orig_Xgboost = TH1F('orig_xgboost', '',40, -1,1)
CW_reg = TH1F('CW_reg', '',40, -1,1)
CW_noreg = TH1F('CW_noreg', '',40, -1,1) 

#=============FILL HIST=====================
CHFonly_hist=fill_hist(CHFonly, chfonly)
CHFplusneEmEF_hist = fill_hist(CHFplusneEmEF, chfplusneemef)
totHEF_hist = fill_hist(totHEF, tothef)
Orig_Xgboost_hist = fill_hist(Orig_Xgboost, xgb)
CW_reg_hist = fill_hist(CW_reg, cw_reg)
CW_noreg_hist = fill_hist(CW_noreg, cw_noreg)




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

hist = CW_noreg
hist_var = 'Noreg'

Bukin = RooBukinPdf("Bukin_fit", "", x,xp,sigp,xi,rho1,rho2)
Roo_Hist = RooDataHist("Roo_Hist", "Roo_Hist", RooArgList(x), hist)


#c = TCanvas("c", "", 900, 600)
#c.cd()

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
 
output = open("params_"+str(hist_var)+".txt", "wb")
output.write(str(out_xp)+'\n'+str(out_sigp)+'\n'+str(vxp_err)+'\n'+str(vsigp_err))
output.close()

c = TCanvas("c", "", 900, 600)
c.cd()
xframe1 = x.frame(ROOT.RooFit.Title(""))
Bukin.plotOn(xframe1)
xframe1.Draw()
#c.SaveAs("fit_"+str(hist_var)+".pdf")
#c.SaveAs("fit_"+str(hist_var)+".png")

##########################################################

hist = CW_reg
hist_var = 'CW_reg'

Bukin = RooBukinPdf("Bukin_fit", "", x,xp,sigp,xi,rho1,rho2)
Roo_Hist = RooDataHist("Roo_Hist", "Roo_Hist", RooArgList(x), hist)

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

output = open("params_"+str(hist_var)+".txt", "wb")
output.write(str(out_xp)+'\n'+str(out_sigp)+'\n'+str(vxp_err)+'\n'+str(vsigp_err))
output.close()

c.cd()
                                                                                         
Bukin.plotOn(xframe1, ROOT.RooFit.LineColor(2))
xframe1.Draw()
#c.SaveAs("fit_"+str(hist_var)+".pdf")
#c.SaveAs("fit_"+str(hist_var)+".png")

#################################################################

hist = Orig_Xgboost
hist_var = 'xgb'

Bukin = RooBukinPdf("Bukin_fit", "", x,xp,sigp,xi,rho1,rho2)
Roo_Hist = RooDataHist("Roo_Hist", "Roo_Hist", RooArgList(x), hist)

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

output = open("params_"+str(hist_var)+".txt", "wb")
output.write(str(out_xp)+'\n'+str(out_sigp)+'\n'+str(vxp_err)+'\n'+str(vsigp_err))
output.close()

c.cd()

Bukin.plotOn(xframe1, ROOT.RooFit.LineColor(1))
xframe1.Draw()

########################################################

hist = CHFonly
hist_var = 'chf'

Bukin = RooBukinPdf("Bukin_fit", "", x,xp,sigp,xi,rho1,rho2)
Roo_Hist = RooDataHist("Roo_Hist", "Roo_Hist", RooArgList(x), hist)

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

output = open("params_"+str(hist_var)+".txt", "wb")
output.write(str(out_xp)+'\n'+str(out_sigp)+'\n'+str(vxp_err)+'\n'+str(vsigp_err))
output.close()

c.cd()

Bukin.plotOn(xframe1, ROOT.RooFit.LineColor(3))
xframe1.Draw()

##########################################################

hist = CHFplusneEmEF
hist_var = 'chfpl'

Bukin = RooBukinPdf("Bukin_fit", "", x,xp,sigp,xi,rho1,rho2)
Roo_Hist = RooDataHist("Roo_Hist", "Roo_Hist", RooArgList(x), hist)

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

output = open("params_"+str(hist_var)+".txt", "wb")
output.write(str(out_xp)+'\n'+str(out_sigp)+'\n'+str(vxp_err)+'\n'+str(vsigp_err))
output.close()

c.cd()

Bukin.plotOn(xframe1, ROOT.RooFit.LineColor(5))
xframe1.Draw()


############################################################

hist = totHEF
hist_var = 'tothef'

Bukin = RooBukinPdf("Bukin_fit", "", x,xp,sigp,xi,rho1,rho2)
Roo_Hist = RooDataHist("Roo_Hist", "Roo_Hist", RooArgList(x), hist)

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

output = open("params_"+str(hist_var)+".txt", "wb")
output.write(str(out_xp)+'\n'+str(out_sigp)+'\n'+str(vxp_err)+'\n'+str(vsigp_err))
output.close()

c.cd()

Bukin.plotOn(xframe1, ROOT.RooFit.LineColor(6))
xframe1.Draw()

c.SaveAs(lead_trail+".pdf")
c.SaveAs(lead_trail+".png")
