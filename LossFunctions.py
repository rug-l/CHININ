from IMPORTS import *
#from RACM_ML_OSTI_config import spcnames, spcnames_imp, spcids
from Small_ML_OSTI_config import spcnames, spcnames_imp, spcids

# ozone-focused loss function
o3 = spcids['O3']
def MSE_focus_o3(pred, target, focus_factor=20):
    pred_o3   = torch.cat((pred, pred[o3:o3+1]*focus_factor))
    target_o3 = torch.cat((target, target[o3:o3+1]*focus_factor))
    return torch.mean(torch.square(pred_o3 - target_o3))

# in work: importance-equalizing loss function
# important: /max is incomplete, the range has to be adjusted instead of max only
# but this gets close to MinMax scaling
def MSE_equalizer(pred, target, dat_minmax):
    names_loc = spcnames if pred.size()==95 else spcnames_imp
    pred_eq_imp   = pred
    target_eq_imp = target
    for iSpc, spc in enumerate(names_loc):
        pred_eq_imp[iSpc]   *= 100/dat_minmax["conc"][spc][1]
        target_eq_imp[iSpc] *= 100/dat_minmax["conc"][spc][1]
    return torch.mean(torch.square(pred_eq_imp - target_eq_imp))

# in work: relative loss function
# considers relative error instead of absolute
# IMPORTANT: this function needs fitting data! no errors catched (e.g. target<<0 can get problematic)
def MSE_equalizer2(pred, target):
    return torch.mean(torch.square(1-(torch.exp(pred)/torch.exp(target))))
