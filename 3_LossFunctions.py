from IMPORTS import *


# ozone-focused loss function
o3 = spcids['O3']
def MSE_focus_o3(pred, target, focus_factor=20):
    pred_o3   = torch.cat((pred, pred[o3:o3+1]*focus_factor))
    target_o3 = torch.cat((target, target[o3:o3+1]*focus_factor))
    return torch.mean(torch.square(pred_o3 - target_o3))

# soon: importance-equalizing loss function
