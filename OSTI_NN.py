from IMPORTS import *
from NNfunctions import *

print("\n  OSTI-Network is initializing. Good learning!\n")

# #################################################### #
#                                                      #
#                        CONFIG                        #
#                                                      #
# #################################################### #

# load configuration from config file
#from SmallStratoML_OSTI_config import *
#from RACM_ML_OSTI_config import *
from Small_ML_OSTI_config import *

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

# #################################################### #
#                                                      #
#                   DATA PREPARING                     #
#                                                      #
# #################################################### #

# GET RAW DATA
conc, met, emis = get_data(BSP, nFiles, spcnames, metnames, emisnames, timepoints, val_perc, test_perc)


if Scaling=='MinMax':
    F_normal = minmax_scaler
    F_denormal = minmax_descaler
    norm_cutperc = 1E-16
elif Scaling=='log':
    F_normal = log_scaler
    F_denormal = log_descaler
    norm_cutperc = 1E-16
else:
    print("")
    print("Please choose a valid scaler. Current: ", Scaling, "  Available: MinMax, log")
    print("")
    sys.exit()

cut_perc=norm_cutperc
dat_minmax = {}
dat_minmax["conc"]={}
dat_minmax["met"]={}
dat_minmax["emis"]={}
for iSpc, spc in enumerate(spcnames):
    val_sorted = np.sort(np.concatenate((conc["train"][:,:,iSpc], conc["val"][:,:,iSpc], conc["test"][:,:,iSpc])).flatten())
    dat_minmax["conc"][spc] = [val_sorted[int(cut_perc*val_sorted.size)], val_sorted[int((1-cut_perc)*val_sorted.size)]]
for iSpc, spc in enumerate(metnames):
    val_sorted = np.sort(np.concatenate((met["train"][:,:,iSpc], met["val"][:,:,iSpc], met["test"][:,:,iSpc])).flatten())
    dat_minmax["met"][spc] = [val_sorted[int(cut_perc*val_sorted.size)], val_sorted[int((1-cut_perc)*val_sorted.size)]]
for iSpc, spc in enumerate(emisnames):
    val_sorted = np.sort(np.concatenate((emis["train"][:,iSpc], emis["val"][:,iSpc], emis["test"][:,iSpc])).flatten())
    dat_minmax["emis"][spc] = [val_sorted[int(cut_perc*val_sorted.size)], val_sorted[int((1-cut_perc)*val_sorted.size)]]


# Data Scaling
for cat in conc.keys():
    for iSpc, spc in enumerate(spcnames):
        for i in range(conc[cat].shape[0]):
            for j in range(conc[cat].shape[1]):
                conc[cat][i,j,iSpc] = F_normal(conc[cat][i,j,iSpc],dat_minmax["conc"][spc][0],dat_minmax["conc"][spc][1])
        #conc[cat][:,:,iSpc] = np.array([F_normal(val,dat_minmax["conc"][spc][0],dat_minmax["conc"][spc][1]) for val in conc[cat][:,:,iSpc]])
    conc[cat] = torch.from_numpy(conc[cat]).float()
for cat in met.keys():
    if met_scaling:
        for iSpc, spc in enumerate(metnames):
            for i in range(met[cat].shape[0]):
                for j in range(met[cat].shape[1]):
                    met[cat][i,j,iSpc] = F_normal(met[cat][i,j,iSpc],dat_minmax["met"][spc][0],dat_minmax["met"][spc][1])
                    #met[cat][:,:,iSpc]  = np.array([F_normal(val,dat_minmax["met"][spc][0],dat_minmax["met"][spc][1]) for val in met[cat][:,:,iSpc]])
    met[cat] = torch.from_numpy(met[cat]).float()
for cat in emis.keys():
    for iSpc, spc in enumerate(emisnames):
        for i in range(emis[cat].shape[0]):
            emis[cat][i,iSpc] = F_normal(emis[cat][i,iSpc],dat_minmax["emis"][spc][0],dat_minmax["emis"][spc][1])
        #emis[cat][:,iSpc] = np.array([F_normal(val,dat_minmax["emis"][spc][0],dat_minmax["emis"][spc][1]) for val in emis[cat][:,iSpc]])
    emis[cat] = torch.from_numpy(emis[cat]).float()



conc_train = torch.empty_like(conc["train"], device=device1);conc_val = torch.empty_like(conc["val"], device=device1);conc_test  = torch.empty_like(conc["test"], device=device1);
met_train = torch.empty_like(met["train"], device=device1);met_val = torch.empty_like(met["val"], device=device1);met_test  = torch.empty_like(met["test"], device=device1);
emis_train = torch.empty_like(emis["train"], device=device1);emis_val = torch.empty_like(emis["val"], device=device1);emis_test  = torch.empty_like(emis["test"], device=device1);

conc_train[:,:,:] = conc["train"][:,:,:]; conc_val[:,:,:] = conc["val"][:,:,:]; conc_test[:,:,:] = conc["test"][:,:,:];
met_train[:,:,:] = met["train"][:,:,:]; met_val[:,:,:] = met["val"][:,:,:]; met_test[:,:,:] = met["test"][:,:,:];
emis_train[:,:] = emis["train"][:,:]; emis_val[:,:] = emis["val"][:,:]; emis_test[:,:] = emis["test"][:,:];

ntrainfiles = conc_train.shape[0]
nvalfiles = conc_val.shape[0]
ntestfiles = conc_test.shape[0]

#print("  Data Min/Max (","{:>6.2f}".format(cut_perc*100),"% outliers cut):\n")
#for i in dat_minmax.keys():
#    for j in dat_minmax[i].keys():
#        print("    ","{:>10}".format(i),"{:>10}".format(j), ": ", "{:>15.6e}".format(dat_minmax[i][j][0]), " to ", "{:>15.6e}".format(dat_minmax[i][j][1]), "   max/min = ", "{:>15.6e}".format(dat_minmax[i][j][1]/(1E-16+dat_minmax[i][j][0])))
#print("")
#print("  Data Min/Max (","{:>6.2f}".format(cut_perc*100),"% outliers cut):\n")
#for i in dat_minmax.keys():
#    for j in dat_minmax[i].keys():
#        print("    ","{:>10}".format(i),"{:>10}".format(j), ": ", "{:>6.2f}".format(np.log(1E-16+dat_minmax[i][j][0])), " to ", "{:>6.2f}".format(np.log(1E-16+dat_minmax[i][j][1])), "   max/min = ", "{:>6.2f}".format(dat_minmax[i][j][1]/(1E-16+dat_minmax[i][j][0])))
#print("")


# #################################################### #
#                                                      #
#                         NN                           #
#                                                      #
# #################################################### #

from NeuralNetworks import *

from LossFunctions import *


print("  Start NN-Procedure.\n")

nSpc   = spcnames.size
nEmis  = emisnames.size
nMet   = metnames.size
nInput = nSpc + nEmis + nMet

# CREATE MODEL (One STep Integrator) ############################################################

# DONT use one network for single and core of diurnal!
model = Feedforward(nInput,hidden_sizes,nSpc)
#model = ResNet(nSpc, nMet, nEmis, hidden_sizes, n_encoded)
#model = CHININ(D, E, nMet, nEmis, hidden_sizes)

model.to(device1)

core = Feedforward(nInput,hidden_sizes,nSpc)
#core = ResNet(nSpc, nMet, nEmis, hidden_sizes, n_encoded)
#core = CHININ(D, E, nMet, nEmis, hidden_sizes)
model_diurnal = diurnal_model(core)

# LOSS FCN ############################################################
criterion1 = torch.nn.MSELoss()
#criterion1 = MSE_equalizer2
#criterion1 = MSE_focus_o3
#criterion1 = partial(MSE_equalizer, dat_minmax=dat_minmax)

# 2nd LOSS FCN
criterion2 = MSE_equalizer2

# OPTIMIZER ############################################################
optimizer1 = torch.optim.SGD(model.parameters(), lr = learning_rate_s, momentum = momentum_s)
optimizer_diurnal = torch.optim.SGD(model_diurnal.parameters(), lr = learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate_s)
#optimizer_diurnal = torch.optim.Adam(model_diurnal.parameters(), lr = learning_rate)

#2nd OPTIMIZER
optimizer2 = torch.optim.SGD(model.parameters(), lr = learning_rate_2, momentum = momentum_2)
#optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate_s)
optim_switch = loss_switch

# SCHEDULER ############################################################
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_gamma)
scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, factor=learning_gamma_s, patience=patience_s, threshold=threshold_s, verbose=1)
scheduler_diurnal = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_diurnal, factor=learning_gamma, patience=patience, threshold=threshold, verbose=1)

#2nd SCHEDULER
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, factor=learning_gamma_s, patience=patience_s, threshold=threshold_s, verbose=1)

criterion = criterion1
if loss_switch<nepoch:
    print("\n  NOTE: A second loss function will be used from epoch "+str(loss_switch)+" of "+str(nepoch)+"!\n")
optimizer = optimizer1
scheduler = scheduler1
if optim_switch<nepoch:
    print("\n  NOTE: A second optimizer will be used from epoch "+str(optim_switch)+" of "+str(nepoch)+"!\n")

# #################################################### #
#                                                      #
#                   LOSS-TRACKING                      #
#                                                      #
# #################################################### #


# create loss-tracking and final predictions variables
train_loss = np.zeros((nepoch+1))
train_loss_diurnal = np.zeros((nepoch+1))
val_loss = np.zeros((6,nepoch+1))
val_loss_diurnal = np.zeros((6,nepoch+1))
val_loss_epoch = np.zeros((nvalfiles, nTimes-1, 2))
vle_diurnal = np.zeros((nvalfiles, 2))
val_pred = np.zeros((nvalfiles, nTimes, nSpc))
val_pred[:,0,:] = conc_val[:,0,:].cpu()
val_pred_diurnal = np.zeros((nvalfiles, nTimes, nSpc))
val_pred_diurnal[:,0,:] = conc_val[:,0,:].cpu()


# predict!

# turn to evaluation mode
model.eval()
#diurnal_pred = torch.zeros()

# track train loss without affecting gradients
with torch.no_grad():
    for iFile in range(nvalfiles):
        if train_single:
            #for iStep in range(nTimes-1):
            for iStep in range(nTimes-1):
                pred = model(torch.cat((conc_val[iFile,iStep,:], met_val[iFile,iStep,:], emis_val[iFile,:])))
                loss = criterion(pred.squeeze(), conc_val[iFile,iStep+1,:])
                loss_imp = criterion(pred.squeeze()[spcids_imp], conc_val[iFile,iStep+1,spcids_imp])
                #print(iFile, iStep, sum(pred), loss)
                val_loss_epoch[iFile, iStep, :] = [ loss.item(), loss_imp.item() ]
        if train_diurnal:
            pred = model_diurnal(conc_val[iFile,0,:], met_val[iFile,:,:], emis_val[iFile,:])
            loss = criterion(pred.squeeze(), conc_val[iFile,1:,:].flatten())
            loss_imp = torch.tensor(0)
            vle_diurnal[iFile, :] = [ loss.item(), loss_imp.item() ]

    for iFile in range(ntrainfiles):
        if train_single:
            #for iStep in range(nTimes-1):
            for iStep in range(nTimes-1):
                pred = model(torch.cat((conc_train[iFile,iStep,:], met_train[iFile,iStep,:], emis_train[iFile,:])))
                loss = criterion(pred.squeeze(), conc_train[iFile,iStep+1,:])
                train_loss[0] += loss.item()/((nTimes-1)*ntrainfiles)
                #print(iFile, iStep, sum(pred), loss)
        if train_diurnal:
            pred = model_diurnal(conc_train[iFile,0,:], met_train[iFile,:,:], emis_train[iFile,:])
            loss = criterion(pred.squeeze(), conc_train[iFile,1:,:].flatten())
            train_loss_diurnal[0] += loss.item()/((nTimes-1)*ntrainfiles)

vle_packed    = np.sort(val_loss_epoch[:,:,0].flatten())
min_loss      = vle_packed[0]
max_loss      = vle_packed[-1]
minn_loss     = vle_packed[int(outlier_perc * nvalfiles)+1]
maxn_loss     = vle_packed[int((1-outlier_perc) * nvalfiles)]
mean_loss     = np.mean(vle_packed); mean_loss_single=mean_loss
mean_loss_imp = np.mean(val_loss_epoch[:,:,1].flatten())
val_loss[:,0] = [mean_loss, min_loss, max_loss, minn_loss, maxn_loss, mean_loss_imp]
if train_single: print("  Mean loss before single training:  ", mean_loss)

vle_packed    = np.sort(vle_diurnal[:,0].flatten())
min_loss      = vle_packed[0]
max_loss      = vle_packed[-1]
minn_loss     = vle_packed[int(outlier_perc * nvalfiles)+1]
maxn_loss     = vle_packed[int((1-outlier_perc) * nvalfiles)]
mean_loss     = np.mean(vle_packed)
mean_loss_imp = np.mean(vle_diurnal[:,1].flatten())
val_loss_diurnal[:,0] = [mean_loss, min_loss, max_loss, minn_loss, maxn_loss, mean_loss_imp]
if train_diurnal: print("  Mean loss before diurnal training: ", mean_loss,"\n")
mean_loss=mean_loss_single

# check device compatibility
same_device_data = conc_train.device==conc_val.device==conc_test.device==met_train.device==met_val.device==met_test.device==emis_train.device==emis_val.device==emis_test.device
same_device = conc_train.device==next(model.parameters()).device
print("")
if same_device_data and same_device:
    print("  Training and data are on device: "+str(conc_train.device)+".")
elif same_device_data:
    print("  Oops! The model and data are on different devices. Please fix this.")
    sys.exit()
else:
    print("  Please take care of data and model being on one device.")
print("")

# #################################################### #
#                                                      #
#                      TRAINING                        #
#                                                      #
# #################################################### #


timer_arb = 0.0
time_IO = 0.0
time_elapsed=0.0
time_estimated=0.0
time_remaining=0.0


#gs = GridSpec(nrows=6, ncols=nSpc)
#for i in range(6):
#    for j in range(nSpc):
#        plt.subplot(gs[i,j])
#        plt.plot(conc_train[130+i,:,j])
#plt.show()
#gs = GridSpec(nrows=6, ncols=nSpc)
#for i in range(6):
#    for j in range(nSpc):
#        plt.subplot(gs[i,j])
#        plt.plot(conc_val[10+i,:,j])
#plt.show()
#gs = GridSpec(nrows=3, ncols=nSpc)
#for i in range(3):
#    for j in range(nSpc):
#        plt.subplot(gs[i,j])
#        plt.plot(conc_test[i,:,j])
#plt.show()

#with autograd.detect_anomaly():
t0=0.0
t_model=0.0
t_backward=0.0
t_optim=0.0
t_val=0.0
if train_single:

    print("  Training with "+str(ntrainfiles*(nTimes-1))+" data samples.")
    start_Timer = time.perf_counter()
    for epoch in range(1,nepoch+1):
        if epoch == loss_switch+1:
            print("\n  Epoch "+str(epoch)+": Switched Loss Function.")
            criterion = criterion2
        if epoch == optim_switch+1:
            print("\n  Epoch "+str(epoch)+": Switched Optimizer.")
            optimizer = optimizer2
            scheduler = scheduler2
        # set to training mode
        model.train()
        
        # train for every training sample
        for iFile in range(ntrainfiles):
            # print progress update
            perc = (100*(((epoch-1)/nepoch)+(iFile/(nFiles*nepoch))))
            print('  Single Training. Epoch: ', epoch, 'of ', nepoch, '(', "%.2f" % perc, '%)',\
                    ' Mean loss: ',"{:.4e}".format(mean_loss),\
                    '  Time elapsed: ',convertTime(time_elapsed),\
                    ' Estimates: Total: ', convertTime(time_estimated),\
                    ' Remaining: ', convertTime(time_remaining) ,'                   ', end='\r')
            
            ## reset gradients
            #optimizer.zero_grad() 
            for iStep in range(nTimes-1):
                # reset gradients
                optimizer.zero_grad() 

                t0=time.perf_counter()
                pred = model(torch.cat((conc_train[iFile,iStep,:], met_train[iFile,iStep,:], emis_train[iFile,:])))
                t_model+=time.perf_counter()-t0

                t0=time.perf_counter()
                loss = criterion(pred.squeeze(), conc_train[iFile,iStep+1,:])
                loss.backward()      # compute gradients 
                t_backward+=time.perf_counter()-t0
            
                train_loss[epoch] += loss.item()/((nTimes-1)*ntrainfiles)
                #model.float()       # something like more precision in calculations
            
                t0=time.perf_counter()
                optimizer.step()     # do a gradient descend step
                t_optim+=time.perf_counter()-t0
            
                # info prints
                #print("DATA", torch.cat((conc_train[iFile,iStep,:], met_train[iFile,iStep,:], emis_train[iFile,:])))
                #print("TARGET", conc_train[iFile,iStep+1,:])
                #print("PRED", pred.detach().numpy())
                #for param in model.parameters():
                #    print("PARAM ", param, "DATAPARAM ", param.data)
                #print("LOSS", loss.item())
                #print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    
                #print("GRADS", [model.layers[i].weight.grad for i in range(len(hidden_sizes))])

            #optimizer.step()     
            time_elapsed = time.perf_counter() - start_Timer
            time_remaining = max(time_estimated - time_elapsed,0)

        # adjust learning rate
        scheduler.step(val_loss[0,epoch-1])
        #scheduler.step()

        # turn to evaluation mode
        model.eval()

        t0=time.perf_counter()
        # track train loss without affecting gradients
        with torch.no_grad():
            for iFile in range(nvalfiles):
                # print progress update
                perc = (100*(((epoch-1)/nepoch)+((iFile+ntrainfiles)/(nFiles*nepoch))))
                print('  Single Training. Epoch: ', epoch, 'of ', nepoch, '(', "%.2f" % perc, '%)',\
                        ' Mean loss: ',"{:.4e}".format(mean_loss),\
                        '  Time elapsed: ',convertTime(time_elapsed),\
                        ' Estimates: Total: ', convertTime(time_estimated),\
                        ' Remaining: ', convertTime(time_remaining) ,'                   ', end='\r')

                for iStep in range(nTimes-1):
                    pred = model(torch.cat((conc_val[iFile,iStep,:], met_val[iFile,iStep,:], emis_val[iFile,:])))
                    loss = criterion(pred.squeeze(), conc_val[iFile,iStep+1,:])
                    loss_imp = criterion(pred.squeeze()[spcids_imp], conc_val[iFile,iStep+1,spcids_imp])
                    val_loss_epoch[iFile, iStep, :] = [ loss.item(), loss_imp.item() ]
                
                    if epoch==nepoch:
                        val_pred[iFile, iStep+1, :] = pred.cpu().detach().numpy()

                time_elapsed = time.perf_counter() - start_Timer
                time_remaining = max(time_estimated - time_elapsed,0)
        t_val+=time.perf_counter()-t0

        vle_packed = np.sort(val_loss_epoch[:,:,0].flatten())
        min_loss       = vle_packed[0]
        max_loss       = vle_packed[-1]
        minn_loss      = vle_packed[int(outlier_perc * nvalfiles*(nTimes-1))+1]
        maxn_loss      = vle_packed[int((1-outlier_perc) * nvalfiles*(nTimes-1))]
        mean_loss      = np.mean(vle_packed)
        mean_loss_imp  = np.mean(val_loss_epoch[:,:,1].flatten())
        val_loss[:,epoch] = [mean_loss, min_loss, max_loss, minn_loss, maxn_loss, mean_loss_imp]

        time_elapsed = time.perf_counter() - start_Timer
        time_estimated = time_elapsed*nepoch/epoch
        time_remaining = time_estimated - time_elapsed


    t_other = time_elapsed-t_model-t_backward-t_optim-t_val
    mean_loss_red = [val_loss[0,i]/val_loss[0,i-1] for i in range(1,nepoch+1)]

    print('  Single Training. Epoch: ', nepoch, 'of ', nepoch, '(', "%.2f" % (100*nepoch/nepoch), \
      '%)  Time elapsed: ',convertTime(time_elapsed),9*'         ')
    print("  Done.","\n")

    f123 = "{:>40}".format
    f124 = "{:>10}".format
    f123n = "{:.4e}".format
    fperc = "{:.2f}".format
    print("  Metrics after training:")
    print(f123("    Mean loss: "),f123n(val_loss[0,-1]))
    print(f123("    Max loss: "),f123n(val_loss[2,-1]))
    print(f123("    Max loss (without "+str(int(outlier_perc*100))+"% outliers): "),f123n(val_loss[4,-1]))
    print(f123("    Min loss (without "+str(int(outlier_perc*100))+"% outliers): "),f123n(val_loss[3,-1]))
    print(f123("    Min loss: "),f123n(val_loss[1,-1]))
    print("")

    print("  Timers:")
    print(f123("    Total: "), f124(convertTime(time_elapsed)),'(',fperc(100),'%)')
    print(f123("    Model: "), f124(convertTime(t_model)),'(',fperc(100*t_model/time_elapsed),'%)')
    print(f123("    Gradient computation: "), f124(convertTime(t_backward)),'(',fperc(100*t_backward/time_elapsed),'%)')
    print(f123("    Optimizer: "), f124(convertTime(t_optim)),'(',fperc(100*t_optim/time_elapsed),'%)')
    print(f123("    Validation: "), f124(convertTime(t_val)),'(',fperc(100*t_val/time_elapsed),'%)')
    print(f123("    Other: "), f124(convertTime(t_other)),'(',fperc(100*t_other/time_elapsed),'%)')
    print("")

t0=0.0
t_model=0.0
t_backward=0.0
t_optim=0.0
t_val=0.0
if train_diurnal:

    print("  Training with ",str(ntrainfiles) , " samples.")
    start_Timer = time.perf_counter()
    for epoch in range(1,nepoch+1):
        if epoch == loss_switch+1:
            print("\n  Epoch "+str(epoch)+": Switched Loss Function.")
            criterion = criterion2
        if epoch == optim_switch+1:
            print("\n  Epoch "+str(epoch)+": Switched Optimizer.")
            optimizer = optimizer2
            scheduler = scheduler2
        # set to training mode
        model_diurnal.train()
        
        # train for every training sample
        for iFile in range(ntrainfiles):
            # print progress update
            perc = (100*(((epoch-1)/nepoch)+(iFile/(nFiles*nepoch))))
            print('  Diurnal Training. Epoch: ', epoch, 'of ', nepoch, '(', "%.2f" % perc, '%)',\
                    ' Mean loss: ',"{:.4e}".format(mean_loss),\
                    '  Time elapsed: ',convertTime(time_elapsed),\
                    ' Estimates: Total: ', convertTime(time_estimated),\
                    ' Remaining: ', convertTime(time_remaining) ,'                   ', end='\r')
            
            ## reset gradients
            optimizer_diurnal.zero_grad() 

            t0=time.perf_counter()
            pred = model_diurnal(conc_train[iFile,0,:], met_train[iFile,:,:], emis_train[iFile,:])
            t_model+=time.perf_counter()-t0

            t0=time.perf_counter()
            loss = criterion(pred.squeeze(), conc_train[iFile,1:,:].flatten())
            loss.backward()      # compute gradients 
            t_backward+=time.perf_counter()-t0
                
            train_loss_diurnal[epoch] += loss.item()/((nTimes-1)*ntrainfiles)
            #model.float()       # something like more precision in calculations
            
            t0=time.perf_counter()
            optimizer_diurnal.step()     # do a gradient descend step
            t_optim+=time.perf_counter()-t0

            time_elapsed = time.perf_counter() - start_Timer
            time_remaining = max(time_estimated - time_elapsed,0)

        # adjust learning rate
        scheduler_diurnal.step(val_loss_diurnal[0,epoch-1])

        # turn to evaluation mode
        model_diurnal.eval()

        t0=time.perf_counter()
        # track train loss without affecting gradients
        with torch.no_grad():
            for iFile in range(nvalfiles):
                # print progress update
                perc = (100*(((epoch-1)/nepoch)+((iFile+ntrainfiles)/(nFiles*nepoch))))
                print('  Diurnal Training. Epoch: ', epoch, 'of ', nepoch, '(', "%.2f" % perc, '%)',\
                        ' Mean loss: ',"{:.4e}".format(mean_loss),\
                        '  Time elapsed: ',convertTime(time_elapsed),\
                        ' Estimates: Total: ', convertTime(time_estimated),\
                        ' Remaining: ', convertTime(time_remaining) ,'                   ', end='\r')

                pred = model_diurnal(conc_val[iFile,0,:], met_val[iFile,:,:], emis_val[iFile,:])
                loss = criterion(pred.squeeze(), conc_val[iFile,1:,:].flatten())
                loss_imp=torch.tensor(0)
                # THE FOLLOWING IS WRONG, pred.squeeze()[spcids_imp] has to consider all steps in diurnal training
                #loss_imp = criterion(pred.squeeze()[spcids_imp], conc_val[iFile,1:,spcids_imp])
                vle_diurnal[iFile, :] = [ loss.item(), loss_imp.item() ]
                
                if epoch==nepoch:
                    val_pred_diurnal[iFile, 1:, :] = pred.detach().numpy().reshape((nTimes-1, nSpc))

                time_elapsed = time.perf_counter() - start_Timer
                time_remaining = max(time_estimated - time_elapsed,0)

        t_val+=time.perf_counter()-t0

        #vle_packed = np.sort(np.sum(val_loss_epoch, axis=1))
        vle_packed = np.sort(vle_diurnal[:,0].flatten())
        min_loss       = vle_packed[0]
        max_loss       = vle_packed[-1]
        minn_loss      = vle_packed[int(outlier_perc * nvalfiles)+1]
        maxn_loss      = vle_packed[int((1-outlier_perc) * nvalfiles)]
        mean_loss      = np.mean(vle_packed)
        mean_loss_imp  = np.mean(vle_diurnal[:,1].flatten())
        val_loss_diurnal[:,epoch] = [mean_loss, min_loss, max_loss, minn_loss, maxn_loss, mean_loss_imp]

        time_elapsed = time.perf_counter() - start_Timer
        time_estimated = time_elapsed*nepoch/epoch
        time_remaining = time_estimated - time_elapsed

    t_other = time_elapsed-t_model-t_backward-t_optim-t_val
    mean_loss_red_diurnal = [val_loss_diurnal[0,i]/val_loss_diurnal[0,i-1] for i in range(1,nepoch+1)]

    print('  Diurnal Training. Epoch: ', nepoch, 'of ', nepoch, '(', "%.2f" % (100*nepoch/nepoch), \
          '%)  Time elapsed: ',convertTime(time_elapsed),9*'         ')
    print("  Done.","\n")

    f123 = "{:>40}".format
    f124 = "{:>10}".format
    f123n = "{:.4e}".format
    fperc = "{:.2f}".format
    print("  Metrics after training:")
    print(f123("    Mean loss: "),f123n(val_loss_diurnal[0,-1]))
    print(f123("    Max loss: "),f123n(val_loss_diurnal[2,-1]))
    print(f123("    Max loss (without "+str(int(outlier_perc*100))+"% outliers): "),f123n(val_loss_diurnal[4,-1]))
    print(f123("    Min loss (without "+str(int(outlier_perc*100))+"% outliers): "),f123n(val_loss_diurnal[3,-1]))
    print(f123("    Min loss: "),f123n(val_loss_diurnal[1,-1]))
    print("")
    #print("  Mean loss improvements: ", ,"\n")

    print("  Timers:")
    print(f123("    Total: "), f124(convertTime(time_elapsed)),'(',fperc(100),'%)')
    print(f123("    Model: "), f124(convertTime(t_model)),'(',fperc(100*t_model/time_elapsed),'%)')
    print(f123("    Gradient computation: "), f124(convertTime(t_backward)),'(',fperc(100*t_backward/time_elapsed),'%)')
    print(f123("    Optimizer: "), f124(convertTime(t_optim)),'(',fperc(100*t_optim/time_elapsed),'%)')
    print(f123("    Validation: "), f124(convertTime(t_val)),'(',fperc(100*t_val/time_elapsed),'%)')
    print(f123("    Other: "), f124(convertTime(t_other)),'(',fperc(100*t_other/time_elapsed),'%)')
    print("")

#print("  Maxima of test solution: ", np.amax(testsoln,axis=1))


# predict!
model.eval()
test_hourly  = torch.zeros((ntestfiles, timepoints.size, nSpc), device=device1)
test_full    = torch.zeros((ntestfiles, timepoints.size, nSpc), device=device1)
test_diurnal = torch.zeros((ntestfiles, timepoints.size, nSpc), device=device1)
test_hourly[:,0,:] = conc_test[:, 0, :]
test_full[:,0,:] = conc_test[:, 0, :]
test_diurnal[:,0,:] = conc_test[:, 0, :]
for iFile in range(ntestfiles):
    if train_single:
        for iStep in range(timepoints.size-1):
            pred_h = model(torch.cat((conc_test[iFile,iStep,:], met_test[iFile,iStep,:], emis_test[iFile,:])))
            pred_f = model(torch.cat((test_full[iFile,iStep,:], met_test[iFile,iStep,:], emis_test[iFile,:])))
            test_hourly[iFile, iStep+1, :] = pred_h
            test_full[iFile, iStep+1, :]   = pred_f
    if train_diurnal:
        pred_diurnal = model_diurnal(conc_test[iFile,0,:], met_test[iFile,:,:], emis_test[iFile,:])
        test_diurnal[iFile, 1:, :] = pred_diurnal.reshape(24,nSpc)

test_hourly  = test_hourly.cpu().detach().numpy()
test_full    = test_full.cpu().detach().numpy()
test_diurnal = test_diurnal.cpu().detach().numpy()

# create de-scaled conc data for objective comparison (used for R^2-values)
true_conc_val = copy.deepcopy(conc_val.cpu())
true_val_pred = copy.deepcopy(val_pred)
for (iSpc, spc) in enumerate(spcnames):
    true_conc_val[:,:,iSpc] = F_denormal(true_conc_val[:,:,iSpc],dat_minmax["conc"][spc][0],dat_minmax["conc"][spc][1])
    true_val_pred[:,:,iSpc] = F_denormal(true_val_pred[:,:,iSpc],dat_minmax["conc"][spc][0],dat_minmax["conc"][spc][1])

# #################################################### #
#                                                      #
#                      PLOTTING                        #
#                                                      #
# #################################################### #

# calculate predicted errors for all species
rel_eps = 0.00001
abs_eps = 1
err_percs = np.array([1, .99, .95, .9, .8])
err_colors = ["gainsboro","silver","#0066FF","gray","dimgray"]
val_err_dat_abs=np.zeros((err_percs.size, timepoints.size-1, nSpc))
val_err_dat_rel=np.zeros((err_percs.size, timepoints.size-1, nSpc))
#val_err_eps=np.amin(abs(true_val_pred-true_conc_val.cpu().numpy()))*0.001
val_err_eps = abs(true_val_pred-true_conc_val.numpy())
val_err_eps = val_err_eps[val_err_eps>0]
val_err_eps = np.amin(val_err_eps)*0.001
err_dat_rel = np.zeros(nvalfiles)


for iStep in range(timepoints.size-1):
    for iSpc in range(nSpc):
        #max, 99%, 95%, 90%, 80%
        clean_a = true_val_pred[:,iStep,iSpc]
        clean_b = true_conc_val[:,iStep,iSpc].numpy()

        clean_a[np.abs(clean_a) < abs_eps] = 0
        clean_b[np.abs(clean_b) < abs_eps] = 0

        #err_dat=abs(clean_a-clean_b)/(clean_b+val_err_eps)
        err_dat_abs = abs(clean_a-clean_b)
        for (i,err_val) in enumerate(err_dat_abs):
            if clean_b[i]<50 and err_val<25:
                err_dat_rel[i] = 0.0
            elif clean_b[i]<50:
                err_dat_rel[i] = -.05
            else:
                err_dat_rel[i] = 100 * err_val / (abs(clean_b[i])+val_err_eps)

        err_dat_abs = np.sort(err_dat_abs)
        err_dat_rel = np.sort(err_dat_rel)
        val_err_dat_abs[:,iStep,iSpc]=[ err_dat_abs[int(err_percs[j]*nvalfiles)-1] for j in range(err_percs.size) ]
        val_err_dat_rel[:,iStep,iSpc]=[ err_dat_rel[int(err_percs[j]*nvalfiles)-1] for j in range(err_percs.size) ]



nspc_plot = spcnames_plot.size

conc_train=conc_train.cpu(); conc_val=conc_val.cpu(); conc_test=conc_test.cpu();

plot_logscale = False
#plt.figure(figsize=(22,18), dpi=80)
plt.figure(figsize=(24,22), dpi=80)
plt.suptitle('Statistics of validation/test data.\n Mean loss after training: '+str(mean_loss))
ntestplot = min(5, ntestfiles)
gs = GridSpec(nrows=4+ntestplot, ncols=nspc_plot)
ii=0
for i in range(nSpc):
    if spcnames[i] in spcnames_plot:
        for iFile in range(ntestplot):
            plt.subplot(gs[iFile,ii])
            plt.plot(timepoints/3600, conc_test[iFile,:,i], 'k', label='AtCSol')
            if train_single:
                plt.plot(timepoints/3600, test_hourly[iFile,:,i], 'b', label='NN hourly')
                plt.plot(timepoints/3600, test_full[iFile,:,i], 'g', label='NN full')
            if train_diurnal:    
                plt.plot(timepoints/3600, test_diurnal[iFile,:,i], 'c', label='NN diurnal')
            plt.xticks([])
            if iFile==0:
                plt.title(spcnames[i])
            if (ii==0 and iFile==0):
                plt.ylabel('exemplary trajectories (test)')
            if (ii==spcnames_plot.size-1 and iFile==0):
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.subplot(gs[-4,ii])
        plt.scatter(true_conc_val[:,:,i].flatten(), true_val_pred[:,:,i].flatten(), marker=',',c='#0066FF',s=7)
        plt.plot([true_conc_val[:,:,i].min(),true_conc_val[:,:,i].max()],[true_conc_val[:,:,i].min(),true_conc_val[:,:,i].max()], 'k')
        plt.title('R^2='+"{:.4f}".format(rsquared(true_conc_val[:,:,i].flatten(), true_val_pred[:,:,i].flatten())), fontsize=10)
        if ii==0:
            plt.ylabel('predicted values\noverall R^2:'+"{:.9f}".format(rsquared(true_conc_val[:,:,:].flatten(), true_val_pred[:,:,:].flatten())))
            plt.xlabel('target')

        plt.subplot(gs[-3,ii])
        for j in range(err_percs.size):
            plt.fill_between(timepoints[1:]/3600, np.zeros(timepoints.size-1), val_err_dat_rel[j,:,i],\
                             facecolor=err_colors[j], label=str(err_percs[j]*100)+'% of errors')
        
        if plot_logscale:
            plt.xscale('log')
        if (ii==0):
            plt.ylabel('rel. conc err (val) [%]')
        if (ii==spcnames_plot.size-1):
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ii+=1


#plt.subplot(gs[1,:int(nspc_plot/2)])
if train_single and train_diurnal:
    plt.subplot(gs[-2,:int(nspc_plot/2)])
else:
    plt.subplot(gs[-2,:])
if train_single:
    plt.title("Single Step")
    plt.plot(val_loss[2,1:],'grey', label='max loss')
    plt.plot(val_loss[0,1:],'k', label='mean loss')
    plt.plot(val_loss[5,1:],'k--', label='mean imp loss')
    plt.plot(train_loss[1:],'b', label='mean train loss')
    plt.plot(val_loss[1,1:],'grey', label='min loss')
    plt.fill_between(range(nepoch), val_loss[3,1:], val_loss[4,1:], facecolor='grey', alpha=0.5, label=str((1-2*outlier_perc)*100)+'% of losses')
    plt.ylabel('log(loss)')
    plt.yscale('log')
    plt.ylabel('loss evolution (val)')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='lightgrey', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgrey', linestyle='--')
    plt.xticks([])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
if train_single and train_diurnal:
    plt.subplot(gs[-2,int(nspc_plot/2):2*int(nspc_plot/2)])
if train_diurnal:
    plt.title("Diurnal")
    plt.plot(val_loss_diurnal[2,1:],'grey', label='max loss')
    plt.plot(val_loss_diurnal[0,1:],'k', label='mean loss')
    plt.plot(val_loss_diurnal[5,1:],'k--', label='mean imp loss')
    plt.plot(train_loss_diurnal[1:],'b', label='mean train loss')
    plt.plot(val_loss_diurnal[1,1:],'grey', label='min loss')
    plt.fill_between(range(nepoch), val_loss_diurnal[3,1:], val_loss_diurnal[4,1:], facecolor='grey', alpha=0.5, label=str((1-2*outlier_perc)*100)+'% of losses')
    plt.ylabel('log(loss)')
    plt.yscale('log')
    plt.ylabel('loss evolution (val)')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='lightgrey', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgrey', linestyle='--')
    plt.xticks([])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#plt.subplot(gs[1,int(nspc_plot/2):])
if train_single and train_diurnal:
    plt.subplot(gs[-1,:int(nspc_plot/2)])
else:
    plt.subplot(gs[-1,:])
if train_single:
    mean_s = np.sort(mean_loss_red)
    plt.plot(mean_loss_red, 'k', label='mean loss reductions')
    plt.xlabel('epoch')
    plt.ylabel('mean loss\n reduction (val)')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='lightgrey', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgrey', linestyle='--')
    plt.ylim([mean_s[int(0.05*mean_s.size)],1.0])
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
if train_single and train_diurnal:
    plt.subplot(gs[-1,int(nspc_plot/2):2*int(nspc_plot/2)])
if train_diurnal:
    mean_s = np.sort(mean_loss_red_diurnal)
    plt.plot(mean_loss_red_diurnal, 'k', label='mean loss reductions')
    plt.xlabel('epoch')
    plt.ylabel('mean loss\n reduction (val)')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='lightgrey', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgrey', linestyle='--')
    plt.ylim([mean_s[int(0.05*mean_s.size)],1.0])
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


modelpath = 'Figures/Spam/OSTI_'+BSP+\
        '_files'+str(nFiles)+\
        "_conc"+str(spcnames_plot).replace("'","").replace(" ","-")+\
        "_emis"+str(emisnames).replace("'","").replace(" ","-")+\
        "_met"+str(metnames).replace("'","").replace(" ","-")+\
        '_epoch'+str(nepoch)+\
        '_hs'+str(hidden_sizes).replace(" ","")+\
        '_lr'+str(learning_rate)+\
        '_g'+str(learning_gamma)+\
        '_pat'+str(patience)+\
        '_thr'+str(threshold)+\
        '_scaling-'+Scaling+\
        '_'+convertTime(time_elapsed)

# save model and figure
plt.savefig(modelpath+'.png')
torch.save(model, modelpath+'.pth')

if Plotting:
    plt.show()
