# collects all saved results from experiments on given datasets
#
# USAGE:
#   python plot_results.py dataset1 dataset2 ...
root_plot = 'plotdata/'
root_pic_power = 'pic_folder/power/'
root_pic_gas = 'pic_folder/gas/'
root_pic_hepmass = 'pic_folder/hepmass/'
root_pic_miniboone = 'pic_folder/miniboone/'

import sys
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt



def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """

    if not os.path.exists(folder):
        os.makedirs(folder)



def plot_results_power(data_name):
    make_folder(root_pic_power)
    
    #-------------------- load data
    with np.load(root_plot+data_name+'.npz') as powerdata:
        # stuff for mean, std, kl-divergence
        xdim=powerdata['xdim'] 
        mean_results=powerdata['mean_results']
        std_results=powerdata['std_results']
        kl_results=powerdata['kl_results']
        # stuff for 1d marginal pdf (here we consider dim=1,4)
        kltruth_1d_0=powerdata['kltruth_1d_0']
        kltruth_1d_3=powerdata['kltruth_1d_3']
        kllabel_1d_0=powerdata['kllabel_1d_0']
        kllabel_1d_3=powerdata['kllabel_1d_3']
        klpred_1d_0=powerdata['klpred_1d_0']
        klpred_1d_3=powerdata['klpred_1d_3']
        # stuff for 2d marginal pdf (here we consider dims=2,5)
        xx=powerdata['xx']
        yy=powerdata['yy']
        f_truth=powerdata['kltruth_2d']
        f_label=powerdata['kllabel_2d']
        f_pred=powerdata['klpred_2d'] 

    #-------------------- plotting
    # 1d marginal truth
    plt.figure()    # d = 1
    plt.cla()
    plt.fill_between(kltruth_1d_0[0,:],kltruth_1d_0[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-1,4])
    plt.savefig(root_pic_power + data_name + 'd1_truth.png', bbox_inches="tight", dpi=300 )
    #
    plt.figure()    # d = 4
    plt.cla()
    plt.fill_between(kltruth_1d_3[0,:],kltruth_1d_3[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-0.5,1])
    plt.savefig(root_pic_power + data_name + 'd4_truth.png', bbox_inches="tight", dpi=300 )

    # 1d marginal label
    plt.figure()    # d = 1
    plt.cla()
    plt.fill_between(kllabel_1d_0[0,:],kllabel_1d_0[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-1,4])
    plt.savefig(root_pic_power + data_name + 'd1_label.png', bbox_inches="tight", dpi=300 )
    #
    plt.figure()    # d = 4
    plt.cla()
    plt.fill_between(kllabel_1d_3[0,:],kllabel_1d_3[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-0.5,1])
    plt.savefig(root_pic_power + data_name + 'd4_label.png', bbox_inches="tight", dpi=300 )

    # 1d marginal pred
    plt.figure()    # d = 1
    plt.cla()
    plt.fill_between(klpred_1d_0[0,:],klpred_1d_0[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-1,4])
    plt.savefig(root_pic_power + data_name + 'd1_pred.png', bbox_inches="tight", dpi=300 )
    #
    plt.figure()    # d = 4
    plt.cla()
    plt.fill_between(klpred_1d_3[0,:],klpred_1d_3[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-0.5,1])
    plt.savefig(root_pic_power + data_name + 'd4_pred.png', bbox_inches="tight", dpi=300 )


    # 2d KDE
    # truth: d=2,5
    plt.figure()
    plt.cla()
    plt.contourf(xx, yy, np.log10(f_truth+1e-10), cmap='coolwarm')
    plt.xlim( -6, 5)
    plt.ylim(-1,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_pic_power + '2dtruth_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # label: d=2,5
    plt.figure()
    plt.cla()
    plt.contourf(xx, yy, np.log10(f_label+1e-10), cmap='coolwarm')
    plt.xlim( -6, 5)
    plt.ylim(-1,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_pic_power + '2dlabel_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # pred: d=2,5
    plt.figure()
    plt.cla()
    plt.contourf(xx, yy, np.log10(f_pred+1e-10), cmap='coolwarm')
    plt.xlim( -6, 5)
    plt.ylim(-1,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_pic_power + '2dpred_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot KL
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,kl_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,kl_results[1,:], color="b")
    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylabel('KL divergence',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Labeled', 'Generated'],fontsize = 12, loc = 'upper left')
    plt.savefig(root_pic_power + 'KL_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot mean
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,mean_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,mean_results[1,:], color="b")
    plt.plot(np.arange(xdim, dtype=int)+1,mean_results[2,:], color="k")

    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylim(-0.1,0.1)
    plt.ylabel('mean',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Truth','Labeled', 'Generated'],fontsize = 12, loc = 'upper right')
    plt.savefig(root_pic_power + 'mean_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot std
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[1,:], color="b")
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[2,:], color="k")
    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylim(0.9,1.1)
    plt.ylabel('std',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Truth','Labeled', 'Generated'],fontsize = 12, loc = 'upper right')
    plt.savefig(root_pic_power + 'std_' + data_name + '.png', bbox_inches="tight", dpi=300 )


def plot_results_gas(data_name):
    make_folder(root_pic_gas)

    #-------------------- load data
    with np.load(root_plot+data_name+'.npz') as gasdata:
        # stuff for mean, std, kl-divergence
        xdim=gasdata['xdim'] 
        mean_results=gasdata['mean_results']
        std_results=gasdata['std_results']
        kl_results=gasdata['kl_results']
        # stuff for 1d marginal pdf (here we consider dim=1,4)
        kltruth_1d_0=gasdata['kltruth_1d_0']
        kltruth_1d_2=gasdata['kltruth_1d_2']
        kllabel_1d_0=gasdata['kllabel_1d_0']
        kllabel_1d_2=gasdata['kllabel_1d_2']
        klpred_1d_0=gasdata['klpred_1d_0']
        klpred_1d_2=gasdata['klpred_1d_2']
        # stuff for 2d marginal pdf (here we consider dims=2,5)
        xx=gasdata['xx']
        yy=gasdata['yy']
        f_truth=gasdata['kltruth_2d']
        f_label=gasdata['kllabel_2d']
        f_pred=gasdata['klpred_2d'] 



    #-------------------- plotting
    # 1d marginal truth
    plt.figure()    # d=1
    plt.cla()
    plt.fill_between(kltruth_1d_0[0,:],kltruth_1d_0[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-2,2])
    plt.savefig(root_pic_gas + data_name + 'd1_truth.png', bbox_inches="tight", dpi=300 )
    #
    plt.figure()    # d=3
    plt.cla()
    plt.fill_between(kltruth_1d_2[0,:],kltruth_1d_2[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-2.5,2.5])
    plt.savefig(root_pic_gas + data_name + 'd3_truth.png', bbox_inches="tight", dpi=300 )

    # 1d marginal label
    plt.figure()    # d=1
    plt.cla()
    plt.fill_between(kllabel_1d_0[0,:],kllabel_1d_0[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-2,2])
    plt.savefig(root_pic_gas + data_name + 'd1_label.png', bbox_inches="tight", dpi=300 )
    #
    plt.figure()    # d=3
    plt.cla()
    plt.fill_between(kllabel_1d_2[0,:],kllabel_1d_2[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-2.5,2.5])
    plt.savefig(root_pic_gas + data_name + 'd3_label.png', bbox_inches="tight", dpi=300 )

    # 1d marginal pred
    plt.figure()    # d=1
    plt.cla()
    plt.fill_between(klpred_1d_0[0,:],klpred_1d_0[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-2,2])
    plt.savefig(root_pic_gas + data_name + 'd1_pred.png', bbox_inches="tight", dpi=300 )
    #
    plt.figure()    # d=3
    plt.cla()
    plt.fill_between(klpred_1d_2[0,:],klpred_1d_2[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-2.5,2.5])
    plt.savefig(root_pic_gas + data_name + 'd3_pred.png', bbox_inches="tight", dpi=300 )




    # 2d KDE
    # truth: # d=2,4
    plt.figure()
    plt.cla()
    plt.contourf(xx, yy, np.log10(f_truth+1e-10), cmap='coolwarm')
    plt.xlim( -1, 5)
    plt.ylim(-3,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_pic_gas + '2dtruth_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # label: # d=2,4
    plt.figure()
    plt.cla()
    plt.contourf(xx, yy, np.log10(f_label+1e-10), cmap='coolwarm')
    plt.xlim( -1, 5)
    plt.ylim(-3,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_pic_gas + '2dlabel_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # pred: # d=2,4
    plt.figure()
    plt.cla()
    plt.contourf(xx, yy, np.log10(f_pred+1e-10), cmap='coolwarm')
    plt.xlim( -1, 5)
    plt.ylim(-3,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_pic_gas + '2dpred_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot KL
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,kl_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,kl_results[1,:], color="b")
    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylabel('KL divergence',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Labeled', 'Generated'],fontsize = 12, loc = 'upper right')
    plt.savefig(root_pic_gas + 'KL_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot mean
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,mean_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,mean_results[1,:], color="b")
    plt.plot(np.arange(xdim, dtype=int)+1,mean_results[2,:], color="k")

    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylim(-0.1,0.1)
    plt.ylabel('mean',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Truth','Labeled', 'Generated'],fontsize = 12, loc = 'upper right')
    plt.savefig(root_pic_gas + 'mean_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot std
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[1,:], color="b")
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[2,:], color="k")
    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylim(0.9,1.1)
    plt.ylabel('std',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Truth','Labeled', 'Generated'],fontsize = 12, loc = 'upper right')
    plt.savefig(root_pic_gas + 'std_' + data_name + '.png', bbox_inches="tight", dpi=300 )


def plot_results_hepmass(data_name):
    make_folder(root_pic_hepmass)
    
    #-------------------- load data
    with np.load(root_plot+data_name+'.npz') as hepmassdata:
        # stuff for mean, std, kl-divergence
        xdim=hepmassdata['xdim'] 
        mean_results=hepmassdata['mean_results']
        std_results=hepmassdata['std_results']
        kl_results=hepmassdata['kl_results']
        # stuff for 1d marginal pdf (here we consider dim=1,4)
        kltruth_1d_8=hepmassdata['kltruth_1d_8']
        kltruth_1d_9=hepmassdata['kltruth_1d_9']
        kllabel_1d_8=hepmassdata['kllabel_1d_8']
        kllabel_1d_9=hepmassdata['kllabel_1d_9']
        klpred_1d_8=hepmassdata['klpred_1d_8']
        klpred_1d_9=hepmassdata['klpred_1d_9']
        # stuff for 2d marginal pdf (here we consider dims=2,5)
        xx=hepmassdata['xx']
        yy=hepmassdata['yy']
        f_truth=hepmassdata['kltruth_2d']
        f_label=hepmassdata['kllabel_2d']
        f_pred=hepmassdata['klpred_2d'] 

    #-------------------- plotting
    # 1d marginal truth
    plt.figure()    # d=9
    plt.cla()
    plt.fill_between(kltruth_1d_8[0,:],kltruth_1d_8[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-1,4])
    plt.savefig(root_pic_hepmass + data_name + 'd9_truth.png', bbox_inches="tight", dpi=300 )
    #
    plt.figure()     # d=10
    plt.cla()
    plt.fill_between(kltruth_1d_9[0,:],kltruth_1d_9[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-3,4])
    plt.savefig(root_pic_hepmass + data_name + 'd10_truth.png', bbox_inches="tight", dpi=300 )

    # 1d marginal label
    plt.figure()     # d=1
    plt.cla()
    plt.fill_between(kllabel_1d_8[0,:],kllabel_1d_8[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-1,4])
    plt.savefig(root_pic_hepmass + data_name + 'd9_label.png', bbox_inches="tight", dpi=300 )
    #
    plt.figure()     # d=10
    plt.cla()
    plt.fill_between(kllabel_1d_9[0,:],kllabel_1d_9[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-3,4])
    plt.savefig(root_pic_hepmass + data_name + 'd10_label.png', bbox_inches="tight", dpi=300 )

    # 1d marginal pred
    plt.figure()     # d=1
    plt.cla()
    plt.fill_between(klpred_1d_8[0,:],klpred_1d_8[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-1,4])
    plt.savefig(root_pic_hepmass + data_name + 'd9_pred.png', bbox_inches="tight", dpi=300 )
    #
    plt.figure()     # d=10
    plt.cla()
    plt.fill_between(klpred_1d_9[0,:],klpred_1d_9[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-3,4])
    plt.savefig(root_pic_hepmass + data_name + 'd10_pred.png', bbox_inches="tight", dpi=300 )


    # 2d KDE
    # truth: d=2,5
    plt.figure()
    plt.cla()
    plt.contourf(xx, yy, np.log10(f_truth+1e-10), cmap='coolwarm')
    plt.xlim( -6, 3)
    plt.ylim(-3,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_pic_hepmass + '2dtruth_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # label: d=2,5
    plt.figure()
    plt.cla()
    plt.contourf(xx, yy, np.log10(f_label+1e-10), cmap='coolwarm')
    plt.xlim( -6, 3)
    plt.ylim(-3,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_pic_hepmass + '2dlabel_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # pred: d=2,5
    plt.figure()
    plt.cla()
    plt.contourf(xx, yy, np.log10(f_pred+1e-10), cmap='coolwarm')
    plt.xlim( -6, 3)
    plt.ylim(-3,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_pic_hepmass + '2dpred_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot KL
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,kl_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,kl_results[1,:], color="b")
    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylabel('KL divergence',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Labeled', 'Generated'],fontsize = 12, loc = 'upper left')
    plt.savefig(root_pic_hepmass + 'KL_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot mean
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,mean_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,mean_results[1,:], color="b")
    plt.plot(np.arange(xdim, dtype=int)+1,mean_results[2,:], color="k")

    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylim(-0.1,0.1)
    plt.ylabel('mean',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Truth','Labeled', 'Generated'],fontsize = 12, loc = 'upper right')
    plt.savefig(root_pic_hepmass + 'mean_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot std
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[1,:], color="b")
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[2,:], color="k")
    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylim(0.9,1.1)
    plt.ylabel('std',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Truth','Labeled', 'Generated'],fontsize = 12, loc = 'upper right')
    plt.savefig(root_pic_hepmass + 'std_' + data_name + '.png', bbox_inches="tight", dpi=300 )



def plot_results_miniboone(data_name):
    make_folder(root_pic_miniboone)
    
    #-------------------- load data
    with np.load(root_plot+data_name+'.npz') as miniboonedata:
        # stuff for mean, std, kl-divergence
        xdim=miniboonedata['xdim'] 
        mean_results=miniboonedata['mean_results']
        std_results=miniboonedata['std_results']
        kl_results=miniboonedata['kl_results']
        # stuff for 1d marginal pdf (here we consider dim=1,4)
        kltruth_1d_0=miniboonedata['kltruth_1d_0']
        kltruth_1d_15=miniboonedata['kltruth_1d_15']
        kllabel_1d_0=miniboonedata['kllabel_1d_0']
        kllabel_1d_15=miniboonedata['kllabel_1d_15']
        klpred_1d_0=miniboonedata['klpred_1d_0']
        klpred_1d_15=miniboonedata['klpred_1d_15']
        # stuff for 2d marginal pdf (here we consider dims=2,5)
        xx=miniboonedata['xx']
        yy=miniboonedata['yy']
        f_truth=miniboonedata['kltruth_2d']
        f_label=miniboonedata['kllabel_2d']
        f_pred=miniboonedata['klpred_2d'] 

    #-------------------- plotting
    # 1d marginal truth
    plt.figure()    # d = 1
    plt.cla()
    plt.fill_between(kltruth_1d_0[0,:],kltruth_1d_0[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-3,4])
    plt.savefig(root_pic_miniboone + data_name + 'd1_truth.png', bbox_inches="tight", dpi=300 )
    #
    plt.figure()    # d = 4
    plt.cla()
    plt.fill_between(kltruth_1d_15[0,:],kltruth_1d_15[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-1,4])
    plt.savefig(root_pic_miniboone + data_name + 'd16_truth.png', bbox_inches="tight", dpi=300 )

    # 1d marginal label
    plt.figure()    # d = 1
    plt.cla()
    plt.fill_between(kllabel_1d_0[0,:],kllabel_1d_0[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-3,4])
    plt.savefig(root_pic_miniboone + data_name + 'd1_label.png', bbox_inches="tight", dpi=300 )
    #
    plt.figure()    # d = 4
    plt.cla()
    plt.fill_between(kllabel_1d_15[0,:],kllabel_1d_15[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-1,4])
    plt.savefig(root_pic_miniboone + data_name + 'd16_label.png', bbox_inches="tight", dpi=300 )

    # 1d marginal pred
    plt.figure()    # d = 1
    plt.cla()
    plt.fill_between(klpred_1d_0[0,:],klpred_1d_0[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-3,4])
    plt.savefig(root_pic_miniboone + data_name + 'd1_pred.png', bbox_inches="tight", dpi=300 )
    #
    plt.figure()    # d = 4
    plt.cla()
    plt.fill_between(klpred_1d_15[0,:],klpred_1d_15[1,:],alpha=0.4)
    plt.xticks(fontsize=14)
    plt.tick_params(left = False, labelleft = False)
    plt.xlim([-1,4])
    plt.savefig(root_pic_miniboone + data_name + 'd16_pred.png', bbox_inches="tight", dpi=300 )


    # 2d KDE
    # truth: d=2,5
    plt.figure()
    plt.cla()
    plt.contourf(xx, yy, np.log10(f_truth+1e-10), cmap='coolwarm')
    plt.xlim( -4, 6)
    plt.ylim(-4,4)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_pic_miniboone + '2dtruth_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # label: d=2,5
    plt.figure()
    plt.cla()
    plt.contourf(xx, yy, np.log10(f_label+1e-10), cmap='coolwarm')
    plt.xlim( -4, 6)
    plt.ylim(-4,4)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_pic_miniboone + '2dlabel_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # pred: d=2,5
    plt.figure()
    plt.cla()
    plt.contourf(xx, yy, np.log10(f_pred+1e-10), cmap='coolwarm')
    plt.xlim( -4, 6)
    plt.ylim(-4,4)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_pic_miniboone + '2dpred_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot KL
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,kl_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,kl_results[1,:], color="b")
    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylabel('KL divergence',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Labeled', 'Generated'],fontsize = 12, loc = 'upper left')
    plt.savefig(root_pic_miniboone + 'KL_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot mean
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,mean_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,mean_results[1,:], color="b")
    plt.plot(np.arange(xdim, dtype=int)+1,mean_results[2,:], color="k")

    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylim(-0.1,0.1)
    plt.ylabel('mean',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Truth','Labeled', 'Generated'],fontsize = 12, loc = 'upper right')
    plt.savefig(root_pic_miniboone + 'mean_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot std
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[1,:], color="b")
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[2,:], color="k")
    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylim(0.9,1.2)
    plt.ylabel('std',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Truth','Labeled', 'Generated'],fontsize = 12, loc = 'upper right')
    plt.savefig(root_pic_miniboone + 'std_' + data_name + '.png', bbox_inches="tight", dpi=300 )


def main():

    for data in sys.argv[1:]:

        if data == 'power':
            plot_results_power(data)

        elif data == 'gas':
            plot_results_gas(data)

        elif data == 'hepmass':
            plot_results_hepmass(data)

        elif data == 'miniboone':
            plot_results_miniboone(data)

        elif data == 'bsds300':
            plot_results_bsds300(data)

        else:
            print('{0} is not a valid dataset'.format(data))
            continue


if __name__ == '__main__':
    main()
