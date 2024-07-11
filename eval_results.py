# collects all saved results from experiments on given datasets
#
# USAGE:
#   python collect_results.py dataset1 dataset2 ...
root_orbit = 'orbit/'      # where to save orbits
root_output = 'output/'   # where to save trained models
root_plot = 'plotdata/'
import sys
import numpy as np
import learning as ln
import os
from scipy import stats
import matplotlib.pyplot as plt
root_results = 'results/'  # folder where to save results


def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """

    if not os.path.exists(folder):
        os.makedirs(folder)



def collect_results_power(data_name):
    make_folder(root_plot)
    make_folder(root_output)
    with open(root_orbit + data_name+'.npy', 'rb') as f:
        X_truth = np.load(f)
        X_T = np.load(f)
        X_label = np.load(f)
    with open(root_orbit + data_name+'_nn.npy', 'rb') as f:
        X_pred = np.load(f)

    xdim = np.shape(X_pred)[1]
    print(xdim)
    kl_results = np.zeros((2,xdim))
    mean_results = np.zeros((3,xdim))
    mean_results[0,:] = np.mean(X_truth,axis=0) 
    mean_results[1,:] = np.mean(X_label,axis=0)
    mean_results[2,:] = np.mean(X_pred,axis=0)
    std_results = np.zeros((3,xdim))
    std_results[0,:] = np.std(X_truth,axis=0) 
    std_results[1,:] = np.std(X_label,axis=0)
    std_results[2,:] = np.std(X_pred,axis=0)
    # marginal KL
    for ii in range(xdim):
    # for ii in range(1):
        print(ii)
        # for kl_div
        xl,xr = np.min(X_truth[:,ii]),np.max(X_truth[:,ii])
        x_kl = np.linspace(xl,xr, 1000)
        kernel_truth = stats.gaussian_kde(X_truth[:,ii], bw_method=0.2)
        kernel_label = stats.gaussian_kde(X_label[:,ii], bw_method=0.2)
        kernel_pred = stats.gaussian_kde(X_pred[:,ii], bw_method=0.2)
        kl_tl = np.sum( kernel_truth(x_kl)*(x_kl[1]-x_kl[0]) * np.log(  (kernel_truth(x_kl) + 1e-16) /  ( kernel_label(x_kl) + 1e-16)  ) )
        kl_tp = np.sum( kernel_truth(x_kl)*(x_kl[1]-x_kl[0]) * np.log(  (kernel_truth(x_kl) + 1e-16) /  ( kernel_pred(x_kl) + 1e-16)  ) )
        kl_results[0,ii] = kl_tl
        kl_results[1,ii] = kl_tp


        if ii == 0 or ii == 3:
            # 1d marginal truth
            plt.figure()
            plt.cla()
            plt.fill_between(x_kl,kernel_truth(x_kl),alpha=0.4)
            plt.xticks(fontsize=14)
            plt.tick_params(left = False, labelleft = False)
            if ii == 3:
                plt.xlim([-0.5,1])
                kltruth_1d_3 = np.vstack((x_kl,kernel_truth(x_kl)))
            elif ii == 0:
                kltruth_1d_0 = np.vstack((x_kl,kernel_truth(x_kl)))
                plt.xlim([-1,4])
            plt.savefig(root_output + data_name + str(ii+1) + 'd_truth.png', bbox_inches="tight", dpi=300 )

            # 1d marginal label
            plt.figure()
            plt.cla()
            plt.fill_between(x_kl,kernel_label(x_kl),alpha=0.4)
            if ii == 3:
                plt.xlim([-0.5,1])
                kllabel_1d_3 = np.vstack((x_kl,kernel_label(x_kl)))
            elif ii == 0:
                kllabel_1d_0 = np.vstack((x_kl,kernel_label(x_kl)))
                plt.xlim([-1,4])
            plt.xticks(fontsize=14)
            plt.tick_params(left = False, labelleft = False)
            plt.savefig(root_output + data_name + str(ii+1) + 'd_label.png', bbox_inches="tight", dpi=300 )

            # 1d marginal pred
            plt.figure()
            plt.cla()
            plt.fill_between(x_kl,kernel_pred(x_kl),alpha=0.4)
            # plt.xlabel('d=' + str(ii+1),fontsize=23)
            if ii == 3:
                plt.xlim([-0.5,1])
                klpred_1d_3 = np.vstack((x_kl,kernel_pred(x_kl)))
            elif ii == 0:
                klpred_1d_0 = np.vstack((x_kl,kernel_pred(x_kl)))
                plt.xlim([-1,4])
            plt.xticks(fontsize=14)
            plt.tick_params(left = False, labelleft = False)
            plt.savefig(root_output + data_name + str(ii+1) + 'd_pred.png', bbox_inches="tight", dpi=300 )


    # 2d KDE
    # truth: xi, p
    xi = X_truth[:,1]
    p = X_truth[:,4]
    deltaX = (max(xi) - min(xi))/100
    deltaY = (max(p) - min(p))/100
    xmin = min(xi) - deltaX
    xmax = max(xi) + deltaX
    ymin = min(p) - deltaY
    ymax = max(p) + deltaY
    xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([xi, p])
    plt.figure()
    plt.cla()
    kernel2d_truth = stats.gaussian_kde(values)
    f_truth = np.reshape(kernel2d_truth(positions).T, xx.shape)
    plt.contourf(xx, yy, np.log10(f_truth+1e-10), cmap='coolwarm')
    plt.xlim( -6, 5)
    plt.ylim(-1,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_output + '2dtruth_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # label: xi, p
    xi = X_label[:,1]
    p = X_label[:,4]
    values = np.vstack([xi, p])
    plt.figure()
    plt.cla()
    kernel2d_label = stats.gaussian_kde(values)
    f_label = np.reshape(kernel2d_label(positions).T, xx.shape)
    plt.contourf(xx, yy, np.log10(f_label+1e-10), cmap='coolwarm')
    plt.xlim( -6, 5)
    plt.ylim(-1,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_output + '2dlabel_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # pred: xi, p
    xi = X_pred[:,1]
    p = X_pred[:,4]
    values = np.vstack([xi, p])
    plt.figure()
    plt.cla()
    kernel2d_pred = stats.gaussian_kde(values)
    f_pred = np.reshape(kernel2d_pred(positions).T, xx.shape)
    plt.contourf(xx, yy, np.log10(f_pred+1e-10), cmap='coolwarm')
    plt.xlim( -6, 5)
    plt.ylim(-1,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_output + '2dnn_' + data_name + '.png', bbox_inches="tight", dpi=300 )

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
    plt.savefig(root_output + 'KL_' + data_name + '.png', bbox_inches="tight", dpi=300 )

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
    plt.savefig(root_output + 'mean_' + data_name + '.png', bbox_inches="tight", dpi=300 )

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
    plt.savefig(root_output + 'std_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    np.savez(root_plot+data_name+'.npz',xdim=xdim, mean_results=mean_results, std_results=std_results, kl_results=kl_results, \
             kltruth_1d_0=kltruth_1d_0, kltruth_1d_3=kltruth_1d_3,\
             kllabel_1d_0=kllabel_1d_0, kllabel_1d_3=kllabel_1d_3,\
             klpred_1d_0=klpred_1d_0, klpred_1d_3=klpred_1d_3,\
             xx=xx, yy=yy, kltruth_2d=f_truth, kllabel_2d=f_label, klpred_2d=f_pred )



def collect_results_gas(data_name):
    make_folder(root_plot)
    make_folder(root_output)
    with open(root_orbit + data_name+'.npy', 'rb') as f:
        X_truth = np.load(f)
        X_T = np.load(f)
        X_label = np.load(f)
    with open(root_orbit + data_name+'_nn.npy', 'rb') as f:
        X_pred = np.load(f)

    xdim = np.shape(X_pred)[1]
    print(xdim)
    kl_results = np.zeros((2,xdim))
    mean_results = np.zeros((3,xdim))
    mean_results[0,:] = np.mean(X_truth,axis=0) 
    mean_results[1,:] = np.mean(X_label,axis=0)
    mean_results[2,:] = np.mean(X_pred,axis=0)
    std_results = np.zeros((3,xdim))
    std_results[0,:] = np.std(X_truth,axis=0) 
    std_results[1,:] = np.std(X_label,axis=0)
    std_results[2,:] = np.std(X_pred,axis=0)
    # marginal KL
    for ii in range(xdim):
    # for ii in range(1):
        print(ii)
        # for kl_div
        xl,xr = np.min(X_truth[:,ii]),np.max(X_truth[:,ii])
        x_kl = np.linspace(xl,xr, 1000)
        kernel_truth = stats.gaussian_kde(X_truth[:,ii], bw_method=0.2)
        kernel_label = stats.gaussian_kde(X_label[:,ii], bw_method=0.2)
        kernel_pred = stats.gaussian_kde(X_pred[:,ii], bw_method=0.2)
        kl_tl = np.sum( kernel_truth(x_kl)*(x_kl[1]-x_kl[0]) * np.log(  (kernel_truth(x_kl) + 1e-16) /  ( kernel_label(x_kl) + 1e-16)  ) )
        kl_tp = np.sum( kernel_truth(x_kl)*(x_kl[1]-x_kl[0]) * np.log(  (kernel_truth(x_kl) + 1e-16) /  ( kernel_pred(x_kl) + 1e-16)  ) )
        kl_results[0,ii] = kl_tl
        kl_results[1,ii] = kl_tp

        if ii == 0 or ii == 2:
            # 1d marginal truth
            plt.figure()
            plt.cla()
            plt.fill_between(x_kl,kernel_truth(x_kl),alpha=0.4)
            plt.xticks(fontsize=14)
            plt.tick_params(left = False, labelleft = False)
            if ii == 2:
                plt.xlim([-2.5,2.5])
                kltruth_1d_2 = np.vstack((x_kl,kernel_truth(x_kl)))
            elif ii == 0:
                plt.xlim([-2,2])
                kltruth_1d_0 = np.vstack((x_kl,kernel_truth(x_kl)))
            plt.savefig(root_output + data_name + str(ii+1) + 'd_truth.png', bbox_inches="tight", dpi=300 )

            # 1d marginal label
            plt.figure()
            plt.cla()
            plt.fill_between(x_kl,kernel_label(x_kl),alpha=0.4)
            # plt.xlabel('d=' + str(ii+1),fontsize=23)
            if ii == 2:
                plt.xlim([-2.5,2.5])
                kllabel_1d_2 = np.vstack((x_kl,kernel_label(x_kl)))
            elif ii == 0:
                plt.xlim([-2,2])
                kllabel_1d_0 = np.vstack((x_kl,kernel_label(x_kl)))
            plt.xticks(fontsize=14)
            plt.tick_params(left = False, labelleft = False)
            plt.savefig(root_output + data_name + str(ii+1) + 'd_label.png', bbox_inches="tight", dpi=300 )

            # 1d marginal pred
            plt.figure()
            plt.cla()
            plt.fill_between(x_kl,kernel_pred(x_kl),alpha=0.4)
            # plt.xlabel('d=' + str(ii+1),fontsize=23)
            if ii == 2:
                plt.xlim([-2.5,2.5])
                klpred_1d_2 = np.vstack((x_kl,kernel_pred(x_kl)))
            elif ii == 0:
                plt.xlim([-2,2])
                klpred_1d_0 = np.vstack((x_kl,kernel_pred(x_kl)))
            plt.xticks(fontsize=14)
            plt.tick_params(left = False, labelleft = False)
            plt.savefig(root_output + data_name + str(ii+1) + 'd_pred.png', bbox_inches="tight", dpi=300 )


    # 2d KDE
    # truth: xi, p
    xi = X_truth[:,1]
    p = X_truth[:,3]
    deltaX = (max(xi) - min(xi))/100
    deltaY = (max(p) - min(p))/100
    xmin = min(xi) - deltaX
    xmax = max(xi) + deltaX
    ymin = min(p) - deltaY
    ymax = max(p) + deltaY
    xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([xi, p])
    plt.figure()
    plt.cla()
    kernel2d_truth = stats.gaussian_kde(values)
    f_truth = np.reshape(kernel2d_truth(positions).T, xx.shape)
    plt.contourf(xx, yy, np.log10(f_truth+1e-10), cmap='coolwarm')
    plt.xlim( -1, 5)
    plt.ylim(-3,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_output + '2dtruth_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # label: xi, p
    xi = X_label[:,1]
    p = X_label[:,3]
    values = np.vstack([xi, p])
    plt.figure()
    plt.cla()
    kernel2d_label = stats.gaussian_kde(values)
    f_label = np.reshape(kernel2d_label(positions).T, xx.shape)
    plt.contourf(xx, yy, np.log10(f_label+1e-10), cmap='coolwarm')
    plt.xlim( -1, 5)
    plt.ylim(-3,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_output + '2dlabel_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # pred: xi, p
    xi = X_pred[:,1]
    p = X_pred[:,3]
    values = np.vstack([xi, p])
    plt.figure()
    plt.cla()
    kernel2d_pred = stats.gaussian_kde(values)
    f_pred = np.reshape(kernel2d_pred(positions).T, xx.shape)
    plt.contourf(xx, yy, np.log10(f_pred+1e-10), cmap='coolwarm')
    plt.xlim( -1, 5)
    plt.ylim(-3,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_output + '2dnn_' + data_name + '.png', bbox_inches="tight", dpi=300 )

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
    plt.savefig(root_output + 'KL_' + data_name + '.png', bbox_inches="tight", dpi=300 )

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
    plt.savefig(root_output + 'mean_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot std
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[1,:], color="b")
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[2,:], color="k")

    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylabel('std',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.9,1.1)
    plt.grid()        
    plt.legend(['Truth','Labeled', 'Generated'],fontsize = 12, loc = 'upper right')
    plt.savefig(root_output + 'std_' + data_name + '.png', bbox_inches="tight", dpi=300 )


    np.savez(root_plot+data_name+'.npz',xdim=xdim, mean_results=mean_results, std_results=std_results, kl_results=kl_results, \
             kltruth_1d_0=kltruth_1d_0, kltruth_1d_2=kltruth_1d_2,\
             kllabel_1d_0=kllabel_1d_0, kllabel_1d_2=kllabel_1d_2,\
             klpred_1d_0=klpred_1d_0, klpred_1d_2=klpred_1d_2,\
             xx=xx, yy=yy, kltruth_2d=f_truth, kllabel_2d=f_label, klpred_2d=f_pred )




def collect_results_hepmass(data_name):
    make_folder(root_plot)
    make_folder(root_output)
    with open(root_orbit + data_name+'.npy', 'rb') as f:
        X_truth = np.load(f)
        X_T = np.load(f)
        X_label = np.load(f)
    with open(root_orbit + data_name+'_nn.npy', 'rb') as f:
        X_pred = np.load(f)

    xdim = np.shape(X_pred)[1]
    print(xdim)
    kl_results = np.zeros((2,xdim))
    mean_results = np.zeros((3,xdim))
    mean_results[0,:] = np.mean(X_truth,axis=0) 
    mean_results[1,:] = np.mean(X_label,axis=0)
    mean_results[2,:] = np.mean(X_pred,axis=0)
    std_results = np.zeros((3,xdim))
    std_results[0,:] = np.std(X_truth,axis=0) 
    std_results[1,:] = np.std(X_label,axis=0)
    std_results[2,:] = np.std(X_pred,axis=0)
    # marginal KL
    for ii in range(xdim):
        print(ii)
        # for kl_div
        xl,xr = np.min(X_truth[:,ii]),np.max(X_truth[:,ii])
        x_kl = np.linspace(xl,xr, 1000)
        kernel_truth = stats.gaussian_kde(X_truth[:,ii], bw_method=0.2)
        kernel_label = stats.gaussian_kde(X_label[:,ii], bw_method=0.2)
        kernel_pred = stats.gaussian_kde(X_pred[:,ii], bw_method=0.2)
        kl_tl = np.sum( kernel_truth(x_kl)*(x_kl[1]-x_kl[0]) * np.log(  (kernel_truth(x_kl) + 1e-16) /  ( kernel_label(x_kl) + 1e-16)  ) )
        kl_tp = np.sum( kernel_truth(x_kl)*(x_kl[1]-x_kl[0]) * np.log(  (kernel_truth(x_kl) + 1e-16) /  ( kernel_pred(x_kl) + 1e-16)  ) )
        kl_results[0,ii] = kl_tl
        kl_results[1,ii] = kl_tp

        if ii == 8 or ii == 9:
            # 1d marginal truth
            plt.figure()
            plt.cla()
            plt.fill_between(x_kl,kernel_truth(x_kl),alpha=0.4)
            # plt.xlabel('d=' + str(ii+1),fontsize=23)
            plt.xticks(fontsize=14)
            plt.tick_params(left = False, labelleft = False)
            if ii == 9:
                plt.xlim([-3,3])
                kltruth_1d_9 = np.vstack((x_kl,kernel_truth(x_kl)))
            elif ii == 8:
                plt.xlim([-3,3])
                kltruth_1d_8 = np.vstack((x_kl,kernel_truth(x_kl)))
            plt.savefig(root_output + data_name + str(ii+1) + 'd_truth.png', bbox_inches="tight", dpi=300 )

            # 1d marginal label
            plt.figure()
            plt.cla()
            plt.fill_between(x_kl,kernel_label(x_kl),alpha=0.4)
            # plt.xlabel('d=' + str(ii+1),fontsize=23)
            if ii == 9:
                plt.xlim([-3,3])
                kllabel_1d_9 = np.vstack((x_kl,kernel_label(x_kl)))
            elif ii == 8:
                plt.xlim([-3,3])
                kllabel_1d_8 = np.vstack((x_kl,kernel_label(x_kl)))
            plt.xticks(fontsize=14)
            plt.tick_params(left = False, labelleft = False)
            plt.savefig(root_output + data_name + str(ii+1) + 'd_label.png', bbox_inches="tight", dpi=300 )

            # 1d marginal pred
            plt.figure()
            plt.cla()
            plt.fill_between(x_kl,kernel_pred(x_kl),alpha=0.4)
            # plt.xlabel('d=' + str(ii+1),fontsize=23)
            if ii == 9:
                plt.xlim([-3,3])
                klpred_1d_9 = np.vstack((x_kl,kernel_pred(x_kl)))
            elif ii == 8:
                plt.xlim([-3,3])
                klpred_1d_8 = np.vstack((x_kl,kernel_pred(x_kl)))
            plt.xticks(fontsize=14)
            plt.tick_params(left = False, labelleft = False)
            plt.savefig(root_output + data_name + str(ii+1) + 'd_pred.png', bbox_inches="tight", dpi=300 )

    
    # 2d KDE
    # truth: xi, p
    xi = X_truth[:,1]
    p = X_truth[:,4]
    deltaX = (max(xi) - min(xi))/100
    deltaY = (max(p) - min(p))/100
    xmin = min(xi) - deltaX
    xmax = max(xi) + deltaX
    ymin = min(p) - deltaY
    ymax = max(p) + deltaY
    xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([xi, p])
    plt.figure()
    plt.cla()
    kernel2d_truth = stats.gaussian_kde(values)
    f_truth = np.reshape(kernel2d_truth(positions).T, xx.shape)
    plt.contourf(xx, yy, np.log10(f_truth+1e-10), cmap='coolwarm')
    plt.xlim( -5, 2.5)
    plt.ylim(-3,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_output + '2dtruth_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # label: xi, p
    xi = X_label[:,1]
    p = X_label[:,4]
    values = np.vstack([xi, p])
    plt.figure()
    plt.cla()
    kernel2d_label = stats.gaussian_kde(values)
    f_label = np.reshape(kernel2d_label(positions).T, xx.shape)
    plt.contourf(xx, yy, np.log10(f_label+1e-10), cmap='coolwarm')
    plt.xlim( -5, 2.5)
    plt.ylim(-3,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_output + '2dlabel_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # pred: xi, p
    xi = X_pred[:,1]
    p = X_pred[:,4]
    values = np.vstack([xi, p])
    plt.figure()
    plt.cla()
    kernel2d_pred = stats.gaussian_kde(values)
    f_pred = np.reshape(kernel2d_pred(positions).T, xx.shape)
    plt.contourf(xx, yy, np.log10(f_pred+1e-10), cmap='coolwarm')
    plt.xlim( -5, 2.5)
    plt.ylim(-3,3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_output + '2dnn_' + data_name + '.png', bbox_inches="tight", dpi=300 )

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
    plt.savefig(root_output + 'KL_' + data_name + '.png', bbox_inches="tight", dpi=300 )

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
    plt.savefig(root_output + 'mean_' + data_name + '.png', bbox_inches="tight", dpi=300 )

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
    plt.savefig(root_output + 'std_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    np.savez(root_plot+data_name+'.npz',xdim=xdim, mean_results=mean_results, std_results=std_results, kl_results=kl_results, \
             kltruth_1d_8=kltruth_1d_8, kltruth_1d_9=kltruth_1d_9,\
             kllabel_1d_8=kllabel_1d_8, kllabel_1d_9=kllabel_1d_9,\
             klpred_1d_8=klpred_1d_8, klpred_1d_9=klpred_1d_9,\
             xx=xx, yy=yy, kltruth_2d=f_truth, kllabel_2d=f_label, klpred_2d=f_pred )




def collect_results_miniboone(data_name):
    make_folder(root_plot)
    make_folder(root_output)
    with open(root_orbit + data_name+'.npy', 'rb') as f:
        X_truth = np.load(f)
        X_T = np.load(f)
        X_label = np.load(f)
    with open(root_orbit + data_name+'_nn.npy', 'rb') as f:
        X_pred = np.load(f)

    xdim = np.shape(X_pred)[1]
    print(xdim)
    kl_results = np.zeros((2,xdim))
    mean_results = np.zeros((3,xdim))
    mean_results[0,:] = np.mean(X_truth,axis=0) 
    mean_results[1,:] = np.mean(X_label,axis=0)
    mean_results[2,:] = np.mean(X_pred,axis=0)
    std_results = np.zeros((3,xdim))
    std_results[0,:] = np.std(X_truth,axis=0) 
    std_results[1,:] = np.std(X_label,axis=0)
    std_results[2,:] = np.std(X_pred,axis=0)
    # marginal KL
    for ii in range(xdim):
    # for ii in range(1):
        print(ii)
        # for kl_div
        xl,xr = np.min(X_truth[:,ii]),np.max(X_truth[:,ii])
        x_kl = np.linspace(xl,xr, 1000)
        kernel_truth = stats.gaussian_kde(X_truth[:,ii], bw_method=0.2)
        kernel_label = stats.gaussian_kde(X_label[:,ii], bw_method=0.2)
        kernel_pred = stats.gaussian_kde(X_pred[:,ii], bw_method=0.2)
        kl_tl = np.sum( kernel_truth(x_kl)*(x_kl[1]-x_kl[0]) * np.log(  (kernel_truth(x_kl) + 1e-16) /  ( kernel_label(x_kl) + 1e-16)  ) )
        kl_tp = np.sum( kernel_truth(x_kl)*(x_kl[1]-x_kl[0]) * np.log(  (kernel_truth(x_kl) + 1e-16) /  ( kernel_pred(x_kl) + 1e-16)  ) )
        kl_results[0,ii] = kl_tl
        kl_results[1,ii] = kl_tp

        if ii == 0 or ii == 15:
            # 1d marginal truth
            plt.figure()
            plt.cla()
            plt.fill_between(x_kl,kernel_truth(x_kl),alpha=0.4)
            plt.xticks(fontsize=14)
            plt.tick_params(left = False, labelleft = False)
            if ii == 15:
                plt.xlim([-1,5])
                kltruth_1d_15 = np.vstack((x_kl,kernel_truth(x_kl)))
            elif ii == 0:
                plt.xlim([-3,5])
                kltruth_1d_0 = np.vstack((x_kl,kernel_truth(x_kl)))            
            plt.savefig(root_output + data_name + str(ii+1) + 'd_truth.png', bbox_inches="tight", dpi=300 )

            # 1d marginal label
            plt.figure()
            plt.cla()
            plt.fill_between(x_kl,kernel_label(x_kl),alpha=0.4)
            if ii == 15:
                plt.xlim([-1,5])
                kllabel_1d_15 = np.vstack((x_kl,kernel_label(x_kl)))
            elif ii == 0:
                plt.xlim([-3,5])
                kllabel_1d_0 = np.vstack((x_kl,kernel_label(x_kl)))
            plt.xticks(fontsize=14)
            plt.tick_params(left = False, labelleft = False)
            plt.savefig(root_output + data_name + str(ii+1) + 'd_label.png', bbox_inches="tight", dpi=300 )

            # 1d marginal pred
            plt.figure()
            plt.cla()
            plt.fill_between(x_kl,kernel_pred(x_kl),alpha=0.4)
            if ii == 15:
                plt.xlim([-1,5])
                klpred_1d_15 = np.vstack((x_kl,kernel_pred(x_kl)))
            elif ii == 0:
                plt.xlim([-3,5])
                klpred_1d_0 = np.vstack((x_kl,kernel_pred(x_kl)))
            plt.xticks(fontsize=14)
            plt.tick_params(left = False, labelleft = False)
            plt.savefig(root_output + data_name + str(ii+1) + 'd_pred.png', bbox_inches="tight", dpi=300 )


    #2d KDE
    #truth: xi, p
    xi = X_truth[:,16]
    p = X_truth[:,30]
    deltaX = (max(xi) - min(xi))/100
    deltaY = (max(p) - min(p))/100
    xmin = min(xi) - deltaX
    xmax = max(xi) + deltaX
    ymin = min(p) - deltaY
    ymax = max(p) + deltaY
    xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([xi, p])
    plt.figure()
    plt.cla()
    kernel2d_truth = stats.gaussian_kde(values)
    f_truth = np.reshape(kernel2d_truth(positions).T, xx.shape)
    plt.contourf(xx, yy, np.log10(f_truth+1e-10), cmap='coolwarm')
    plt.xlim( -1, 5)
    plt.ylim(-3,5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_output + '2dtruth_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # label: xi, p
    xi = X_label[:,16]
    p = X_label[:,30]
    values = np.vstack([xi, p])
    plt.figure()
    plt.cla()
    kernel2d_label = stats.gaussian_kde(values)
    f_label = np.reshape(kernel2d_label(positions).T, xx.shape)
    plt.contourf(xx, yy, np.log10(f_label+1e-10), cmap='coolwarm')
    plt.xlim( -1, 5)
    plt.ylim(-3,5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_output + '2dlabel_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # pred: xi, p
    xi = X_pred[:,16]
    p = X_pred[:,30]
    values = np.vstack([xi, p])
    plt.figure()
    plt.cla()
    kernel2d_pred = stats.gaussian_kde(values)
    f_pred = np.reshape(kernel2d_pred(positions).T, xx.shape)
    plt.contourf(xx, yy, np.log10(f_pred+1e-10), cmap='coolwarm')
    plt.xlim( -1, 5)
    plt.ylim(-3,5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(root_output + '2dnn_' + data_name + '.png', bbox_inches="tight", dpi=300 )

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
    plt.savefig(root_output + 'KL_' + data_name + '.png', bbox_inches="tight", dpi=300 )

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
    plt.savefig(root_output + 'mean_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    # plot std
    plt.figure()
    plt.cla()
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[0,:], color="r")
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[1,:], color="b")
    plt.plot(np.arange(xdim, dtype=int)+1,std_results[2,:], color="k")

    plt.xlabel('dimension',fontsize=20)
    plt.xlim(1, xdim)
    plt.ylim(0.9,1.3)
    plt.ylabel('std',fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()        
    plt.legend(['Truth','Labeled', 'Generated'],fontsize = 12, loc = 'upper right')
    plt.savefig(root_output + 'std_' + data_name + '.png', bbox_inches="tight", dpi=300 )

    np.savez(root_plot+data_name+'.npz',xdim=xdim, mean_results=mean_results, std_results=std_results, kl_results=kl_results, \
             kltruth_1d_0=kltruth_1d_0, kltruth_1d_15=kltruth_1d_15,\
             kllabel_1d_0=kllabel_1d_0, kllabel_1d_15=kllabel_1d_15,\
             klpred_1d_0=klpred_1d_0, klpred_1d_15=klpred_1d_15,\
             xx=xx, yy=yy, kltruth_2d=f_truth, kllabel_2d=f_label, klpred_2d=f_pred )









def main():

    for data in sys.argv[1:]:

        if data == 'power':
            collect_results_power(data)

        elif data == 'gas':
            collect_results_gas(data)

        elif data == 'hepmass':
            collect_results_hepmass(data)

        elif data == 'miniboone':
            collect_results_miniboone(data)

        else:
            print('{0} is not a valid dataset'.format(data))
            continue


if __name__ == '__main__':
    main()
