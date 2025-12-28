#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:25:23 2022

@author: fabioditrani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:02:48 2022

@author: fabioditrani
"""
import numpy as np
from numba import njit


class fif_ultranest:
    def __init__(self, lambda_obs,flux_obs,err_obs,flux_2d,ind_obs,ind_err_obs,vdisp,lambda_models,flux_models, age_uni,age_grid, zh_grid,alpha_grid,
                 blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx,name_ind,res_instr,fwhm_mod,sel_fif,sel_ind,nsimul):
        
        #input information
        self.lambda_obs = lambda_obs
        self.flux_obs = flux_obs
        self.err_obs = err_obs
        self.ind_obs = ind_obs
        self.ind_err_obs = ind_err_obs
        self.vdisp = vdisp
        self.lambda_models = lambda_models
        self.flux_models = flux_models
        self.age_grid = age_grid
        self.age_uni = age_uni
        self.zh_grid = zh_grid
        self.alpha_grid = alpha_grid
        self.blue_sx = blue_sx
        self.blue_dx = blue_dx
        self.feat_sx = feat_sx
        self.feat_dx = feat_dx
        self.red_sx = red_sx
        self.red_dx = red_dx
        self.name_ind = name_ind
        self.sel_fif = sel_fif
        self.sel_ind = sel_ind


        
        self.clight = 299792.458 #km/s
        
        self.sel_obs = np.where((self.lambda_obs >= (blue_sx[0]-100)) & (self.lambda_obs <= (red_dx[-1]+100)))[0]
        self.sel_mod = np.where((self.lambda_models >= (blue_sx[0]-100)) & (self.lambda_models <= (red_dx[-1]+100)))[0]

        self.lambda_obs_cut = self.lambda_obs[self.sel_obs]
        self.lambda_models_cut = self.lambda_models[self.sel_mod]
        self.models_spec_cut = self.flux_models[:,:,:,self.sel_mod]#da sistemare meglio
        flux_2d_cut = flux_2d[self.sel_obs,:]
        flux_obs_cut = self.flux_obs[self.sel_obs]
        err_obs_sel = self.err_obs[self.sel_obs]
        
        self.fluxfeat_obs, self.errfeat_obs_pre, self.selblue, self.selred, self.selfeat, self.meanbluex, self.meanredx,self.iniblue,self.ifiblue,self.inired,self.ifired, self.dblue_vec,self.dred_vec,self.titblue_vec,self.titred_vec=fif_obs_pro_more(self.lambda_obs_cut,flux_obs_cut,err_obs_sel,
                                                                                                                                  self.name_ind[self.sel_fif],self.blue_sx[self.sel_fif],
                                                                                                                                  self.blue_dx[self.sel_fif],self.feat_sx[self.sel_fif],self.feat_dx[self.sel_fif],
                                                                                                                                  self.red_sx[self.sel_fif],self.red_dx[self.sel_fif])
        
        
        self.fluxfeat_obs = np.asarray(self.fluxfeat_obs,dtype = 'object')
        self.errfeat_obs_pre = np.asarray(self.errfeat_obs_pre,dtype = 'object')
        self.selblue = np.asarray(self.selblue,dtype = 'object')
        self.selred = np.asarray(self.selred,dtype = 'object')
        self.selfeat = np.asarray(self.selfeat,dtype = 'object')
        self.meanbluex = np.asarray(self.meanbluex,dtype = 'object')
        self.meanredx = np.asarray(self.meanredx,dtype = 'object')
        self.dblue_vec = np.asarray(self.dblue_vec,dtype = 'object')
        self.dred_vec = np.asarray(self.dred_vec,dtype = 'object')
        self.titblue_vec = np.asarray(self.titblue_vec,dtype = 'object')
        self.titred_vec = np.asarray(self.titred_vec,dtype = 'object')

        self.errfeat_obs = fif_obs_with_errcont(self.lambda_obs_cut,flux_2d_cut,err_obs_sel,self.errfeat_obs_pre,self.name_ind[self.sel_fif],self.blue_sx[self.sel_fif],
                            self.blue_dx[self.sel_fif],self.feat_sx[self.sel_fif],self.feat_dx[self.sel_fif],
                              self.red_sx[self.sel_fif],self.red_dx[self.sel_fif],self.dblue_vec,self.dred_vec,self.titblue_vec,self.titred_vec,
                              self.selblue,self.selred,self.selfeat,self.iniblue,self.ifiblue,self.inired,self.ifired,nsimul)
        self.errfeat_obs = np.asarray(self.errfeat_obs,dtype = 'object')
        
        
        
        self.ind_selected1 = np.where( (name_ind[sel_fif] == 'Fe4383') |(name_ind[sel_fif] == 'Mgb') |  (name_ind[sel_fif] == 'Fe5270')  
                                      |  (name_ind[sel_fif] == 'Fe5335'))[0]
        
        self.ind_selected2 = np.where( (name_ind[sel_fif] == 'Gband4300') |  (name_ind[sel_fif] == 'CaK') |(name_ind[sel_fif] == 'CaH')  |  (name_ind[sel_fif] == 'HgammaF')|(name_ind[sel_fif] == 'HdeltaF')
                                  |  (name_ind[sel_fif] == 'Hbeta') |  (name_ind[sel_fif] == 'Hbo'))[0]
        
        self.ind_selected2 = np.where( (name_ind[sel_fif] == 'Gband4300'))[0]
        
        
        
        
        
        fwhm_instr =self.lambda_obs_cut/res_instr
        csffw = np.interp(self.lambda_models_cut, self.lambda_obs_cut, fwhm_instr)
        fwhm_instr_temp = csffw
        fwhm_gal = vdisp/self.clight*self.lambda_models_cut*2.355
        
        self.sigma_fin_cut = np.sqrt(fwhm_gal**2+fwhm_instr_temp**2-fwhm_mod**2)/2.355
        self.sigma_fin_cut[np.isnan(self.sigma_fin_cut)] = 0

    def my_prior_transform(self,cube):
        params = cube.copy()

        # transform age parameter: uniform prior
        lo = self.age_grid[0]
        hi = self.age_grid[-1]-0.00001
        params[:,0] = cube[:,0] * (hi - lo) + lo

        
        # transform zh parameter: uniform prior
        lo = self.zh_grid[0]
        hi = self.zh_grid[-1]
        params[:,1] = cube[:,1] * (hi - lo) + lo
        
        # transform alpha parameter: uniform prior
        lo = self.alpha_grid[0]
        hi = self.alpha_grid[-1]
        params[:,2] = cube[:,2] * (hi - lo) + lo
        

        
        
        return params
    
    def log_likelihood_ultranest(self,theta):
        '''
        Computes the log-likelihood for the age and metallicity parameters
        ------------------------------------------------------------------
        theta : array of free parameters
        visit : number of visits
        mydata : observed spectra
        myfeat : index
        k : multiplier for index
        age : initial array of ages
        metal : initial array of metallicities
        mymodels : array of used models
        z0 : zero redshift
        zobs : observed redshift
        kernel : convolution kernel of the galaxy image
        ---------------------------------------------------------------------------
        RETURN : log-likelihood or -inf whether theta is out of the parameter space
        '''
        

        newage = theta[:,0]
        sel_noage = newage > self.age_uni
        newmetal = theta[:,1]
        newalpha = theta[:,2]
        
        flux_mod = interpmodel_tau_all(newmetal,newage,newalpha,self.zh_grid,self.age_grid,self.alpha_grid,self.flux_models)
        flux_mod_conv = varsmooth_vec(self.lambda_models_cut,flux_mod[:,self.sel_mod],self.sigma_fin_cut)

        flux_mod_conv_rebin = vectorized_interp(self.lambda_obs_cut,self.lambda_models_cut,flux_mod_conv)

        flux_mod_2 = interpmodel_all(newmetal,newage,self.zh_grid,self.age_grid,self.flux_models[:,:,1,:])
        flux_mod_conv_2 = varsmooth_vec(self.lambda_models_cut,flux_mod_2[:,self.sel_mod],self.sigma_fin_cut)

        flux_mod_conv_rebin_2 = vectorized_interp(self.lambda_obs_cut,self.lambda_models_cut,flux_mod_conv_2)

        chi2_fif1 = calcchi2_fif_all_pro_already(self.lambda_obs_cut,flux_mod_conv_rebin,self.fluxfeat_obs[self.ind_selected1],self.errfeat_obs[self.ind_selected1],
                                                  self.name_ind[self.sel_fif][self.ind_selected1],self.selblue[self.ind_selected1],self.selfeat[self.ind_selected1],
                                                  self.selred[self.ind_selected1],self.meanbluex[self.ind_selected1],self.meanredx[self.ind_selected1],
                                                  self.iniblue[self.ind_selected1],self.ifiblue[self.ind_selected1],self.inired[self.ind_selected1],
                                                  self.ifired[self.ind_selected1],self.dblue_vec[self.ind_selected1],self.dred_vec[self.ind_selected1],
                                                  self.titblue_vec[self.ind_selected1],self.titred_vec[self.ind_selected1],self.blue_sx[self.sel_fif][self.ind_selected1],
                                                  self.blue_dx[self.sel_fif][self.ind_selected1],self.feat_sx[self.sel_fif][self.ind_selected1],
                                                  self.feat_dx[self.sel_fif][self.ind_selected1],
                                                  self.red_sx[self.sel_fif][self.ind_selected1],self.red_dx[self.sel_fif][self.ind_selected1])
        
        chi2_fif2 = calcchi2_fif_all_pro_already(self.lambda_obs_cut,flux_mod_conv_rebin_2,self.fluxfeat_obs[self.ind_selected2],self.errfeat_obs[self.ind_selected2],
                                                  self.name_ind[self.sel_fif][self.ind_selected2],self.selblue[self.ind_selected2],self.selfeat[self.ind_selected2],
                                                  self.selred[self.ind_selected2],self.meanbluex[self.ind_selected2],self.meanredx[self.ind_selected2],
                                                  self.iniblue[self.ind_selected2],self.ifiblue[self.ind_selected2],self.inired[self.ind_selected2],
                                                  self.ifired[self.ind_selected2],self.dblue_vec[self.ind_selected2],self.dred_vec[self.ind_selected2],
                                                  self.titblue_vec[self.ind_selected2],self.titred_vec[self.ind_selected2],self.blue_sx[self.sel_fif][self.ind_selected2],
                                                  self.blue_dx[self.sel_fif][self.ind_selected2],self.feat_sx[self.sel_fif][self.ind_selected2],
                                                  self.feat_dx[self.sel_fif][self.ind_selected2],
                                                  self.red_sx[self.sel_fif][self.ind_selected2],self.red_dx[self.sel_fif][self.ind_selected2])
        
        loglikelihood = (-(chi2_fif1+chi2_fif2)/2)
        loglikelihood[sel_noage] = -1e32
        return loglikelihood


def vectorized_interp(x, xp, fp_arrays):
    """
    Vectorized interpolation for multiple sets of y-values (fp_arrays) with the same x-values (xp).

    Parameters:
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xp : 1-D sequence of floats
        The x-coordinates of the data points, must be increasing.
    fp_arrays : 2-D array_like
        The y-coordinates of the data points, where each row corresponds to a different set of y-values.
    
    Returns:
    np.ndarray
        Interpolated values for each set of y-values at the coordinates x.
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    xp = np.asarray(xp)
    fp_arrays = np.asarray(fp_arrays)

    # Ensure xp is 1-dimensional and fp_arrays is 2-dimensional
    assert xp.ndim == 1, "xp should be a 1-dimensional array."
    assert fp_arrays.ndim == 2, "fp_arrays should be a 2-dimensional array where each row is a different set of y-values."

    # Find indices in xp for the positions in x
    indices = np.searchsorted(xp, x, side='left')
    indices = np.clip(indices, 1, len(xp) - 1)

    # Gather the relevant x and y points for interpolation
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp_arrays[:, indices - 1]
    y1 = fp_arrays[:, indices]

    # Perform the linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    interpolated_values = y0 + slope * (x - x0)

    return interpolated_values

@njit
def numba_vectorized_interp(x, xp, fp_arrays):
    x = np.asarray(x)
    xp = np.asarray(xp)
    fp_arrays = np.asarray(fp_arrays)

    # Find indices in xp for the positions in x
    indices = np.searchsorted(xp, x, side='left')
    indices = np.clip(indices, 1, len(xp) - 1)

    # Gather the relevant x and y points for interpolation
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp_arrays[:, indices - 1]
    y1 = fp_arrays[:, indices]

    # Perform the linear interpolation
    slope = (y1 - y0) / (x1 - x0)
    interpolated_values = y0 + slope * (x - x0)

    return interpolated_values

@njit
def fast_interp(x, xp, fp_arrays):
    n_rows, n_cols = fp_arrays.shape
    n_points = len(x)
    result = np.empty((n_rows, n_points))

    for i in range(n_points):
        xi = x[i]
        # Trova la posizione in xp
        j = np.searchsorted(xp, xi)
        if j == 0:
            j = 1
        elif j >= len(xp):
            j = len(xp) - 1

        x0 = xp[j - 1]
        x1 = xp[j]
        t = (xi - x0) / (x1 - x0)

        for r in range(n_rows):
            y0 = fp_arrays[r, j - 1]
            y1 = fp_arrays[r, j]
            result[r, i] = y0 + t * (y1 - y0)
    return result

def calcchi2_fif_all_pro_already(x,y,yfeat_obs,err_obs,name_ind,selblue,selfeat,selred,meanbluex,meanredx,iniblue,ifiblue,inired,ifired,dblue_vec,dred_vec,titblue_vec,titred_vec,blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx):
    '''
    Computes chi2 for given model and observed spectrum
    --------------------------------------------------
    x : model lambda
    y : model flux
    xobs : observed lambda
    yobs : observed flux
    err : observed error
    zobs : galaxy redshift
    myfeat : index
    k : multiplier for index
    -------------
    RETURN : chi2
    '''
    chi2 = 0
    for i in range(len(name_ind)):
        
            
        meanblue = np.sum(y[:,iniblue[i]:ifiblue[i]+1]*dblue_vec[i],axis = 1)/titblue_vec[i]
        meanred = np.sum(y[:,inired[i]:ifired[i]+1]*dred_vec[i], axis = 1)/titred_vec[i]
    
    
        m = (meanblue-meanred)/(meanbluex[i]-meanredx[i])
        q = (meanbluex[i]*meanred-meanredx[i]*meanblue)/(meanbluex[i]-meanredx[i])
        ycont = m[:,None]*x[selfeat[i]][None,:]+q[:,None]
    
        yfeat = y[:,selfeat[i]]/ycont
    
        #chi2 = chi2 + sum(((yfeat_obs[i]-yfeat)/err_obs[i])**2-np.log(1/err_obs[i]**2))#
        var = 1/(err_obs[i])**2
        #print((var*(yfeat_obs[i]-yfeat)**2 - np.log(var) + np.log(2.*np.pi)).shape)
        # if((name_ind[i] == 'Fe5270') | (name_ind[i] == 'Fe5335')):
        #     chi2 = chi2 + np.nansum(var*(yfeat_obs[i]-yfeat)**2 - np.log(var) + np.log(2.*np.pi),axis = 1)*0
        # else:
        chi2 = chi2 + np.nansum(var*(yfeat_obs[i]-yfeat)**2 - np.log(var) + np.log(2.*np.pi),axis = 1)
        #print(chi2.shape)
        #chi2 = chi2 + sum(var*(yfeat_obs[i]-yfeat)**2)#
        
    #chi2r = chi2/(len(y1[sel])-2)

    return chi2

def interpmodel_all(newmetal, newage, metal, age, mymodels):
    '''
    Compute logarithmically in age and metallicity a model with fixed age and metallicity
    ------------------------------------------
    newmetal : new metallicity (log10)
    newage : new age [Gyr]
    metal : metallicity grid (log10)
    age : age grid [Gyr]
    mymodels : models
    --------------------------------------------------------------
    RETURN : lambda and flux of model with new age and metallicity
    '''
    iage = np.searchsorted(age,newage,side = 'right')-1
    imet = np.searchsorted(metal,newmetal,side = 'right')-1
    
    aged = [age[iage],age[iage+1]]
    met = [metal[imet],metal[imet+1]]
    #lower left of the square
    y1 = mymodels[iage,imet,:]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,:]
    #lower right of the square
    y3 = mymodels[iage+1,imet,:]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,:]
    #vertical interpolation on the square, left side
    ymet1 = (newmetal-met[0])[:,None]*(y2-y1)/(met[-1]-met[0])[:,None]+y1
    
    #ymet1 = y2**((newmetal-met[0])/(met[-1]-met[0]))*y1**(1-(newmetal-met[0])/(met[1]-met[0]))
    #vertical interpolation on the square, right side
    ymet2 = (newmetal-met[0])[:,None]*(y4-y3)/(met[-1]-met[0])[:,None]+y3
    #ymet2 = y4**((newmetal-met[0])/(met[-1]-met[0]))*y3**(1-(newmetal-met[0])/(met[1]-met[0]))
    #final interpolation (interior point of the square)
    y = (newage-aged[0])[:,None]*(ymet2-ymet1)/(aged[-1]-aged[0])[:,None]+ymet1
    #y = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    return y

def fif_obs_with_errcont(xobs,yobs,err,errfeat_obs_pre,name_ind,blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx,dblue_vec,
                         dred_vec,titblue_vec,titred_vec,selblue,selred,selfeat,iniblue,ifiblue,inired,ifired,nsimul):
    '''
    Computes chi2 for given model and observed spectrum
    --------------------------------------------------
    x : model lambda
    y : model flux
    xobs : observed lambda
    yobs : observed flux
    err : observed error
    zobs : galaxy redshift
    myfeat : index
    k : multiplier for index
    -------------
    RETURN : chi2
    '''


    meanbluex = (blue_sx+blue_dx)/2
    meanredx = (red_sx+red_dx)/2
    err_final = []
    
    for i in range(len(name_ind)):
        flux_simul_cont = np.zeros([len(errfeat_obs_pre[i]),nsimul])
        for j in range(nsimul):
            if((xobs[0] < blue_sx[i]) & (xobs[-1] > red_dx[i])):


                #bisogna mettere una flag per i nan 
                new_dred = dred_vec[i][err[inired[i]:ifired[i]+1] != 999]
                new_dblue = dblue_vec[i][err[iniblue[i]:ifiblue[i]+1] != 999]
                
                new_titred = np.nansum(new_dred)
                new_titblue = np.nansum(new_dblue)
                
                new_yobs_blue = yobs[:,j][iniblue[i]:ifiblue[i]+1][err[iniblue[i]:ifiblue[i]+1] != 999]
                new_yobs_red = yobs[:,j][inired[i]:ifired[i]+1][err[inired[i]:ifired[i]+1] != 999]
                
                
                new_meanblue_obs = np.nansum(new_yobs_blue*new_dblue)/new_titblue
                new_meanred_obs = np.nansum(new_yobs_red*new_dred)/new_titred
                
                
                m_obs = (new_meanblue_obs-new_meanred_obs)/(meanbluex[i]-meanredx[i])
                #m_err = np.sqrt(new_meanblue_obs_err**2 + new_meanred_obs_err**2)/(meanbluex[i]-meanredx[i])
                
                q_obs = (meanbluex[i]*new_meanred_obs-meanredx[i]*new_meanblue_obs)/(meanbluex[i]-meanredx[i])
    
                ycont_obs = m_obs*xobs[selfeat[i]]+q_obs
                
                flux_simul_cont[:,j] = yobs[:,j][selfeat[i]]/ycont_obs
                #errfeat_obs.append(np.sqrt((err[selfeat[i]]/ycont_obs)**2+(yobs[selfeat[i]]/ycont_obs**2)**2 * ycont_err**2))
                #print(np.sqrt((err[selfeat[i]]/ycont_obs)**2+(yobs[selfeat[i]]/ycont_obs**2)**2 * ycont_err**2))
                #print(np.sqrt((err[selfeat[i]]/ycont_obs)**2+(yobs[selfeat[i]]/ycont_obs**2)**2 * ycont_err**2))
                # print(yobs[selfeat[i]],ycont_obs)
        err_cont = np.std(flux_simul_cont,axis = 1)
        err_final.append(np.sqrt(errfeat_obs_pre[i]**2+err_cont**2))

    return err_final



def fif_obs_pro_more(xobs,yobs,err,name_ind,blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx):
    '''
    Computes chi2 for given model and observed spectrum
    --------------------------------------------------
    x : model lambda
    y : model flux
    xobs : observed lambda
    yobs : observed flux
    err : observed error
    zobs : galaxy redshift
    myfeat : index
    k : multiplier for index
    -------------
    RETURN : chi2
    '''
    yfeat_obs = []
    errfeat_obs = []
    iniblue = np.zeros([len(name_ind)],dtype = int)
    ifiblue = np.zeros([len(name_ind)],dtype = int)
    inired = np.zeros([len(name_ind)],dtype = int)
    ifired = np.zeros([len(name_ind)],dtype = int)
    dblue_vec = []
    dred_vec = []
    titblue_vec = []
    titred_vec = []
    selblue = []
    selred = []
    selfeat = []
    meanbluex = (blue_sx+blue_dx)/2
    meanredx = (red_sx+red_dx)/2
    for i in range(len(name_ind)):
        if((xobs[0] < blue_sx[i]) & (xobs[-1] > red_dx[i])):
            iniblue[i]     = np.max(np.where(xobs < blue_sx[i])[0]) + 1
            ifiblue[i]     = np.max(np.where(xobs < blue_dx[i])[0])
            inired[i]     = np.max(np.where(xobs < red_sx[i])[0]) + 1
            ifired[i]     = np.max(np.where(xobs < red_dx[i])[0])
            dblue = (xobs[iniblue[i]:ifiblue[i]+1]-xobs[iniblue[i]-1:ifiblue[i]])*0.5+(xobs[iniblue[i]+1:ifiblue[i]+2]-xobs[iniblue[i]:ifiblue[i]+1])*0.5
            dred = (xobs[inired[i]:ifired[i]+1]-xobs[inired[i]-1:ifired[i]])*0.5+(xobs[inired[i]+1:ifired[i]+2]-xobs[inired[i]:ifired[i]+1])*0.5
            
            #parte blue sx
            if (xobs[iniblue[i]] - (xobs[iniblue[i]]-xobs[iniblue[i]-1])*0.5 > blue_sx[i]):
                iniblue[i] = iniblue[i] - 1
                dblue = np.append(abs(xobs[iniblue[i]] + (xobs[iniblue[i]+1] - xobs[iniblue[i]])*0.5 - blue_sx[i]),dblue)
            elif (xobs[iniblue[i]] - (xobs[iniblue[i]]-xobs[iniblue[i]-1])*0.5 < blue_sx[i]):
                dblue[0] = dblue[0] - (blue_sx[i] - (xobs[iniblue[i]] - (xobs[iniblue[i]] - xobs[iniblue[i]-1])*0.5))

            #parte blue dx
            if (xobs[ifiblue[i]] + (xobs[ifiblue[i]+1]-xobs[ifiblue[i]])*0.5 < blue_dx[i]):
                ifiblue[i] = ifiblue[i] + 1
                dblue = np.append(dblue,abs((xobs[ifiblue[i]] - (xobs[ifiblue[i]] - xobs[ifiblue[i] - 1])*0.5) - blue_dx[i]))
            elif (xobs[ifiblue[i]] + (xobs[ifiblue[i] + 1]-xobs[ifiblue[i]])*0.5 > blue_dx[i]):
                dblue[-1] = dblue[-1] - abs(xobs[ifiblue[i]] + (xobs[ifiblue[i] + 1] - xobs[ifiblue[i]])*0.5 - blue_dx[i])

            #parte red sx
            if (xobs[inired[i]] - (xobs[inired[i]]-xobs[inired[i]-1])*0.5 > red_sx[i]):
                inired[i] = inired[i] - 1
                dred = np.append(abs(xobs[inired[i]] + (xobs[inired[i] + 1] - xobs[inired[i]])*0.5 - red_sx[i]),dred)
            elif (xobs[inired[i]] - (xobs[inired[i]]-xobs[inired[i]-1])*0.5 < red_sx[i]):
                dred[0] = dred[0] - abs(xobs[inired[i]] - (xobs[inired[i]] - xobs[inired[i]-1])*0.5 - red_sx[i])

            #parte red dx
            if (xobs[ifired[i]] + (xobs[ifired[i]+1]-xobs[ifired[i]])*0.5 < red_dx[i]):
                ifired[i] = ifired[i] + 1
                dred = np.append(dred,abs(xobs[ifired[i]] - (xobs[ifired[i] + 1] - xobs[ifired[i]])*0.5 - red_dx[i]))
            elif (xobs[ifired[i]] + (xobs[ifired[i] + 1]-xobs[ifired[i]])*0.5 > red_dx[i]):
                dred[-1] = dred[-1] - abs(xobs[ifired[i]] + (xobs[ifired[i] + 1] - xobs[ifired[i]])*0.5 - red_dx[i])
                
            dblue_vec.append(dblue)
            dred_vec.append(dred)
            
            titblue_vec.append(sum(dblue))
            titred_vec.append(sum(dred))
            
            #bisogna mettere una flag per i nan 
            new_dred = dred[err[inired[i]:ifired[i]+1] != 999]
            new_dblue = dblue[err[iniblue[i]:ifiblue[i]+1] != 999]
            
            new_titred = np.nansum(new_dred)
            new_titblue = np.nansum(new_dblue)
            #titblue = sum(dblue)
            #titred = sum(dred)
            #print(dblue)
            
            #meanblue_obs = np.sum(yobs[iniblue[i]:ifiblue[i]+1]*dblue)/titblue
            #meanred_obs = np.sum(yobs[inired[i]:ifired[i]+1]*dred)/titred
            
            new_yobs_blue = yobs[iniblue[i]:ifiblue[i]+1][err[iniblue[i]:ifiblue[i]+1] != 999]
            new_yobs_red = yobs[inired[i]:ifired[i]+1][err[inired[i]:ifired[i]+1] != 999]
            
            new_meanblue_obs = np.nansum(new_yobs_blue*new_dblue)/new_titblue
            new_meanred_obs = np.nansum(new_yobs_red*new_dred)/new_titred
            #print(meanblue_obs,new_meanblue_obs)
            #print(meanblue_obs,meanred_obs)
            selblue.append(np.where((xobs >= blue_sx[i]) & (xobs <= blue_dx[i]))[0])
            selred.append(np.where((xobs >= red_sx[i]) & (xobs <= red_dx[i]))[0])
            selfeat.append(np.where((xobs >= feat_sx[i]) & (xobs <= feat_dx[i]))[0])
        
            #meanblue_obs = np.nanmedian(yobs[selblue[i]])
            #meanred_obs = np.nanmedian(yobs[selred[i]])
            #print(meanblue_obs,meanred_obs)
        
            #m_obs = (meanblue_obs-meanred_obs)/(meanbluex[i]-meanredx[i])
            #q_obs = (meanbluex[i]*meanred_obs-meanredx[i]*meanblue_obs)/(meanbluex[i]-meanredx[i])
            
            m_obs = (new_meanblue_obs-new_meanred_obs)/(meanbluex[i]-meanredx[i])
            q_obs = (meanbluex[i]*new_meanred_obs-meanredx[i]*new_meanblue_obs)/(meanbluex[i]-meanredx[i])
            ycont_obs = m_obs*xobs[selfeat[i]]+q_obs

            yfeat_obs.append(yobs[selfeat[i]]/ycont_obs)
            errfeat_obs.append(err[selfeat[i]]/ycont_obs)

        

    return yfeat_obs,errfeat_obs,selblue,selred,selfeat,meanbluex,meanredx,iniblue,ifiblue,inired,ifired,dblue_vec,dred_vec,titblue_vec,titred_vec



def fif_obs_with_errcont_pro(xobs,yobs,err,name_ind,blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx):
    '''
    Computes chi2 for given model and observed spectrum
    --------------------------------------------------
    x : model lambda
    y : model flux
    xobs : observed lambda
    yobs : observed flux
    err : observed error
    zobs : galaxy redshift
    myfeat : index
    k : multiplier for index
    -------------
    RETURN : chi2
    '''
    yfeat_obs = []
    errfeat_obs = []
    iniblue = np.zeros([len(name_ind)],dtype = int)
    ifiblue = np.zeros([len(name_ind)],dtype = int)
    inired = np.zeros([len(name_ind)],dtype = int)
    ifired = np.zeros([len(name_ind)],dtype = int)
    dblue_vec = []
    dred_vec = []
    titblue_vec = []
    titred_vec = []
    selblue = []
    selred = []
    selfeat = []
    meanbluex = (blue_sx+blue_dx)/2
    meanredx = (red_sx+red_dx)/2
    for i in range(len(name_ind)):
        if((xobs[0] < blue_sx[i]) & (xobs[-1] > red_dx[i])):
            iniblue[i]     = np.max(np.where(xobs < blue_sx[i])[0]) + 1
            ifiblue[i]     = np.max(np.where(xobs < blue_dx[i])[0])
            inired[i]     = np.max(np.where(xobs < red_sx[i])[0]) + 1
            ifired[i]     = np.max(np.where(xobs < red_dx[i])[0])
            dblue = (xobs[iniblue[i]:ifiblue[i]+1]-xobs[iniblue[i]-1:ifiblue[i]])*0.5+(xobs[iniblue[i]+1:ifiblue[i]+2]-xobs[iniblue[i]:ifiblue[i]+1])*0.5
            dred = (xobs[inired[i]:ifired[i]+1]-xobs[inired[i]-1:ifired[i]])*0.5+(xobs[inired[i]+1:ifired[i]+2]-xobs[inired[i]:ifired[i]+1])*0.5
            
            #parte blue sx
            if (xobs[iniblue[i]] - (xobs[iniblue[i]]-xobs[iniblue[i]-1])*0.5 > blue_sx[i]):
                iniblue[i] = iniblue[i] - 1
                dblue = np.append(abs(xobs[iniblue[i]] + (xobs[iniblue[i]+1] - xobs[iniblue[i]])*0.5 - blue_sx[i]),dblue)
            elif (xobs[iniblue[i]] - (xobs[iniblue[i]]-xobs[iniblue[i]-1])*0.5 < blue_sx[i]):
                dblue[0] = dblue[0] - (blue_sx[i] - (xobs[iniblue[i]] - (xobs[iniblue[i]] - xobs[iniblue[i]-1])*0.5))

            #parte blue dx
            if (xobs[ifiblue[i]] + (xobs[ifiblue[i]+1]-xobs[ifiblue[i]])*0.5 < blue_dx[i]):
                ifiblue[i] = ifiblue[i] + 1
                dblue = np.append(dblue,abs((xobs[ifiblue[i]] - (xobs[ifiblue[i]] - xobs[ifiblue[i] - 1])*0.5) - blue_dx[i]))
            elif (xobs[ifiblue[i]] + (xobs[ifiblue[i] + 1]-xobs[ifiblue[i]])*0.5 > blue_dx[i]):
                dblue[-1] = dblue[-1] - abs(xobs[ifiblue[i]] + (xobs[ifiblue[i] + 1] - xobs[ifiblue[i]])*0.5 - blue_dx[i])

            #parte red sx
            if (xobs[inired[i]] - (xobs[inired[i]]-xobs[inired[i]-1])*0.5 > red_sx[i]):
                inired[i] = inired[i] - 1
                dred = np.append(abs(xobs[inired[i]] + (xobs[inired[i] + 1] - xobs[inired[i]])*0.5 - red_sx[i]),dred)
            elif (xobs[inired[i]] - (xobs[inired[i]]-xobs[inired[i]-1])*0.5 < red_sx[i]):
                dred[0] = dred[0] - abs(xobs[inired[i]] - (xobs[inired[i]] - xobs[inired[i]-1])*0.5 - red_sx[i])

            #parte red dx
            if (xobs[ifired[i]] + (xobs[ifired[i]+1]-xobs[ifired[i]])*0.5 < red_dx[i]):
                ifired[i] = ifired[i] + 1
                dred = np.append(dred,abs(xobs[ifired[i]] - (xobs[ifired[i] + 1] - xobs[ifired[i]])*0.5 - red_dx[i]))
            elif (xobs[ifired[i]] + (xobs[ifired[i] + 1]-xobs[ifired[i]])*0.5 > red_dx[i]):
                dred[-1] = dred[-1] - abs(xobs[ifired[i]] + (xobs[ifired[i] + 1] - xobs[ifired[i]])*0.5 - red_dx[i])
                
            dblue_vec.append(dblue)
            dred_vec.append(dred)
            
            titblue_vec.append(sum(dblue))
            titred_vec.append(sum(dred))
            
            #bisogna mettere una flag per i nan 
            new_dred = dred[err[inired[i]:ifired[i]+1] != 999]
            new_dblue = dblue[err[iniblue[i]:ifiblue[i]+1] != 999]
            
            new_titred = np.nansum(new_dred)
            new_titblue = np.nansum(new_dblue)
            #titblue = sum(dblue)
            #titred = sum(dred)
            #print(dblue)
            
            #meanblue_obs = np.sum(yobs[iniblue[i]:ifiblue[i]+1]*dblue)/titblue
            #meanred_obs = np.sum(yobs[inired[i]:ifired[i]+1]*dred)/titred
            
            new_yobs_blue = yobs[iniblue[i]:ifiblue[i]+1][err[iniblue[i]:ifiblue[i]+1] != 999]
            new_yobs_red = yobs[inired[i]:ifired[i]+1][err[inired[i]:ifired[i]+1] != 999]
            
            # new_errobs_blue = err[iniblue[i]:ifiblue[i]+1][err[iniblue[i]:ifiblue[i]+1] != 999]
            # new_errobs_red = err[inired[i]:ifired[i]+1][err[inired[i]:ifired[i]+1] != 999]
            
            new_meanblue_obs = np.nansum(new_yobs_blue*new_dblue)/new_titblue
            new_meanred_obs = np.nansum(new_yobs_red*new_dred)/new_titred
            
            # new_meanblue_obs_err = np.sqrt(np.nansum((new_errobs_blue*new_dblue)**2))/new_titblue
            # new_meanred_obs_err = np.sqrt(np.nansum((new_errobs_red*new_dred)**2))/new_titred
            #print(meanblue_obs,new_meanblue_obs)
            #print(meanblue_obs,meanred_obs)
            selblue.append(np.where((xobs >= blue_sx[i]) & (xobs <= blue_dx[i]))[0])
            selred.append(np.where((xobs >= red_sx[i]) & (xobs <= red_dx[i]))[0])
            selfeat.append(np.where((xobs >= feat_sx[i]) & (xobs <= feat_dx[i]))[0])
        
            #meanblue_obs = np.nanmedian(yobs[selblue[i]])
            #meanred_obs = np.nanmedian(yobs[selred[i]])
            #print(meanblue_obs,meanred_obs)
        
            #m_obs = (meanblue_obs-meanred_obs)/(meanbluex[i]-meanredx[i])
            #q_obs = (meanbluex[i]*meanred_obs-meanredx[i]*meanblue_obs)/(meanbluex[i]-meanredx[i])
            
            m_obs = (new_meanblue_obs-new_meanred_obs)/(meanbluex[i]-meanredx[i])
            #m_err = np.sqrt(new_meanblue_obs_err**2 + new_meanred_obs_err**2)/(meanbluex[i]-meanredx[i])
            
            q_obs = (meanbluex[i]*new_meanred_obs-meanredx[i]*new_meanblue_obs)/(meanbluex[i]-meanredx[i])
            

            # #q_err = np.sqrt((meanbluex[i]*new_meanred_obs_err)**2 + (meanredx[i]*new_meanblue_obs_err)**2)/(meanbluex[i]-meanredx[i])
            # #q_err = np.sqrt((meanbluexprime*new_meanred_obs_err)**2 + (meanredxprime*new_meanblue_obs_err)**2)/(meanbluexprime-meanredxprime)
          
            ycont_obs = m_obs*xobs[selfeat[i]]+q_obs
            
            # #x_centered = xobs[selfeat[i]]-np.mean(xobs[selfeat[i]])
            # #ycont_err = np.sqrt((x_centered*m_err)**2+q_err**2)
            # #print(m_err,q_err)
            # yfeat_obs.append(yobs[selfeat[i]]/ycont_obs)
            #errfeat_obs.append((err[selfeat[i]]/ycont_obs))
            #errfeat_obs.append(np.sqrt((err[selfeat[i]]/ycont_obs)**2+(yobs[selfeat[i]]/ycont_obs**2)**2 * ycont_err**2))
            #print(np.sqrt((err[selfeat[i]]/ycont_obs)**2+(yobs[selfeat[i]]/ycont_obs**2)**2 * ycont_err**2))
            #print(np.sqrt((err[selfeat[i]]/ycont_obs)**2+(yobs[selfeat[i]]/ycont_obs**2)**2 * ycont_err**2))
            # print(yobs[selfeat[i]],ycont_obs)
        
    print(err[selfeat[i]]/ycont_obs)
    return yobs[selfeat[i]]/ycont_obs

def varsmooth_vec(x, y, sig_x, xout=None, oversample=1):
    """    
    Fast and accurate convolution with a Gaussian of variable width.

    This function performs an accurate Fourier convolution of a vector, or the
    columns of an array, with a Gaussian kernel that has a varying or constant
    standard deviation (sigma) per pixel. The convolution is done using fast
    Fourier transform (FFT) and the analytic expression of the Fourier
    transform of the Gaussian function, like in the pPXF method. This allows
    for an accurate convolution even when the Gaussian is severely
    undersampled.

    This function is recommended over standard convolution even when dealing
    with a constant Gaussian width, due to its more accurate handling of
    undersampling issues.

    This function implements Algorithm 1 in `Cappellari (2023)
    <https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C>`_

    Input Parameters
    ----------------

    x : array_like
        Coordinate of every pixel in `y`.
    y : array_like
        Input vector or array of column-spectra.
    sig_x : float or array_like
        Gaussian sigma of every pixel in units of `x`.
        If sigma is constant, `sig_x` can be a scalar. 
        In this case, `x` must be uniformly sampled.
    oversample : float, optional
        Oversampling factor before convolution (default: 1).
    xout : array_like, optional
        Output `x` coordinate used to compute the convolved `y`.

    Output Parameters
    -----------------

    yout : array_like
        Convolved vector or columns of the array `y`.

    """
    assert len(x) == len(y[0,:]), "`x` and `y` must have the same length"

    if np.isscalar(sig_x):
        dx = np.diff(x)
        assert np.all(np.isclose(dx[0], dx)), "`x` must be uniformly spaced, when `sig_x` is a scalar"
        n = len(x)
        sig_max = sig_x*(n - 1)/(x[-1] - x[0])
        y_new = y.T
    else:
        assert len(x) == len(sig_x), "`x` and `sig_x` must have the same length"
        # Stretches spectrum to have equal sigma in the new coordinate
        sig = sig_x/np.gradient(x)
        sig = sig.clip(0.1)   # Clip to >=0.1 pixels
        sig_max = np.max(sig)*oversample
        xs = np.cumsum(sig_max/sig)
        n = int(np.ceil(xs[-1] - xs[0]))
        x_new = np.linspace(xs[0], xs[-1], n)
        #y_new = np.zeros([len(y[:,0]),len(x_new)])
        # for i in range(len(y[:,0])):
        #     y_new[i,:] = interp(x_new, xs, y[i,:].T)
        y_new = vectorized_interp(x_new, xs, y)
        #print(y_new.shape)

    # Convolve spectrum with a Gaussian using analytic FT like pPXF
    npad = 2**int(np.ceil(np.log2(n)))
    ft = np.fft.rfft(y_new, npad,axis = -1)
    w = np.linspace(0, np.pi*sig_max, ft.shape[-1])
    ft_gau = np.exp(-0.5*w**2)
    yout = np.fft.irfft(ft*ft_gau, npad,axis = -1)[:,:n]

    if not np.isscalar(sig_x):
        if xout is not None:
            xs = np.interp(xout, x, xs)  # xs is 1-dim
        #yout = interp(xs, x_new, yout.T)
        yout = vectorized_interp(xs, x_new, yout)


    return yout


def fif_obs(xobs,yobs,err,name_ind,blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx):
    '''
    Computes chi2 for given model and observed spectrum
    --------------------------------------------------
    x : model lambda
    y : model flux
    xobs : observed lambda
    yobs : observed flux
    err : observed error
    zobs : galaxy redshift
    myfeat : index
    k : multiplier for index
    -------------
    RETURN : chi2
    '''
    yfeat_obs = []
    errfeat_obs = []
    selblue = []
    selred = []
    selfeat = []
    meanbluex = (blue_sx+blue_dx)/2
    meanredx = (red_sx+red_dx)/2
    for i in range(len(name_ind)):
        selblue.append(np.where((xobs >= blue_sx[i]) & (xobs <= blue_dx[i]))[0])
        selred.append(np.where((xobs >= red_sx[i]) & (xobs <= red_dx[i]))[0])
        selfeat.append(np.where((xobs >= feat_sx[i]) & (xobs <= feat_dx[i]))[0])
        
        meanblue_obs = np.nanmedian(yobs[selblue[i]])
        meanred_obs = np.nanmedian(yobs[selred[i]])
        
        
        m_obs = (meanblue_obs-meanred_obs)/(meanbluex[i]-meanredx[i])
        q_obs = (meanbluex[i]*meanred_obs-meanredx[i]*meanblue_obs)/(meanbluex[i]-meanredx[i])
        
        ycont_obs = m_obs*xobs[selfeat[i]]+q_obs

        yfeat_obs.append(yobs[selfeat[i]]/ycont_obs)
        errfeat_obs.append(err[selfeat[i]]/ycont_obs)

        

    return yfeat_obs,errfeat_obs,selblue,selred,selfeat,meanbluex,meanredx

def fif_obs_pro(xobs,yobs,err,name_ind,blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx):
    '''
    Computes chi2 for given model and observed spectrum
    --------------------------------------------------
    x : model lambda
    y : model flux
    xobs : observed lambda
    yobs : observed flux
    err : observed error
    zobs : galaxy redshift
    myfeat : index
    k : multiplier for index
    -------------
    RETURN : chi2
    '''
    yfeat_obs = []
    errfeat_obs = []
    selblue = []
    selred = []
    selfeat = []
    meanbluex = (blue_sx+blue_dx)/2
    meanredx = (red_sx+red_dx)/2
    for i in range(len(name_ind)):
        if((xobs[0] < blue_sx[i]) & (xobs[-1] > red_dx[i])):
            iniblue     = np.max(np.where(xobs < blue_sx[i])[0]) + 1
            ifiblue     = np.max(np.where(xobs < blue_dx[i])[0])
            inired     = np.max(np.where(xobs < red_sx[i])[0]) + 1
            ifired     = np.max(np.where(xobs < red_dx[i])[0])

            dblue = (xobs[iniblue:ifiblue+1]-xobs[iniblue-1:ifiblue])*0.5+(xobs[iniblue+1:ifiblue+2]-xobs[iniblue:ifiblue+1])*0.5
            dred = (xobs[inired:ifired+1]-xobs[inired-1:ifired])*0.5+(xobs[inired+1:ifired+2]-xobs[inired:ifired+1])*0.5
            
            #parte blue sx
            if (xobs[iniblue] - (xobs[iniblue]-xobs[iniblue-1])*0.5 > blue_sx[i]):
                iniblue = iniblue - 1
                dblue = np.append(abs(xobs[iniblue] + (xobs[iniblue+1] - xobs[iniblue])*0.5 - blue_sx[i]),dblue)
            elif (xobs[iniblue] - (xobs[iniblue]-xobs[iniblue-1])*0.5 < blue_sx[i]):
                dblue[0] = dblue[0] - (blue_sx[i] - (xobs[iniblue] - (xobs[iniblue] - xobs[iniblue-1])*0.5))

            #parte blue dx
            if (xobs[ifiblue] + (xobs[ifiblue+1]-xobs[ifiblue])*0.5 < blue_dx[i]):
                ifiblue = ifiblue + 1
                dblue = np.append(dblue,abs((xobs[ifiblue] - (xobs[ifiblue] - xobs[ifiblue - 1])*0.5) - blue_dx[i]))
            elif (xobs[ifiblue] + (xobs[ifiblue + 1]-xobs[ifiblue])*0.5 > blue_dx[i]):
                dblue[-1] = dblue[-1] - abs(xobs[ifiblue] + (xobs[ifiblue + 1] - xobs[ifiblue])*0.5 - blue_dx[i])

            #parte red sx
            if (xobs[inired] - (xobs[inired]-xobs[inired-1])*0.5 > red_sx[i]):
                inired = inired - 1
                dred = np.append(abs(xobs[inired] + (xobs[inired + 1] - xobs[inired])*0.5 - red_sx[i]),dred)
            elif (xobs[inired] - (xobs[inired]-xobs[inired-1])*0.5 < red_sx[i]):
                dred[0] = dred[0] - abs(xobs[inired] - (xobs[inired] - xobs[inired-1])*0.5 - red_sx[i])

            #parte red dx
            if (xobs[ifired] + (xobs[ifired+1]-xobs[ifired])*0.5 < red_dx[i]):
                ifired = ifired + 1
                dred = np.append(dred,abs(xobs[ifired] - (xobs[ifired + 1] - xobs[ifired])*0.5 - red_dx[i]))
            elif (xobs[ifired] + (xobs[ifired + 1]-xobs[ifired])*0.5 > red_dx[i]):
                dred[-1] = dred[-1] - abs(xobs[ifired] + (xobs[ifired + 1] - xobs[ifired])*0.5 - red_dx[i])
        
            
            titblue = sum(dblue)
            titred = sum(dred)
            
            meanblue_obs = np.sum(yobs[iniblue:ifiblue+1]*dblue)/titblue
            meanred_obs = np.sum(yobs[inired:ifired+1]*dred)/titred
            #print(meanblue_obs,meanred_obs)
            selblue.append(np.where((xobs >= blue_sx[i]) & (xobs <= blue_dx[i]))[0])
            selred.append(np.where((xobs >= red_sx[i]) & (xobs <= red_dx[i]))[0])
            selfeat.append(np.where((xobs >= feat_sx[i]) & (xobs <= feat_dx[i]))[0])
        
            #meanblue_obs = np.nanmedian(yobs[selblue[i]])
            #meanred_obs = np.nanmedian(yobs[selred[i]])
            #print(meanblue_obs,meanred_obs)
        
            m_obs = (meanblue_obs-meanred_obs)/(meanbluex[i]-meanredx[i])
            q_obs = (meanbluex[i]*meanred_obs-meanredx[i]*meanblue_obs)/(meanbluex[i]-meanredx[i])
        
            ycont_obs = m_obs*xobs[selfeat[i]]+q_obs

            yfeat_obs.append(yobs[selfeat[i]]/ycont_obs)
            errfeat_obs.append(err[selfeat[i]]/ycont_obs)

        

    return yfeat_obs,errfeat_obs,selblue,selred,selfeat,meanbluex,meanredx



def calcchi2_fif_all_pro(x,y,yfeat_obs,err_obs,name_ind,selblue,selfeat,selred,meanbluex,meanredx,blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx):
    '''
    Computes chi2 for given model and observed spectrum
    --------------------------------------------------
    x : model lambda
    y : model flux
    xobs : observed lambda
    yobs : observed flux
    err : observed error
    zobs : galaxy redshift
    myfeat : index
    k : multiplier for index
    -------------
    RETURN : chi2
    '''
    chi2 = 0
    for i in range(len(name_ind)):
        if((x[0] < blue_sx[i]) & (x[-1] > red_dx[i])):
            iniblue     = np.max(np.where(x < blue_sx[i])[0]) + 1
            ifiblue     = np.max(np.where(x < blue_dx[i])[0])
            inired     = np.max(np.where(x < red_sx[i])[0]) + 1
            ifired     = np.max(np.where(x < red_dx[i])[0])

            dblue = (x[iniblue:ifiblue+1]-x[iniblue-1:ifiblue])*0.5+(x[iniblue+1:ifiblue+2]-x[iniblue:ifiblue+1])*0.5
            dred = (x[inired:ifired+1]-x[inired-1:ifired])*0.5+(x[inired+1:ifired+2]-x[inired:ifired+1])*0.5
            
            #parte blue sx
            if (x[iniblue] - (x[iniblue]-x[iniblue-1])*0.5 > blue_sx[i]):
                iniblue = iniblue - 1
                dblue = np.append(abs(x[iniblue] + (x[iniblue+1] - x[iniblue])*0.5 - blue_sx[i]),dblue)
            elif (x[iniblue] - (x[iniblue]-x[iniblue-1])*0.5 < blue_sx[i]):
                dblue[0] = dblue[0] - (blue_sx[i] - (x[iniblue] - (x[iniblue] - x[iniblue-1])*0.5))

            #parte blue dx
            if (x[ifiblue] + (x[ifiblue+1]-x[ifiblue])*0.5 < blue_dx[i]):
                ifiblue = ifiblue + 1
                dblue = np.append(dblue,abs((x[ifiblue] - (x[ifiblue] - x[ifiblue - 1])*0.5) - blue_dx[i]))
            elif (x[ifiblue] + (x[ifiblue + 1]-x[ifiblue])*0.5 > blue_dx[i]):
                dblue[-1] = dblue[-1] - abs(x[ifiblue] + (x[ifiblue + 1] - x[ifiblue])*0.5 - blue_dx[i])

            #parte red sx
            if (x[inired] - (x[inired]-x[inired-1])*0.5 > red_sx[i]):
                inired = inired - 1
                dred = np.append(abs(x[inired] + (x[inired + 1] - x[inired])*0.5 - red_sx[i]),dred)
            elif (x[inired] - (x[inired]-x[inired-1])*0.5 < red_sx[i]):
                dred[0] = dred[0] - abs(x[inired] - (x[inired] - x[inired-1])*0.5 - red_sx[i])

            #parte red dx
            if (x[ifired] + (x[ifired+1]-x[ifired])*0.5 < red_dx[i]):
                ifired = ifired + 1
                dred = np.append(dred,abs(x[ifired] - (x[ifired + 1] - x[ifired])*0.5 - red_dx[i]))
            elif (x[ifired] + (x[ifired + 1]-x[ifired])*0.5 > red_dx[i]):
                dred[-1] = dred[-1] - abs(x[ifired] + (x[ifired + 1] - x[ifired])*0.5 - red_dx[i])
        
            
            titblue = sum(dblue)
            titred = sum(dred)
            
            meanblue = np.sum(y[:,iniblue:ifiblue+1]*dblue,axis = 1)/titblue
            meanred = np.sum(y[:,inired:ifired+1]*dred, axis = 1)/titred
        
        
            m = (meanblue-meanred)/(meanbluex[i]-meanredx[i])
            q = (meanbluex[i]*meanred-meanredx[i]*meanblue)/(meanbluex[i]-meanredx[i])
            ycont = m[:,None]*x[selfeat[i]][None,:]+q[:,None]
        
            yfeat = y[:,selfeat[i]]/ycont
        
            #chi2 = chi2 + sum(((yfeat_obs[i]-yfeat)/err_obs[i])**2-np.log(1/err_obs[i]**2))#
            var = 1/(err_obs[i])**2
            #print((var*(yfeat_obs[i]-yfeat)**2 - np.log(var) + np.log(2.*np.pi)).shape)
            # if((name_ind[i] == 'Fe5270') | (name_ind[i] == 'Fe5335')):
            #     chi2 = chi2 + np.nansum(var*(yfeat_obs[i]-yfeat)**2 - np.log(var) + np.log(2.*np.pi),axis = 1)*0
            # else:
            chi2 = chi2 + np.nansum(var*(yfeat_obs[i]-yfeat)**2 - np.log(var) + np.log(2.*np.pi),axis = 1)#*(32.5/len(yfeat_obs[i]))#
            #print(chi2.shape)
            #chi2 = chi2 + sum(var*(yfeat_obs[i]-yfeat)**2)#
        
    #chi2r = chi2/(len(y1[sel])-2)

    return chi2

def fif_obs_mc_error(xobs,yobs,err,name_ind,blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx):
    '''
    Computes chi2 for given model and observed spectrum
    --------------------------------------------------
    x : model lambda
    y : model flux
    xobs : observed lambda
    yobs : observed flux
    err : observed error
    zobs : galaxy redshift
    myfeat : index
    k : multiplier for index
    -------------
    RETURN : chi2
    '''
    yfeat_obs = []
    errfeat_obs = []
    selblue = []
    selred = []
    selfeat = []
    meanbluex = (blue_sx+blue_dx)/2
    meanredx = (red_sx+red_dx)/2
    
    for i in range(len(name_ind)):
        selblue.append(np.where((xobs >= blue_sx[i]) & (xobs <= blue_dx[i]))[0])
        selred.append(np.where((xobs >= red_sx[i]) & (xobs <= red_dx[i]))[0])
        selfeat.append(np.where((xobs >= feat_sx[i]) & (xobs <= feat_dx[i]))[0])
        meanblue_obs = np.median(yobs[selblue[i],:],axis = 0)
        meanred_obs = np.median(yobs[selred[i],:],axis = 0)
        m_obs = (meanblue_obs-meanred_obs)/(meanbluex[i]-meanredx[i])
        q_obs = (meanbluex[i]*meanred_obs-meanredx[i]*meanblue_obs)/(meanbluex[i]-meanredx[i])
        ycont_obs = m_obs*xobs[selfeat[i],None]+q_obs
        yfeat_obs.append(yobs[selfeat[i],:]/ycont_obs)
        errfeat_obs.append((np.percentile(yobs[selfeat[i],:]/ycont_obs,84,axis = 1)-np.percentile(yobs[selfeat[i],:]/ycont_obs,16,axis = 1))/2)
    

    return errfeat_obs


def calcchi2fif_already_alsoobs(x,y,yfeat_obs,err_obs,name_ind,selblue,selfeat,selred,meanbluex,meanredx):
    '''
    Computes chi2 for given model and observed spectrum
    --------------------------------------------------
    x : model lambda
    y : model flux
    xobs : observed lambda
    yobs : observed flux
    err : observed error
    zobs : galaxy redshift
    myfeat : index
    k : multiplier for index
    -------------
    RETURN : chi2
    '''
    chi2 = 0
    for i in range(len(name_ind)):
        meanblue = np.median(y[selblue[i]])
        meanred = np.median(y[selred[i]])
        
        m = (meanblue-meanred)/(meanbluex[i]-meanredx[i])
        q = (meanbluex[i]*meanred-meanredx[i]*meanblue)/(meanbluex[i]-meanredx[i])
        
        ycont = m*x[selfeat[i]]+q
        
        yfeat = y[selfeat[i]]/ycont
        
        #chi2 = chi2 + sum(((yfeat_obs[i]-yfeat)/err_obs[i])**2-np.log(1/err_obs[i]**2))#
        var = 1/(err_obs[i])**2
        chi2 = chi2 + np.nansum(var*(yfeat_obs[i]-yfeat)**2 - np.log(var) + np.log(2.*np.pi))#
        #chi2 = chi2 + sum(var*(yfeat_obs[i]-yfeat)**2)#
        
    #chi2r = chi2/(len(y1[sel])-2)

    return chi2


def calcchi2_fif_all(x,y,yfeat_obs,err_obs,name_ind,selblue,selfeat,selred,meanbluex,meanredx):
    '''
    Computes chi2 for given model and observed spectrum
    --------------------------------------------------
    x : model lambda
    y : model flux
    xobs : observed lambda
    yobs : observed flux
    err : observed error
    zobs : galaxy redshift
    myfeat : index
    k : multiplier for index
    -------------
    RETURN : chi2
    '''
    chi2 = 0
    for i in range(len(name_ind)):
        meanblue = np.median(y[:,selblue[i]],axis = 1)
        meanred = np.median(y[:,selred[i]],axis = 1)
        
        m = (meanblue-meanred)/(meanbluex[i]-meanredx[i])
        q = (meanbluex[i]*meanred-meanredx[i]*meanblue)/(meanbluex[i]-meanredx[i])
        ycont = m[:,None]*x[selfeat[i]][None,:]+q[:,None]
        
        yfeat = y[:,selfeat[i]]/ycont
        
        #chi2 = chi2 + sum(((yfeat_obs[i]-yfeat)/err_obs[i])**2-np.log(1/err_obs[i]**2))#
        var = 1/(err_obs[i])**2
        #print((var*(yfeat_obs[i]-yfeat)**2 - np.log(var) + np.log(2.*np.pi)).shape)
        chi2 = chi2 + np.nansum(var*(yfeat_obs[i]-yfeat)**2 - np.log(var) + np.log(2.*np.pi),axis = 1)#
        #print(chi2.shape)
        #chi2 = chi2 + sum(var*(yfeat_obs[i]-yfeat)**2)#
        
    #chi2r = chi2/(len(y1[sel])-2)

    return chi2

def calcchi2_ind(x,y,ind_obs,err_obs,name_ind,blue_sx, blue_dx, feat_sx, feat_dx, red_sx, red_dx):
    '''
    Computes chi2 for given model and observed spectrum
    --------------------------------------------------
    x : model lambda
    y : model flux
    xobs : observed lambda
    yobs : observed flux
    err : observed error
    zobs : galaxy redshift
    myfeat : index
    k : multiplier for index
    -------------
    RETURN : chi2
    '''
    chi2 = 0
    ind_mod = calcindex(x, y, blue_sx, blue_dx, feat_sx, feat_dx, red_sx, red_dx, name_ind)
        
        #chi2 = chi2 + sum(((yfeat_obs[i]-yfeat)/err_obs[i])**2-np.log(1/err_obs[i]**2))#
    var = 1/(err_obs)**2
    #print((ind_obs[i]-ind_mod[i])**2)
    #chi2 = chi2 + sum(var*(ind_obs[i]-ind_mod[i])**2 - np.log(var) + np.log(2.*np.pi))#
    chi2 = chi2 + np.nansum(var*(ind_obs-ind_mod)**2 - np.log(var) + np.log(2.*np.pi))
        #chi2 = chi2 + sum(var*(yfeat_obs[i]-yfeat)**2)#
        
    #chi2r = chi2/(len(y1[sel])-2)

    return chi2

def calcchi2_ind_vec(x,y,ind_obs,err_obs,name_ind,blue_sx, blue_dx, feat_sx, feat_dx, red_sx, red_dx):
    '''
    Computes chi2 for given model and observed spectrum
    --------------------------------------------------
    x : model lambda
    y : model flux
    xobs : observed lambda
    yobs : observed flux
    err : observed error
    zobs : galaxy redshift
    myfeat : index
    k : multiplier for index
    -------------
    RETURN : chi2
    '''
    ind_mod = calcindex_all(x, y, blue_sx, blue_dx, feat_sx, feat_dx, red_sx, red_dx, name_ind)

        #chi2 = chi2 + sum(((yfeat_obs[i]-yfeat)/err_obs[i])**2-np.log(1/err_obs[i]**2))#
    var = 1/(err_obs)**2
    #print((ind_obs[i]-ind_mod[i])**2)
    #chi2 = chi2 + sum(var*(ind_obs[i]-ind_mod[i])**2 - np.log(var) + np.log(2.*np.pi))#
    #print(ind_obs.shape,ind_mod.shape)
    chi2 = np.nansum(var*(ind_obs-ind_mod)**2 - np.log(var) + np.log(2.*np.pi),axis = 0)

        #chi2 = chi2 + sum(var*(yfeat_obs[i]-yfeat)**2)#
        
    #chi2r = chi2/(len(y1[sel])-2)

    return chi2

def interpmodel(newmetal, newage, metal, age, mymodels):
    '''
    Compute logarithmically in age and metallicity a model with fixed age and metallicity
    ------------------------------------------
    newmetal : new metallicity (log10)
    newage : new age [Gyr]
    metal : metallicity grid (log10)
    age : age grid [Gyr]
    mymodels : models
    --------------------------------------------------------------
    RETURN : lambda and flux of model with new age and metallicity
    '''
    imet = np.where(newmetal >= metal)[0][-1]
    met = [metal[imet],metal[imet+1]]
    iage = np.where(newage >= age)[0][-1]
    aged = [age[iage],age[iage+1]]
    #lower left of the square
    y1 = mymodels[iage,imet,:]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,:]
    #lower right of the square
    y3 = mymodels[iage+1,imet,:]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,:]
    #vertical interpolation on the square, left side
    ymet1 = (newmetal-met[0])*(y2-y1)/(met[-1]-met[0])+y1
    
    #ymet1 = y2**((newmetal-met[0])/(met[-1]-met[0]))*y1**(1-(newmetal-met[0])/(met[1]-met[0]))
    #vertical interpolation on the square, right side
    ymet2 = (newmetal-met[0])*(y4-y3)/(met[-1]-met[0])+y3
    #ymet2 = y4**((newmetal-met[0])/(met[-1]-met[0]))*y3**(1-(newmetal-met[0])/(met[1]-met[0]))
    #final interpolation (interior point of the square)
    y = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    #y = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    return y

def interpmodel_tau_all(newmetal, newage, newtau, zh, age, tau, mymodels):
    '''
    Compute logarithmically in age and metallicity a model with fixed age and metallicity
    ------------------------------------------
    newmetal : new metallicity (log10)
    newage : new age [Gyr]
    metal : metallicity grid (log10)
    age : age grid [Gyr]
    mymodels : models
    --------------------------------------------------------------
    RETURN : lambda and flux of model with new age and metallicity
    '''
    iage = np.searchsorted(age,newage,side = 'right')-1
    imet = np.searchsorted(zh,newmetal,side = 'right')-1
    itau = np.searchsorted(tau,newtau,side = 'right')-1

    aged = [age[iage],age[iage+1]]
    met = [zh[imet],zh[imet+1]]
    taud = [tau[itau],tau[itau+1]]
    #three parameters, need a cube
    #first part of the cube
    #lower left of the square
    y1 = mymodels[iage,imet,itau]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,itau]

    y3 = mymodels[iage+1,imet,itau]

    y4 = mymodels[iage+1,imet+1,itau]
    #lower right of the square

    ymet1 = (newmetal-met[0])[:,None]*(y2-y1)/(met[-1]-met[0])[:,None]+y1
    ymet2 = (newmetal-met[0])[:,None]*(y4-y3)/(met[-1]-met[0])[:,None]+y3

    y_firstsquare = (newage-aged[0])[:,None]*(ymet2-ymet1)/(aged[-1]-aged[0])[:,None]+ymet1
    
    #second part of the cube
    y1 = mymodels[iage,imet,itau+1]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,itau+1]
    #lower right of the square
    y3 = mymodels[iage+1,imet,itau+1]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,itau+1]
    #vertical interpolation on the square, left side
    ymet1 = (newmetal-met[0])[:,None]*(y2-y1)/(met[-1]-met[0])[:,None]+y1
    #ymet1 = y2**((newmetal-met[0])/(met[-1]-met[0]))*y1**(1-(newmetal-met[0])/(met[1]-met[0]))
    #vertical interpolation on the square, right side
    ymet2 = (newmetal-met[0])[:,None]*(y4-y3)/(met[-1]-met[0])[:,None]+y3
    #ymet2 = y4**((newmetal-met[0])/(met[-1]-met[0]))*y3**(1-(newmetal-met[0])/(met[1]-met[0]))
    #final interpolation (interior point of the square), second part of the cube
    y_secondsquare = (newage-aged[0])[:,None]*(ymet2-ymet1)/(aged[-1]-aged[0])[:,None]+ymet1
    #y = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    y_cube = (newtau-taud[0])[:,None]*(y_secondsquare-y_firstsquare)/(taud[-1]-taud[0])[:,None]+y_firstsquare
    return y_cube

def interpmodel_4d_all(newmetal, newage, newtau, newdust, zh, age, tau, dust, mymodels):
    '''
    Compute logarithmically in age and metallicity a model with fixed age and metallicity
    ------------------------------------------
    newmetal : new metallicity (log10)
    newage : new age [Gyr]
    newtau: new tau 
    metal : metallicity grid (log10)
    age : age grid [Gyr]
    tau : tau grid
    mymodels : models
    --------------------------------------------------------------
    RETURN : lambda and flux of model with new age and metallicity
    '''
    
    iage = np.searchsorted(age,newage,side = 'right')-1
    imet = np.searchsorted(zh,newmetal,side = 'right')-1
    itau = np.searchsorted(tau,newtau,side = 'right')-1
    idust = np.searchsorted(dust,newdust,side = 'right')-1
    aged = [age[iage],age[iage+1]]
    met = [zh[imet],zh[imet+1]]
    taud = [tau[itau],tau[itau+1]]
    dustd = [dust[idust],dust[idust+1]]
    
    #three parameters, need a cube
    #lower left of the square
    y1 = mymodels[iage,imet,itau,idust,:]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,itau,idust,:]
    #lower right of the square
    y3 = mymodels[iage+1,imet,itau,idust,:]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,itau,idust,:]
    #vertical interpolation on the square, left side
    
    ymet1 = (newmetal-met[0])[:,None]*(y2-y1)/(met[-1]-met[0])[:,None]+y1
    ymet2 = (newmetal-met[0])[:,None]*(y4-y3)/(met[-1]-met[0])[:,None]+y3

    y_firstsquare = (newage-aged[0])[:,None]*(ymet2-ymet1)/(aged[-1]-aged[0])[:,None]+ymet1
    
    
    #second part of the cube
    y1 = mymodels[iage,imet,itau+1,idust,:]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,itau+1,idust,:]
    #lower right of the square
    y3 = mymodels[iage+1,imet,itau+1,idust,:]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,itau+1,idust,:]
    #vertical interpolation on the square, left side
    
    ymet1 = (newmetal-met[0])[:,None]*(y2-y1)/(met[-1]-met[0])[:,None]+y1
    ymet2 = (newmetal-met[0])[:,None]*(y4-y3)/(met[-1]-met[0])[:,None]+y3

    y_secondsquare = (newage-aged[0])[:,None]*(ymet2-ymet1)/(aged[-1]-aged[0])[:,None]+ymet1
    
    
    y_firstcube = (newtau-taud[0])[:,None]*(y_secondsquare-y_firstsquare)/(taud[-1]-taud[0])[:,None]+y_firstsquare
    
    y1 = mymodels[iage,imet,itau,idust+1,:]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,itau,idust+1,:]
    #lower right of the square
    y3 = mymodels[iage+1,imet,itau,idust+1,:]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,itau,idust+1,:]
    #vertical interpolation on the square, left side
    ymet1 = (newmetal-met[0])[:,None]*(y2-y1)/(met[-1]-met[0])[:,None]+y1
    ymet2 = (newmetal-met[0])[:,None]*(y4-y3)/(met[-1]-met[0])[:,None]+y3

    y_firstsquare = (newage-aged[0])[:,None]*(ymet2-ymet1)/(aged[-1]-aged[0])[:,None]+ymet1
    
    
    #second part of the cube
    y1 = mymodels[iage,imet,itau+1,idust+1,:]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,itau+1,idust+1,:]
    #lower right of the square
    y3 = mymodels[iage+1,imet,itau+1,idust+1,:]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,itau+1,idust+1,:]
    #vertical interpolation on the square, left side
    ymet1 = (newmetal-met[0])[:,None]*(y2-y1)/(met[-1]-met[0])[:,None]+y1
    ymet2 = (newmetal-met[0])[:,None]*(y4-y3)/(met[-1]-met[0])[:,None]+y3

    y_secondsquare = (newage-aged[0])[:,None]*(ymet2-ymet1)/(aged[-1]-aged[0])[:,None]+ymet1
    #y = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    
    y_secondcube = (newtau-taud[0])[:,None]*(y_secondsquare-y_firstsquare)/(taud[-1]-taud[0])[:,None]+y_firstsquare
    
    y_4d = (newdust-dustd[0])[:,None]*(y_secondcube-y_firstcube)/(dustd[-1]-dustd[0])[:,None]+y_firstcube
    return y_4d




def interpmodel_4d(newmetal, newage, newtau, newdust, metal, age, tau, dust, mymodels):
    '''
    Compute logarithmically in age and metallicity a model with fixed age and metallicity
    ------------------------------------------
    newmetal : new metallicity (log10)
    newage : new age [Gyr]
    newtau: new tau 
    metal : metallicity grid (log10)
    age : age grid [Gyr]
    tau : tau grid
    mymodels : models
    --------------------------------------------------------------
    RETURN : lambda and flux of model with new age and metallicity
    '''
    imet = np.where(newmetal >= metal)[0][-1]
    met = [metal[imet],metal[imet+1]]
    iage = np.where(newage >= age)[0][-1]
    aged = [age[iage],age[iage+1]]
    itau = np.where(newtau >= tau)[0][-1]
    taud = [tau[itau], tau[itau +1]]
    idust = np.where(newdust >= dust)[0][-1]
    dustd = [dust[idust], dust[idust +1]]
    
    
    #three parameters, need a cube
    #lower left of the square
    y1 = mymodels[iage,imet,itau,idust,:]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,itau,idust,:]
    #lower right of the square
    y3 = mymodels[iage+1,imet,itau,idust,:]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,itau,idust,:]
    #vertical interpolation on the square, left side
    ymet1 = (newmetal-met[0])*(y2-y1)/(met[-1]-met[0])+y1
    
    #ymet1 = y2**((newmetal-met[0])/(met[-1]-met[0]))*y1**(1-(newmetal-met[0])/(met[1]-met[0]))
    #vertical interpolation on the square, right side
    ymet2 = (newmetal-met[0])*(y4-y3)/(met[-1]-met[0])+y3
    #ymet2 = y4**((newmetal-met[0])/(met[-1]-met[0]))*y3**(1-(newmetal-met[0])/(met[1]-met[0]))
    #final interpolation (interior point of the square), first part of the cube
    y_firstsquare = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    
    
    #second part of the cube
    y1 = mymodels[iage,imet,itau+1,idust,:]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,itau+1,idust,:]
    #lower right of the square
    y3 = mymodels[iage+1,imet,itau+1,idust,:]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,itau+1,idust,:]
    #vertical interpolation on the square, left side
    ymet1 = (newmetal-met[0])*(y2-y1)/(met[-1]-met[0])+y1
    
    #ymet1 = y2**((newmetal-met[0])/(met[-1]-met[0]))*y1**(1-(newmetal-met[0])/(met[1]-met[0]))
    #vertical interpolation on the square, right side
    ymet2 = (newmetal-met[0])*(y4-y3)/(met[-1]-met[0])+y3
    #ymet2 = y4**((newmetal-met[0])/(met[-1]-met[0]))*y3**(1-(newmetal-met[0])/(met[1]-met[0]))
    #final interpolation (interior point of the square), second part of the cube
    y_secondsquare = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    #y = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    
    y_firstcube = (newtau-taud[0])*(y_secondsquare-y_firstsquare)/(taud[-1]-taud[0])+y_firstsquare
    
    y1 = mymodels[iage,imet,itau,idust+1,:]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,itau,idust+1,:]
    #lower right of the square
    y3 = mymodels[iage+1,imet,itau,idust+1,:]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,itau,idust+1,:]
    #vertical interpolation on the square, left side
    ymet1 = (newmetal-met[0])*(y2-y1)/(met[-1]-met[0])+y1
    
    #ymet1 = y2**((newmetal-met[0])/(met[-1]-met[0]))*y1**(1-(newmetal-met[0])/(met[1]-met[0]))
    #vertical interpolation on the square, right side
    ymet2 = (newmetal-met[0])*(y4-y3)/(met[-1]-met[0])+y3
    #ymet2 = y4**((newmetal-met[0])/(met[-1]-met[0]))*y3**(1-(newmetal-met[0])/(met[1]-met[0]))
    #final interpolation (interior point of the square), first part of the cube
    y_firstsquare = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    
    
    #second part of the cube
    y1 = mymodels[iage,imet,itau+1,idust+1,:]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,itau+1,idust+1,:]
    #lower right of the square
    y3 = mymodels[iage+1,imet,itau+1,idust+1,:]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,itau+1,idust+1,:]
    #vertical interpolation on the square, left side
    ymet1 = (newmetal-met[0])*(y2-y1)/(met[-1]-met[0])+y1
    
    #ymet1 = y2**((newmetal-met[0])/(met[-1]-met[0]))*y1**(1-(newmetal-met[0])/(met[1]-met[0]))
    #vertical interpolation on the square, right side
    ymet2 = (newmetal-met[0])*(y4-y3)/(met[-1]-met[0])+y3
    #ymet2 = y4**((newmetal-met[0])/(met[-1]-met[0]))*y3**(1-(newmetal-met[0])/(met[1]-met[0]))
    #final interpolation (interior point of the square), first part of the cube
    y_secondsquare = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    #y = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    
    y_secondcube = (newtau-taud[0])*(y_secondsquare-y_firstsquare)/(taud[-1]-taud[0])+y_firstsquare
    
    y_4d = (newdust-dustd[0])*(y_secondcube-y_firstcube)/(dustd[-1]-dustd[0])+y_firstcube
    return y_4d




def interpmodel_tau(newmetal, newage, newtau, metal, age, tau, mymodels):
    '''
    Compute logarithmically in age and metallicity a model with fixed age and metallicity
    ------------------------------------------
    newmetal : new metallicity (log10)
    newage : new age [Gyr]
    newimf: new imf 
    metal : metallicity grid (log10)
    age : age grid [Gyr]
    imf : imf grid
    mymodels : models
    --------------------------------------------------------------
    RETURN : lambda and flux of model with new age and metallicity
    '''
    imet = np.where(newmetal >= metal)[0][-1]
    met = [metal[imet],metal[imet+1]]
    iage = np.where(newage >= age)[0][-1]
    aged = [age[iage],age[iage+1]]
    itau = np.where(newtau >= tau)[0][-1]
    taud = [tau[itau], tau[itau +1]]
    
    #three parameters, need a cube
    #lower left of the square
    y1 = mymodels[iage,imet,itau,:]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,itau,:]
    #lower right of the square
    y3 = mymodels[iage+1,imet,itau,:]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,itau,:]
    #vertical interpolation on the square, left side
    ymet1 = (newmetal-met[0])*(y2-y1)/(met[-1]-met[0])+y1
    
    #ymet1 = y2**((newmetal-met[0])/(met[-1]-met[0]))*y1**(1-(newmetal-met[0])/(met[1]-met[0]))
    #vertical interpolation on the square, right side
    ymet2 = (newmetal-met[0])*(y4-y3)/(met[-1]-met[0])+y3
    #ymet2 = y4**((newmetal-met[0])/(met[-1]-met[0]))*y3**(1-(newmetal-met[0])/(met[1]-met[0]))
    #final interpolation (interior point of the square), first part of the cube
    y_firstsquare = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    
    
    #second part of the cube
    y1 = mymodels[iage,imet,itau+1,:]
    #upper left of the square (highest metallicity)
    y2 = mymodels[iage,imet+1,itau+1,:]
    #lower right of the square
    y3 = mymodels[iage+1,imet,itau+1,:]
    #upper right of the square
    y4 = mymodels[iage+1,imet+1,itau+1,:]
    #vertical interpolation on the square, left side
    ymet1 = (newmetal-met[0])*(y2-y1)/(met[-1]-met[0])+y1
    
    #ymet1 = y2**((newmetal-met[0])/(met[-1]-met[0]))*y1**(1-(newmetal-met[0])/(met[1]-met[0]))
    #vertical interpolation on the square, right side
    ymet2 = (newmetal-met[0])*(y4-y3)/(met[-1]-met[0])+y3
    #ymet2 = y4**((newmetal-met[0])/(met[-1]-met[0]))*y3**(1-(newmetal-met[0])/(met[1]-met[0]))
    #final interpolation (interior point of the square), second part of the cube
    y_secondsquare = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    #y = (newage-aged[0])*(ymet2-ymet1)/(aged[-1]-aged[0])+ymet1
    y_cube = (newtau-taud[0])*(y_secondsquare-y_firstsquare)/(taud[-1]-taud[0])+y_firstsquare
    return y_cube

def gaussian_filter1d(spec, sig):
    """
    Convolve a spectrum by a Gaussian with different sigma for every pixel.
    If all sigma are the same this routine produces the same output as
    scipy.ndimage.gaussian_filter1d, except for the border treatment.
    Here the first/last p pixels are filled with zeros.
    When creating a template library for SDSS data, this implementation
    is 60x faster than a naive for loop over pixels.

    :param spec: vector with the spectrum to convolve
    :param sig: vector of sigma values (in pixels) for every pixel
    :return: spec convolved with a Gaussian with dispersion sig

    """
    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(np.ceil(np.max(3*sig)))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n-m+j+1]

    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= np.sum(gau, 0)[None, :]  # Normalize kernel

    conv_spectrum = np.sum(a*gau, 0)

    return conv_spectrum

def gaussian_filter1d_vec(spec, sig):
    """
    Convolve a spectrum by a Gaussian with different sigma for every pixel.
    If all sigma are the same this routine produces the same output as
    scipy.ndimage.gaussian_filter1d, except for the border treatment.
    Here the first/last p pixels are filled with zeros.
    When creating a template library for SDSS data, this implementation
    is 60x faster than a naive for loop over pixels.

    :param spec: vector with the spectrum to convolve
    :param sig: vector of sigma values (in pixels) for every pixel
    :return: spec convolved with a Gaussian with dispersion sig

    """
    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(np.ceil(np.max(3*sig)))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    n = spec[0,:].size
    a = np.zeros((len(spec[:,0]),m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[:,j, p:-p] = spec[:,j:n-m+j+1]

    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= np.sum(gau, 0)[None, :]  # Normalize kernel

    conv_spectrum = np.sum(a*gau, 1)

    return conv_spectrum

def calcindex(ll,flux,blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx,name_ind):
    c = 2.998e+18
    output = np.zeros([len(name_ind)])
    for k in range(len(name_ind)):
        if((ll[0] < blue_sx[k]) & (ll[-1] > red_dx[k])):
            iniblue     = np.max(np.where(ll < blue_sx[k])[0]) + 1
            ifiblue     = np.max(np.where(ll < blue_dx[k])[0])
            inired     = np.max(np.where(ll < red_sx[k])[0]) + 1
            ifired     = np.max(np.where(ll < red_dx[k])[0])

            dblue = (ll[iniblue:ifiblue+1]-ll[iniblue-1:ifiblue])*0.5+(ll[iniblue+1:ifiblue+2]-ll[iniblue:ifiblue+1])*0.5
            dred = (ll[inired:ifired+1]-ll[inired-1:ifired])*0.5+(ll[inired+1:ifired+2]-ll[inired:ifired+1])*0.5
            
            #parte blue sx
            if (ll[iniblue] - (ll[iniblue]-ll[iniblue-1])*0.5 > blue_sx[k]):
                iniblue = iniblue - 1
                dblue = np.append(abs(ll[iniblue] + (ll[iniblue+1] - ll[iniblue])*0.5 - blue_sx[k]),dblue)
            elif (ll[iniblue] - (ll[iniblue]-ll[iniblue-1])*0.5 < blue_sx[k]):
                dblue[0] = dblue[0] - (blue_sx[k] - (ll[iniblue] - (ll[iniblue] - ll[iniblue-1])*0.5))

            #parte blue dx
            if (ll[ifiblue] + (ll[ifiblue+1]-ll[ifiblue])*0.5 < blue_dx[k]):
                ifiblue = ifiblue + 1
                dblue = np.append(dblue,abs((ll[ifiblue] - (ll[ifiblue] - ll[ifiblue - 1])*0.5) - blue_dx[k]))
            elif (ll[ifiblue] + (ll[ifiblue + 1]-ll[ifiblue])*0.5 > blue_dx[k]):
                dblue[-1] = dblue[-1] - abs(ll[ifiblue] + (ll[ifiblue + 1] - ll[ifiblue])*0.5 - blue_dx[k])

            #parte red sx
            if (ll[inired] - (ll[inired]-ll[inired-1])*0.5 > red_sx[k]):
                inired = inired - 1
                dred = np.append(abs(ll[inired] + (ll[inired + 1] - ll[inired])*0.5 - red_sx[k]),dred)
            elif (ll[inired] - (ll[inired]-ll[inired-1])*0.5 < red_sx[k]):
                dred[0] = dred[0] - abs(ll[inired] - (ll[inired] - ll[inired-1])*0.5 - red_sx[k])

            #parte red dx
            if (ll[ifired] + (ll[ifired+1]-ll[ifired])*0.5 < red_dx[k]):
                ifired = ifired + 1
                dred = np.append(dred,abs(ll[ifired] - (ll[ifired + 1] - ll[ifired])*0.5 - red_dx[k]))
            elif (ll[ifired] + (ll[ifired + 1]-ll[ifired])*0.5 > red_dx[k]):
                dred[-1] = dred[-1] - abs(ll[ifired] + (ll[ifired + 1] - ll[ifired])*0.5 - red_dx[k])
        
            
            titblue = sum(dblue)
            titred = sum(dred)
            
            intblue = sum(flux[iniblue:ifiblue+1]*dblue)/titblue
            intred = sum(flux[inired:ifired+1]*dred)/titred
            
            xrm = (red_dx[k] + red_sx[k])/2
            xbm = (blue_dx[k] + blue_sx[k])/2
    
            coef = (intred-intblue)/(xrm-xbm)
            ini     = np.max(np.where(ll <= feat_sx[k])[0]) + 1
            ifi     = np.max(np.where(ll <= feat_dx[k])[0])
    
            d = (ll[ini:ifi+1]-ll[ini-1:ifi])*0.5+(ll[ini+1:ifi+2]-ll[ini:ifi+1])*0.5
    
            #parte feat sx
            if (ll[ini] - (ll[ini]-ll[ini-1])*0.5 > feat_sx[k]):
                ini = ini - 1
                d = np.append(abs(ll[ini] + (ll[ini + 1] - ll[ini])*0.5 - feat_sx[k]),d)
            elif (ll[ini] - (ll[ini]-ll[ini-1])*0.5 < feat_sx[k]):
                d[0] = d[0] - abs(ll[ini] - (ll[ini] - ll[ini-1])*0.5 - feat_sx[k])


            #parte feat dx
            if (ll[ifi] + (ll[ifi+1]-ll[ifi])*0.5 < feat_dx[k]):
                ifi = ifi + 1
                d = np.append(d,abs(ll[ifi] - (ll[ifi + 1] - ll[ifi])*0.5 - feat_dx[k]))
            elif (ll[ifi] + (ll[ifi + 1]-ll[ifi])*0.5 > feat_dx[k]):
                d[-1] = d[-1] - abs(ll[ifi] + (ll[ifi + 1] - ll[ifi])*0.5 - feat_dx[k])
                
            #if (k == 0):
            #    print(len(flux[ini:ifi]))
        
            if ((name_ind[k] == 'D4000') | (name_ind[k] == 'Dn4000')):
                dldnu_red = ll[inired:ifired+1]**2/c
                dldnu_blue = ll[iniblue:ifiblue+1]**2/c
                intblue = sum(flux[iniblue:ifiblue+1]*dblue*dldnu_blue)/titblue
                intred = sum(flux[inired:ifired+1]*dred*dldnu_red)/titred
                output[k] = intred/intblue
            #molecular
            elif ((name_ind[k] == 'CN3883') | (name_ind[k] == 'CN4170') | (name_ind[k] == 'CN1') | (name_ind[k] == 'CN2') 
                | (name_ind[k] == 'Mg_1') | (name_ind[k] == 'Mg_2') | (name_ind[k] == 'TiO1') | (name_ind[k] == 'TiO2')):        
                tit = sum(d)
                output[k] = sum((flux[ini:(ifi+1)]/((ll[ini:(ifi+1)]*coef) + intred - coef*xrm))*d)
                output[k] = -2.5*np.log10(output[k]/tit)
            
            #atomic
            else:
                output[k] = sum((1. - (flux[ini:ifi+1]/((ll[ini:ifi+1]*coef) + intred - coef*xrm)))*d)
    return output

def calcindex_all(ll,flux,blue_sx,blue_dx,feat_sx,feat_dx,red_sx,red_dx,name_ind):
    c = 2.998e+18

    output = np.zeros([len(name_ind),len(flux[:,0])])
    for k in range(len(name_ind)):
        if((ll[0] < blue_sx[k]) & (ll[-1] > red_dx[k])):
            iniblue     = np.max(np.where(ll < blue_sx[k])[0]) + 1
            ifiblue     = np.max(np.where(ll < blue_dx[k])[0])
            inired     = np.max(np.where(ll < red_sx[k])[0]) + 1
            ifired     = np.max(np.where(ll < red_dx[k])[0])

            dblue = (ll[iniblue:ifiblue+1]-ll[iniblue-1:ifiblue])*0.5+(ll[iniblue+1:ifiblue+2]-ll[iniblue:ifiblue+1])*0.5
            dred = (ll[inired:ifired+1]-ll[inired-1:ifired])*0.5+(ll[inired+1:ifired+2]-ll[inired:ifired+1])*0.5
            
            #parte blue sx
            if (ll[iniblue] - (ll[iniblue]-ll[iniblue-1])*0.5 > blue_sx[k]):
                iniblue = iniblue - 1
                dblue = np.append(abs(ll[iniblue] + (ll[iniblue+1] - ll[iniblue])*0.5 - blue_sx[k]),dblue)
            elif (ll[iniblue] - (ll[iniblue]-ll[iniblue-1])*0.5 < blue_sx[k]):
                dblue[0] = dblue[0] - (blue_sx[k] - (ll[iniblue] - (ll[iniblue] - ll[iniblue-1])*0.5))

            #parte blue dx
            if (ll[ifiblue] + (ll[ifiblue+1]-ll[ifiblue])*0.5 < blue_dx[k]):
                ifiblue = ifiblue + 1
                dblue = np.append(dblue,abs((ll[ifiblue] - (ll[ifiblue] - ll[ifiblue - 1])*0.5) - blue_dx[k]))
            elif (ll[ifiblue] + (ll[ifiblue + 1]-ll[ifiblue])*0.5 > blue_dx[k]):
                dblue[-1] = dblue[-1] - abs(ll[ifiblue] + (ll[ifiblue + 1] - ll[ifiblue])*0.5 - blue_dx[k])

            #parte red sx
            if (ll[inired] - (ll[inired]-ll[inired-1])*0.5 > red_sx[k]):
                inired = inired - 1
                dred = np.append(abs(ll[inired] + (ll[inired + 1] - ll[inired])*0.5 - red_sx[k]),dred)
            elif (ll[inired] - (ll[inired]-ll[inired-1])*0.5 < red_sx[k]):
                dred[0] = dred[0] - abs(ll[inired] - (ll[inired] - ll[inired-1])*0.5 - red_sx[k])

            #parte red dx
            if (ll[ifired] + (ll[ifired+1]-ll[ifired])*0.5 < red_dx[k]):
                ifired = ifired + 1
                dred = np.append(dred,abs(ll[ifired] - (ll[ifired + 1] - ll[ifired])*0.5 - red_dx[k]))
            elif (ll[ifired] + (ll[ifired + 1]-ll[ifired])*0.5 > red_dx[k]):
                dred[-1] = dred[-1] - abs(ll[ifired] + (ll[ifired + 1] - ll[ifired])*0.5 - red_dx[k])
        
            
            titblue = sum(dblue)
            titred = sum(dred)
            
            intblue = np.sum(flux[:,iniblue:ifiblue+1]*dblue,axis = 1)/titblue
            intred = np.sum(flux[:,inired:ifired+1]*dred,axis = 1)/titred
            #print(len(intblue),len(intred))
            
            xrm = (red_dx[k] + red_sx[k])/2
            xbm = (blue_dx[k] + blue_sx[k])/2
    
            coef = (intred-intblue)/(xrm-xbm)

            ini     = np.max(np.where(ll <= feat_sx[k])[0]) + 1
            ifi     = np.max(np.where(ll <= feat_dx[k])[0])
    
            d = (ll[ini:ifi+1]-ll[ini-1:ifi])*0.5+(ll[ini+1:ifi+2]-ll[ini:ifi+1])*0.5
    
            #parte feat sx
            if (ll[ini] - (ll[ini]-ll[ini-1])*0.5 > feat_sx[k]):
                ini = ini - 1
                d = np.append(abs(ll[ini] + (ll[ini + 1] - ll[ini])*0.5 - feat_sx[k]),d)
            elif (ll[ini] - (ll[ini]-ll[ini-1])*0.5 < feat_sx[k]):
                d[0] = d[0] - abs(ll[ini] - (ll[ini] - ll[ini-1])*0.5 - feat_sx[k])


            #parte feat dx
            if (ll[ifi] + (ll[ifi+1]-ll[ifi])*0.5 < feat_dx[k]):
                ifi = ifi + 1
                d = np.append(d,abs(ll[ifi] - (ll[ifi + 1] - ll[ifi])*0.5 - feat_dx[k]))
            elif (ll[ifi] + (ll[ifi + 1]-ll[ifi])*0.5 > feat_dx[k]):
                d[-1] = d[-1] - abs(ll[ifi] + (ll[ifi + 1] - ll[ifi])*0.5 - feat_dx[k])
                
            #if (k == 0):
            #    print(len(flux[ini:ifi]))
        
            if ((name_ind[k] == 'D4000') | (name_ind[k] == 'Dn4000')):
                dldnu_red = ll[inired:ifired+1]**2/c
                dldnu_blue = ll[iniblue:ifiblue+1]**2/c
                intblue = np.sum(flux[:,iniblue:ifiblue+1]*dblue*dldnu_blue,axis = 1)/titblue
                intred = np.sum(flux[:,inired:ifired+1]*dred*dldnu_red,axis = 1)/titred
                output[k,:] = intred/intblue
            #molecular
            elif ((name_ind[k] == 'CN3883') | (name_ind[k] == 'CN4170') | (name_ind[k] == 'CN1') | (name_ind[k] == 'CN2') 
                | (name_ind[k] == 'Mg_1') | (name_ind[k] == 'Mg_2') | (name_ind[k] == 'TiO1') | (name_ind[k] == 'TiO2')):        
                tit = sum(d)
                
                output[k,:] = np.sum((flux[:,ini:(ifi+1)].transpose()/(np.outer(ll[ini:(ifi+1)],coef) + intred - coef*xrm)).transpose()*d,axis = 1)
                output[k,:] = -2.5*np.log10(output[k]/tit)
            
            #atomic
            else:
                output[k,:] = np.sum((1. - (flux[:,ini:ifi+1].transpose()/(np.outer(ll[ini:ifi+1],coef) + intred - coef*xrm))).transpose()*d,axis = 1)
    return output

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)





##############################################################################
# NAME:
#   varsmooth
#
# MODIFICATION HISTORY:
#   V1.0.0: Michele Cappellari, Oxford, 3 September 2022
#   V1.1.0: Faster convolution for a scalar sigma. MC, Oxford, 12 November 2023
#   V1.1.1: Removed dependency on legacy scipy.interpolate.interp1d using 
#       faster loop over np.interp. MC, Oxford, 26 April 2024

def varsmooth(x, y, sig_x, xout=None, oversample=1):
    """    
    Fast and accurate convolution with a Gaussian of variable width.

    This function performs an accurate Fourier convolution of a vector, or the
    columns of an array, with a Gaussian kernel that has a varying or constant
    standard deviation (sigma) per pixel. The convolution is done using fast
    Fourier transform (FFT) and the analytic expression of the Fourier
    transform of the Gaussian function, like in the pPXF method. This allows
    for an accurate convolution even when the Gaussian is severely
    undersampled.

    This function is recommended over standard convolution even when dealing
    with a constant Gaussian width, due to its more accurate handling of
    undersampling issues.

    This function implements Algorithm 1 in `Cappellari (2023)
    <https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C>`_

    Input Parameters
    ----------------

    x : array_like
        Coordinate of every pixel in `y`.
    y : array_like
        Input vector or array of column-spectra.
    sig_x : float or array_like
        Gaussian sigma of every pixel in units of `x`.
        If sigma is constant, `sig_x` can be a scalar. 
        In this case, `x` must be uniformly sampled.
    oversample : float, optional
        Oversampling factor before convolution (default: 1).
    xout : array_like, optional
        Output `x` coordinate used to compute the convolved `y`.

    Output Parameters
    -----------------

    yout : array_like
        Convolved vector or columns of the array `y`.

    """
    assert len(x) == len(y), "`x` and `y` must have the same length"

    if np.isscalar(sig_x):
        dx = np.diff(x)
        assert np.all(np.isclose(dx[0], dx)), "`x` must be uniformly spaced, when `sig_x` is a scalar"
        n = len(x)
        sig_max = sig_x*(n - 1)/(x[-1] - x[0])
        y_new = y.T
    else:
        assert len(x) == len(sig_x), "`x` and `sig_x` must have the same length"
        # Stretches spectrum to have equal sigma in the new coordinate
        sig = sig_x/np.gradient(x)
        sig = sig.clip(0.1)   # Clip to >=0.1 pixels
        sig_max = np.max(sig)*oversample
        xs = np.cumsum(sig_max/sig)
        n = int(np.ceil(xs[-1] - xs[0]))
        x_new = np.linspace(xs[0], xs[-1], n)
        y_new = interp(x_new, xs, y.T)

    # Convolve spectrum with a Gaussian using analytic FT like pPXF
    npad = 2**int(np.ceil(np.log2(n)))
    ft = np.fft.rfft(y_new, npad)
    w = np.linspace(0, np.pi*sig_max, ft.shape[-1])
    ft_gau = np.exp(-0.5*w**2)
    yout = np.fft.irfft(ft*ft_gau, npad).T[:n]

    if not np.isscalar(sig_x):
        if xout is not None:
            xs = np.interp(xout, x, xs)  # xs is 1-dim
        yout = interp(xs, x_new, yout.T).T

    return yout
