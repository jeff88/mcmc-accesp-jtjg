import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import pandas as pan
import corner
import numpy as np
import a
import math as mt 
import pickle
plt.style.use('seaborn-darkgrid')

#Multiply by the stuff, correct the energies

t_x = np.loadtxt('Input/Xray.dat',usecols=(0))
tr6g = np.loadtxt('Input/R6g.dat',usecols=(0))
tr3g = np.loadtxt('Input/R3g.dat',usecols=(0))

F_x = np.loadtxt('Input/Xray.dat',usecols=(1))
F_r6g = np.loadtxt('Input/R6g.dat',usecols=(1))
F_r3g = np.loadtxt('Input/R3g.dat',usecols=(1))

#print (t_x, F_x)
Gamma=490.
gama0b=1.5
t= t_x * a.day #160*day#  1.0*sec
tr3= tr3g * a.day #160*day#  1.0*sec
tr6= tr6g * a.day #160*day#  1.0*sec

z= 0.01 #Redshift, fixed parameter
D= 2. * pow(10,26.0) * a.cm  # Luminosity distance, fixed parameter
niter = 7000 
tune = 3000
kk = 1
        
aa=1.4#0.5
bb=1.5#0.3
cc=1.3#0.25 
dd=2.15#063
ll=1.
"""
with pm.Model() as X_ray_model:

########################## Prior Information ##########################

    eta0 =pm.Normal(r'$n \ (10^{-4}cm^{-3})$',1.0,0.1)  #1
    E_off0 = pm.Normal(r'$E_{off} \ (10^{49}erg)$',3.,0.1) #5
    E_0 = pm.Normal(r'$\tilde{E} \ (10^{49}erg)$', 5, 0.1)#3
    dtheta1 = pm.Uniform(r'$\Delta\theta \ (deg)$',15.5, 16.5)#16
    dtheta1a = pm.Uniform(r'$\theta_j \ (deg)$',2.5,3.5) #3
    alpha = pm.Normal(r'$p$',2.25,0.1)#2.2
    delta_1 =pm.Normal(r'$\alpha_S$',2.3,0.1)#2.3
    xiBf0 = pm.Normal(r'$\epsilon_B\ (10^{-4})$',2.,0.1) #2
    xief0 = pm.Normal(r'$\epsilon_e \ (10^{-1}) $',1.0,0.1) #1
    sigma = pm.HalfNormal(r'$\sigma$',10)

########################## Relationships  ##########################
    dtheta = pm.Deterministic('delta_theta', dtheta1 * mt.pi/180.)
    theta_j = pm.Deterministic('theta_j',dtheta1a * mt.pi/180.) 
    delta = pm.Deterministic('delta',4. + delta_1)
    eta=pm.Deterministic('eta',eta0*1e-4*pow(a.cm,-3))
    E0 = pm.Deterministic('E0',E_0*1e49*a.erg)  
    E_off=pm.Deterministic('E_off',E_off0*1e49*a.erg)
    xiBf=pm.Deterministic('xiBf',xiBf0*1e-3)
    xief=pm.Deterministic('xief',xief0*1e-1)
    xf = pm.Deterministic('xf',(-1.0+pow(1.0+4.0*xief/xiBf,1.0/2.0))/2.0)
 
########################## Function Definitions ##########################

    def tdx_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta):
     A1=pow(3./(2.*mt.pi*a.mp), 1./3.)
     return 0.5*kk*A1*pow(1+z,1.)* pow(eta, -1./3. ) * pow(E, 1./3.) * pow(dtheta,2.)*1/a.day

    def Gammax_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):
     A1=pow(32.*mt.pi*a.mp/3., -1./2.) 
     return A1 *pow(1+z,3./2.)* pow(eta,-1./2.)* pow(E,1./2.)* pow(theta_j,-1.) * pow(dtheta,3.) * pow(t,-3./2.)

    def gammam_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):
     A1=pow(32.*mt.pi*a.mp/3., -1./2.)*a.mp/a.me*(alpha-2.)/(alpha-1.)
     return A1*pow(1+z,3./2.)*xief* pow(eta, -1./2.)* pow(E,1./2.)*pow(dtheta,3.)* pow(theta_j,-1.) * pow(t,-3./2.)

    def gammac_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):
     A1=pow(3.*mt.pi/(32.*a.mp), 1./2.) * a.me/a.sigmaT
     nn=1.
     xf=(-1.0+pow(1.0+4.0*nn*xief/xiBf,1.0/2.0))/2.0
     return A1* pow(1+xf,-1.0) * pow(1+z,-1./2.) *pow(xiBf,-1.) * pow(eta,-1./2.)* pow(E,-1./2.)*pow(dtheta,-1.)* pow(theta_j,1.) * pow(t,1./2.)

    def xff_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):
     gam_m_jet=gammam_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)
     gam_c_jet=gammac_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)
     nn=pow(gam_c_jet/gam_m_jet,2.-alpha)
     return  (-1.0+pow(1.0+4.0*nn*xief/xiBf,1.0/2.0))/2.0

    def Em_bb_jet(xiBf,xief,gama0b,E,alpha,eta,D,z,t,delta,dtheta,theta_j): #This module calculates the characteristic frecuency and give the value in Hz
     A1=3.*pow(2.,1./2)*a.e*pow(a.mp,3./2.)/(8.*pow(mt.pi,3./2)*pow(a.me,3.)) * pow((alpha-2)/(alpha-1),2.0)
     return A1*pow(1+z,2.) *pow(xief,2.0)* pow(xiBf,1.0/2.0)*pow(eta, -1./2.) * pow(E,1.0)*pow(dtheta, 4.)* pow(theta_j,-2.) * pow(t,-3.)*1/a.GHz*1/1.48

    def Ec_bb_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):#This module calculates the cut off frecuency and give the value in Hz
     A1=3.*mt.pi*a.e*pow(a.me,1.)/(pow(a.sigmaT,2.0)*pow(32.*mt.pi*a.mp,1./2.))
     xf=xff_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)
     return A1*pow(1+z,-2.)*pow(1+xf,-2.0) * pow(xiBf,-3.0/2.0) * pow(eta,-1./2.) *pow(E,-1.)*pow(dtheta,-4.)* pow(theta_j,2.)*pow(t,1.)*1/a.KeV *1.48 #

    def Fmax_bb_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):#This module calculates the cut off frecuency and give the value in Hz
     A1= 64.* a.sigmaT*a.me*pow(32.*mt.pi*a.mp,3./2.)/(27.*a.e)
     return A1*pow(1+z,-4.) * pow(xiBf,1.0/2.0) * pow(eta,5./2.) * pow(D,-2.)* pow(E,-1.)*pow(dtheta,-18.)* pow(theta_j,2.) *pow(t,6.)*1./a.mJy*1/pow(1.48,2.)*1./(2.*mt.pi)

    def Em_ab_jet(xiBf,xief,gama0b,E,alpha,eta,D,z,t,delta): #This module calculates the characteristic frecuency and give the value in Hz
     A1= pow(8.,1./2)*a.e*pow(a.mp,5./2.)/(pow(mt.pi,1./2)*pow(a.me,3.)) * pow((alpha-2)/(alpha-1),2.0)*pow(3./(2.*mt.pi*a.mp),2./3.)
     return A1*pow(1+z,1.) *pow(xief,2.0)* pow(xiBf,1.0/2.0)*pow(eta, -1./6.) * pow(E,2./3.)* pow(t,-2.)*1/a.GHz*1/1.48

    def Ec_ab_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta):#This module calculates the cut off frecuency and give the value in Hz
     A1=18.*mt.pi*a.e*a.me/(pow(a.sigmaT,2.0)*pow(32.*mt.pi*a.mp,3./2.))* pow(3./(2.*mt.pi*a.mp),-2./3.)
     xf=xff_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)
     return A1*pow(1+z,-1.)*pow(1+xf,-2.0) * pow(xiBf,-3.0/2.0) * pow(eta,-5./6.) *pow(E,-2./3.)*1/a.KeV *1.48 *1.5 #

    def Fmax_ab_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta):#This module calculates the cut off frecuency and give the value in Hz
     A1= a.sigmaT*a.me*pow(32.*mt.pi*a.mp,1./2.)/(9.*a.e)*pow(3./(2.*mt.pi*a.mp),5./3.)
     return pow(kk,2.0)* A1*pow(1+z,4.) * pow(xiBf,1.0/2.0) * pow(eta,-1./6.) * pow(D,-2.)* pow(E,5./3.)*pow(t,-2.)*1./a.mJy*1/pow(1.48,2.)*1./(2.*mt.pi)*0.4

    def gammam_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):
     A1=pow(3./(32.*mt.pi*a.mp), 1/(delta+8))*a.mp/a.me*(alpha-2.)/(alpha-1.)
     return A1*xief*pow(1+z,3./(delta+8.))* pow(eta, -1./(delta+8) )* pow(E1,1./(delta+8)) * pow(t,-3./(delta+8))

    def gammac_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):
     A1=pow(3./(32.*mt.pi*a.mp), -3/(delta+8)) * 3.*a.me/(16.*a.mp*a.sigmaT)
     nn=1.
     xf=(-1.0+pow(1.0+4.0*nn*xief/xiBf,1.0/2.0))/2.0
     return A1* pow(xiBf,-1.)*pow(1+xf,-1.0) * pow(1+z,(delta-1.)/(delta+8.)) * pow(eta, -(delta+5.)/(delta+8) )* pow(E1,-3./(delta+8))* pow(t,(1.-delta)/(delta+8))

    def xff_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):
     gam_m_coc=gammam_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)
     gam_c_coc=gammac_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)
     nn=pow(gam_c_coc/gam_m_coc,2.-alpha)
     return  (-1.0+pow(1.0+4.0*nn*xief/xiBf,1.0/2.0))/2.0

    def tdx_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta1):
     A1=pow(3./(32.*mt.pi*a.mp), 1./3.)
     dtheta1=10.*3.1416/180.
     return 0.5*kk*A1*pow(1+z,1.)* pow(eta, -1./3. ) * pow(E1, 1./3.) * pow(dtheta1,(delta+6.)/3.)*1/a.day

    def Em_bb_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta,dtheta): #This module calculates the characteristic frecuency and give the value in Hz
     A1=pow(2.,1./2)*a.e*pow(a.mp,5./2.)/(pow(mt.pi,1./2)*pow(a.me,3.)) * pow((alpha-2)/(alpha-1),2.0)
     A2=pow(3./(32.*mt.pi*a.mp),4./(delta+8.))
     return 3.* A1*A2*pow(1+z,(4.-delta)/(delta+8)) *pow(xief,2.0)* pow(xiBf,1.0/2.0)*pow(eta, delta/(2*(delta+8))) * pow(E1, 4./(delta+8))* pow(t,-12./(delta+8.))*1/a.GHz*1/1.48

    def Ec_bb_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):#This module calculates the cut off frecuency and give the value in Hz
     A1=pow(2.,1./2.)*a.e*pow(a.me,1.)/(128. * pow(mt.pi,1./2.)* pow(a.mp,3./2.)*pow(a.sigmaT,2.0))
     A2=pow(3./(32.*mt.pi*a.mp), -4./(delta+8))
     xf=xff_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)
     t0=0.
     return A1*A2*pow(1+z,(delta-4.)/(delta+8.))*pow(1+xf,-2.0) * pow(xiBf,-3.0/2.0) * pow(eta,-(16.+3*delta)/(2*(delta+8))) *pow(E1,-4./(delta+8))*pow(t-t0,-2.*(delta+2.)/(delta+8))*1/a.KeV *1.48#
    
    def Fmax_bb_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):#This module calculates the cut off frecuency and give the value in Hz
     A1= pow(2.,1./2) *a.me*a.sigmaT/(12.*pow(mt.pi,1./2.) *pow(a.mp,1./2.)*a.e)
     A2=pow(3./(32.*mt.pi*a.mp), -delta/(delta+8))
     return A1*A2*pow(1+z,-4.*(delta+2.)/(delta+8.)) * pow(xiBf,1.0/2.0) * pow(eta,(3.*delta+8.)/(2*(delta+8))) * pow(D,-2.)* pow(E1,8./(delta+8))*pow(t,3.*delta/(delta+8))*1./a.mJy*1/pow(1.48,2.)

    def Em_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1): #This module calculates the characteristic frecuency and give the value in Hz
     A1=pow(2.,1./2)*a.e*pow(a.mp,5./2.)/(pow(mt.pi,1./2)*pow(a.me,3.)) * pow((alpha-2)/(alpha-1),2.0)
     A2=pow(3./(32.*mt.pi*a.mp),4./(delta1+6.))
     return 3.*A1*A2*pow(1+z,(6-delta1)/(delta1+6.))*pow(xief,2.)*pow(xiBf,1.0/2.0)*pow(eta, (delta1-2.)/(2.*(delta1+6.))) * pow(E1, 4./(delta1+6.)) * pow(t,-12./(delta1+6.))*1/a.GHz 

    def Ec_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1): #This module calculates the characteristic frecuency and give the value in Hz
     A1=pow(2.,1./2.)*a.e*pow(a.me,1.)/(128. * pow(mt.pi,1./2.)* pow(a.mp,3./2.)*pow(sigmaT,2.0))
     A2=pow(3./(32.*mt.pi*a.mp), -4./(delta1+6))
     return A1*A2*pow(1+z, (delta1-6.)/(delta1+6.))*pow(xiBf,-3.0/2.0)*pow(eta,-(3.*delta1 + 10.)/(2.*(delta1+6.)) ) * pow(E1, -4./(delta1+6.)) * pow(1+xf,-2.) * pow(t,-(2.*delta1)/(delta1+6.))*1/a.KeV

    def Fmax_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1):#This module calculates the cut off frecuency and give the value in Hz
     A1= pow(2.,1./2) *a.me*a.sigmaT/(12.*pow(mt.pi,1./2.) *pow(a.mp,1./2.)*a.e)
     A2=pow(3./(32.*mt.pi*a.mp), (2.-delta1)/(delta1+6))
     return A1*A2*pow(1+z, (12.- 2.*delta1)/(delta1+6.))*pow(xiBf,1.0/2.0)*pow(eta, (3.*delta1+2.)/(2.*(delta1+6.))) * pow(D,-2.0)*pow(E1, 8./(delta1+6.))*pow(t, -3*(2.-delta1)/(delta1+6.))*1/a.mJy



########################## Fit Model ##########################



    td_jet = pm.Deterministic('tdj', tdx_jet(xiBf,xief,E_off,alpha,eta,D,z,t,delta,dtheta))
    #E_m_jet = pm.Deterministic('emj', Em_bb_jet(xiBf,xief,gama0b,E,alpha,eta,D,z,t,delta,dtheta,theta_j)*a.GHz)
    #E_c_jet = pm.Deterministic('ecj', Ec_bb_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)*a.KeV)
    #Fmax_jet = pm.Deterministic('fmj', Fmax_bb_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)*a.mJy)
    #E_c_R_jet = pm.Deterministic('ecrj', Ec_ab_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta)*a.KeV)
    #E_m_R_jet = pm.Deterministic('emrj', Em_ab_jet(xiBf,xief,gama0b,E,alpha,eta,D,z,t,delta)*a.GHz)
    #Fmax_R_jet = pm.Deterministic('fmrj', Fmax_ab_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta)*a.mJy)
    td_coc = pm.Deterministic('tdc',tdx_coc(xiBf,xief,E0,alpha,eta,D,z,t,delta,dtheta))
    #E_m_coc=pm.Deterministic('emc',Em_bb_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta,dtheta)*a.GHz)
    #Fmax_coc=pm.Deterministic('fmc',Fmax_bb_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)*a.mJy)
    #E_c_coc=pm.Deterministic('ecc',Ec_bb_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)*a.KeV)
    #E_m_R_coc=pm.Deterministic('emrc',Em_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1)*a.GHz)
    #E_c_R_coc=pm.Deterministic('ecrc',Ec_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1)*a.KeV)
    #Fmax_R_coc=pm.Deterministic('emrc',Fmax_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1)*a.mJy)

    if((t<td_jet).any()):
     Egammam_sc = 1e3 * a.eV  #  X-rays
     E_m_jet=pm.Deterministic('E_m_jet',Em_bb_jet(xiBf,xief,gama0b,E_off,alpha,eta,D,z,t,delta,dtheta,theta_j)*a.GHz)
     E_c_jet=pm.Deterministic('E_c_jet',Ec_bb_jet(xiBf,xief,E_off,alpha,eta,D,z,t,delta,dtheta,theta_j)*a.KeV)
     Fmax_jet=pm.Deterministic('F_max_jet',Fmax_bb_jet(xiBf,xief,E_off,alpha,eta,D,z,t,delta,dtheta,theta_j)*a.mJy*1)
     Fm_sc_Xray=pm.Deterministic('Fm_sc_Xray',Fmax_jet*pow(Egammam_sc/E_m_jet,-(alpha-1.)/2.)*1./a.micJy)
     Fm_sc_Xray_obs = pm.Normal('Fm_sc_Xray_obs', mu=Fm_sc_Xray, sd=sigma, observed=F_x)
    else:
     Egammam_sc_R=1e3 * a.eV  # X-ray
     E_m_R_jet=pm.Deterministic('E_m_R_jet',Em_ab_jet(xiBf,xief,gama0b,E_off,alpha,eta,D,z,t,delta)*a.GHz)
     E_c_R_jet=pm.Deterministic('E_c_R_jet',Ec_ab_jet(xiBf,xief,E_off,alpha,eta,D,z,t,delta,dtheta)*a.KeV)
     Fmax_R_jet=pm.Deterministic('F_max_R_jet',Fmax_ab_jet(xiBf,xief,E_off,alpha,eta,D,z,t,delta,dtheta)*a.mJy*1.)
     Fm_sc_Xray=pm.Deterministic('Fm_sc_Xray',Fmax_R_jet*pow(Egammam_sc_R/E_m_R_jet,-(alpha-1.)/2.)*1./a.micJy)
     Fm_sc_Xray_obs = pm.Normal('Fm_sc_Xray_obs', mu=Fm_sc_Xray, sd=sigma, observed=F_x)
    
   
    if((t<td_coc).any()):
     Egammam_sc = 1e3 * a.eV  #  X-rays
     E_m_coc=pm.Deterministic('E_m_coc',Em_bb_coc(xiBf,xief,gama0b,E0,alpha,eta,D,z,t,delta,dtheta)*a.GHz)
     E_c_coc=pm.Deterministic('E_c_coc',Ec_bb_coc(xiBf,xief,E0,alpha,eta,D,z,t,delta,dtheta)*a.KeV)
     Fmax_coc=pm.Deterministic('Fmax_coc',Fmax_bb_coc(xiBf,xief,E0,alpha,eta,D,z,t,delta,dtheta)*a.mJy)
     Fm_sc_Xray_c=pm.Deterministic('Fm_sc_Xray_c',Fmax_coc*pow(Egammam_sc/E_m_coc,-(alpha-1.)/2.)*1./a.micJy)
     Fm_sc_Xray_obs = pm.Normal('Fm_sc_Xray_obs_c', mu=Fm_sc_Xray_c, sd=sigma, observed=F_x)
    else:
     Egammam_sc_R=1e3 * a.eV  # X-ray
     E_m_R_coc=pm.Deterministic('E_m_R_coc',Em_ab_coc(xiBf,xief,gama0b,E0,alpha,eta,D,z,t,delta)*a.GHz)
     E_c_R_coc=pm.Deterministic('E_c_R_coc',Ec_ab_coc(xiBf,xief,gama0b,E0,alpha,eta,D,z,t,delta)*a.KeV)
     Fmax_R_coc=pm.Deterministic('Fmax_R_coc',Fmax_ab_coc(xiBf,xief,gama0b,E0,alpha,eta,D,z,t,delta)*a.mJy)
     Fm_sc_Xray_c=pm.Deterministic('Fm_sc_Xray_c',Fmax_R_coc*pow(Egammam_sc_R/E_m_R_coc,-(alpha-1.)/2.) * 1./a.micJy)
     Fm_sc_Xray_obs = pm.Normal('Fm_sc_Xray_obs_c', mu=Fm_sc_Xray_c, sd=sigma, observed=F_x)
    

    
########################## Step Definition ##########################

    step1 = pm.NUTS(target_accept=0.99)

########################## Sampling ##########################

    trace = pm.sample(niter, step=[step1], init="adapt_diag",tune=tune, random_seed = 123)
    aa    = pm.backends.tracetab.trace_to_dataframe(trace, varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
    aa.to_csv(path_or_buf = "Output/X-Ray/output.csv", sep="\t")
    summary=pm.stats.summary(trace,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
    summary.to_csv(path_or_buf = "Output/X-Ray/summary.csv", sep="\t")
    print(summary)
"""
with pm.Model() as Radio6GHz_model:

########################## Prior Information ##########################

    eta0 =pm.Normal(r'$n \ (10^{-4}cm^{-3})$',1.0,0.1)  #1
    E_off0 = pm.Normal(r'$E_{off} \ (10^{49}erg)$',3.,0.1) #5
    E_0 = pm.Normal(r'$\tilde{E} \ (10^{49}erg)$', 5, 0.1)#3
    dtheta1 = pm.Uniform(r'$\Delta\theta \ (deg)$',15.5, 16.5)#16
    dtheta1a = pm.Uniform(r'$\theta_j \ (deg)$',2.5,3.5) #3
    alpha = pm.Normal(r'$p$',2.25,0.1)#2.2
    delta_1 =pm.Normal(r'$\alpha_S$',2.3,0.1)#2.3
    xiBf0 = pm.Normal(r'$\epsilon_B\ (10^{-4})$',2.,0.1) #2
    xief0 = pm.Normal(r'$\epsilon_e \ (10^{-1}) $',1.0,0.1) #1
    sigma = pm.HalfNormal(r'$\sigma$',10)

########################## Relationships  ##########################
    dtheta = pm.Deterministic('delta_theta', dtheta1 * mt.pi/180.)
    theta_j = pm.Deterministic('theta_j',dtheta1a * mt.pi/180.) 
    delta = pm.Deterministic('delta',4. + delta_1)
    eta=pm.Deterministic('eta',eta0*1e-4*pow(a.cm,-3))
    E0 = pm.Deterministic('E0',E_0*1e49*a.erg)  
    E_off=pm.Deterministic('E_off',E_off0*1e49*a.erg)
    xiBf=pm.Deterministic('xiBf',xiBf0*1e-3)
    xief=pm.Deterministic('xief',xief0*1e-1)
    xf = pm.Deterministic('xf',(-1.0+pow(1.0+4.0*xief/xiBf,1.0/2.0))/2.0)
 
########################## Function Definitions ##########################

    def tdx_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta):
     A1=pow(3./(2.*mt.pi*a.mp), 1./3.)
     return 0.5*kk*A1*pow(1+z,1.)* pow(eta, -1./3. ) * pow(E, 1./3.) * pow(dtheta,2.)*1/a.day

    def Gammax_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):
     A1=pow(32.*mt.pi*a.mp/3., -1./2.) 
     return A1 *pow(1+z,3./2.)* pow(eta,-1./2.)* pow(E,1./2.)* pow(theta_j,-1.) * pow(dtheta,3.) * pow(t,-3./2.)

    def gammam_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):
     A1=pow(32.*mt.pi*a.mp/3., -1./2.)*a.mp/a.me*(alpha-2.)/(alpha-1.)
     return A1*pow(1+z,3./2.)*xief* pow(eta, -1./2.)* pow(E,1./2.)*pow(dtheta,3.)* pow(theta_j,-1.) * pow(t,-3./2.)

    def gammac_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):
     A1=pow(3.*mt.pi/(32.*a.mp), 1./2.) * a.me/a.sigmaT
     nn=1.
     xf=(-1.0+pow(1.0+4.0*nn*xief/xiBf,1.0/2.0))/2.0
     return A1* pow(1+xf,-1.0) * pow(1+z,-1./2.) *pow(xiBf,-1.) * pow(eta,-1./2.)* pow(E,-1./2.)*pow(dtheta,-1.)* pow(theta_j,1.) * pow(t,1./2.)

    def xff_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):
     gam_m_jet=gammam_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)
     gam_c_jet=gammac_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)
     nn=pow(gam_c_jet/gam_m_jet,2.-alpha)
     return  (-1.0+pow(1.0+4.0*nn*xief/xiBf,1.0/2.0))/2.0

    def Em_bb_jet(xiBf,xief,gama0b,E,alpha,eta,D,z,t,delta,dtheta,theta_j): #This module calculates the characteristic frecuency and give the value in Hz
     A1=3.*pow(2.,1./2)*a.e*pow(a.mp,3./2.)/(8.*pow(mt.pi,3./2)*pow(a.me,3.)) * pow((alpha-2)/(alpha-1),2.0)
     return A1*pow(1+z,2.) *pow(xief,2.0)* pow(xiBf,1.0/2.0)*pow(eta, -1./2.) * pow(E,1.0)*pow(dtheta, 4.)* pow(theta_j,-2.) * pow(t,-3.)*1/a.GHz*1/1.48

    def Ec_bb_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):#This module calculates the cut off frecuency and give the value in Hz
     A1=3.*mt.pi*a.e*pow(a.me,1.)/(pow(a.sigmaT,2.0)*pow(32.*mt.pi*a.mp,1./2.))
     xf=xff_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)
     return A1*pow(1+z,-2.)*pow(1+xf,-2.0) * pow(xiBf,-3.0/2.0) * pow(eta,-1./2.) *pow(E,-1.)*pow(dtheta,-4.)* pow(theta_j,2.)*pow(t,1.)*1/a.KeV *1.48 #

    def Fmax_bb_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):#This module calculates the cut off frecuency and give the value in Hz
     A1= 64.* a.sigmaT*a.me*pow(32.*mt.pi*a.mp,3./2.)/(27.*a.e)
     return A1*pow(1+z,-4.) * pow(xiBf,1.0/2.0) * pow(eta,5./2.) * pow(D,-2.)* pow(E,-1.)*pow(dtheta,-18.)* pow(theta_j,2.) *pow(t,6.)*1./a.mJy*1/pow(1.48,2.)*1./(2.*mt.pi)

    def Em_ab_jet(xiBf,xief,gama0b,E,alpha,eta,D,z,t,delta): #This module calculates the characteristic frecuency and give the value in Hz
     A1= pow(8.,1./2)*a.e*pow(a.mp,5./2.)/(pow(mt.pi,1./2)*pow(a.me,3.)) * pow((alpha-2)/(alpha-1),2.0)*pow(3./(2.*mt.pi*a.mp),2./3.)
     return A1*pow(1+z,1.) *pow(xief,2.0)* pow(xiBf,1.0/2.0)*pow(eta, -1./6.) * pow(E,2./3.)* pow(t,-2.)*1/a.GHz*1/1.48

    def Ec_ab_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta):#This module calculates the cut off frecuency and give the value in Hz
     A1=18.*mt.pi*a.e*a.me/(pow(a.sigmaT,2.0)*pow(32.*mt.pi*a.mp,3./2.))* pow(3./(2.*mt.pi*a.mp),-2./3.)
     xf=xff_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)
     return A1*pow(1+z,-1.)*pow(1+xf,-2.0) * pow(xiBf,-3.0/2.0) * pow(eta,-5./6.) *pow(E,-2./3.)*1/a.KeV *1.48 *1.5 #

    def Fmax_ab_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta):#This module calculates the cut off frecuency and give the value in Hz
     A1= a.sigmaT*a.me*pow(32.*mt.pi*a.mp,1./2.)/(9.*a.e)*pow(3./(2.*mt.pi*a.mp),5./3.)
     return pow(kk,2.0)* A1*pow(1+z,4.) * pow(xiBf,1.0/2.0) * pow(eta,-1./6.) * pow(D,-2.)* pow(E,5./3.)*pow(t,-2.)*1./a.mJy*1/pow(1.48,2.)*1./(2.*mt.pi)*0.4

    def gammam_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):
     A1=pow(3./(32.*mt.pi*a.mp), 1/(delta+8))*a.mp/a.me*(alpha-2.)/(alpha-1.)
     return A1*xief*pow(1+z,3./(delta+8.))* pow(eta, -1./(delta+8) )* pow(E1,1./(delta+8)) * pow(t,-3./(delta+8))

    def gammac_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):
     A1=pow(3./(32.*mt.pi*a.mp), -3/(delta+8)) * 3.*a.me/(16.*a.mp*a.sigmaT)
     nn=1.
     xf=(-1.0+pow(1.0+4.0*nn*xief/xiBf,1.0/2.0))/2.0
     return A1* pow(xiBf,-1.)*pow(1+xf,-1.0) * pow(1+z,(delta-1.)/(delta+8.)) * pow(eta, -(delta+5.)/(delta+8) )* pow(E1,-3./(delta+8))* pow(t,(1.-delta)/(delta+8))

    def xff_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):
     gam_m_coc=gammam_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)
     gam_c_coc=gammac_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)
     nn=pow(gam_c_coc/gam_m_coc,2.-alpha)
     return  (-1.0+pow(1.0+4.0*nn*xief/xiBf,1.0/2.0))/2.0

    def tdx_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta1):
     A1=pow(3./(32.*mt.pi*a.mp), 1./3.)
     dtheta1=10.*3.1416/180.
     return 0.5*kk*A1*pow(1+z,1.)* pow(eta, -1./3. ) * pow(E1, 1./3.) * pow(dtheta1,(delta+6.)/3.)*1/a.day

    def Em_bb_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta,dtheta): #This module calculates the characteristic frecuency and give the value in Hz
     A1=pow(2.,1./2)*a.e*pow(a.mp,5./2.)/(pow(mt.pi,1./2)*pow(a.me,3.)) * pow((alpha-2)/(alpha-1),2.0)
     A2=pow(3./(32.*mt.pi*a.mp),4./(delta+8.))
     return 3.* A1*A2*pow(1+z,(4.-delta)/(delta+8)) *pow(xief,2.0)* pow(xiBf,1.0/2.0)*pow(eta, delta/(2*(delta+8))) * pow(E1, 4./(delta+8))* pow(t,-12./(delta+8.))*1/a.GHz*1/1.48

    def Ec_bb_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):#This module calculates the cut off frecuency and give the value in Hz
     A1=pow(2.,1./2.)*a.e*pow(a.me,1.)/(128. * pow(mt.pi,1./2.)* pow(a.mp,3./2.)*pow(a.sigmaT,2.0))
     A2=pow(3./(32.*mt.pi*a.mp), -4./(delta+8))
     xf=xff_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)
     t0=0.
     return A1*A2*pow(1+z,(delta-4.)/(delta+8.))*pow(1+xf,-2.0) * pow(xiBf,-3.0/2.0) * pow(eta,-(16.+3*delta)/(2*(delta+8))) *pow(E1,-4./(delta+8))*pow(t-t0,-2.*(delta+2.)/(delta+8))*1/a.KeV *1.48#
    
    def Fmax_bb_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):#This module calculates the cut off frecuency and give the value in Hz
     A1= pow(2.,1./2) *a.me*a.sigmaT/(12.*pow(mt.pi,1./2.) *pow(a.mp,1./2.)*a.e)
     A2=pow(3./(32.*mt.pi*a.mp), -delta/(delta+8))
     return A1*A2*pow(1+z,-4.*(delta+2.)/(delta+8.)) * pow(xiBf,1.0/2.0) * pow(eta,(3.*delta+8.)/(2*(delta+8))) * pow(D,-2.)* pow(E1,8./(delta+8))*pow(t,3.*delta/(delta+8))*1./a.mJy*1/pow(1.48,2.)

    def Em_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1): #This module calculates the characteristic frecuency and give the value in Hz
     A1=pow(2.,1./2)*a.e*pow(a.mp,5./2.)/(pow(mt.pi,1./2)*pow(a.me,3.)) * pow((alpha-2)/(alpha-1),2.0)
     A2=pow(3./(32.*mt.pi*a.mp),4./(delta1+6.))
     return 3.*A1*A2*pow(1+z,(6-delta1)/(delta1+6.))*pow(xief,2.)*pow(xiBf,1.0/2.0)*pow(eta, (delta1-2.)/(2.*(delta1+6.))) * pow(E1, 4./(delta1+6.)) * pow(t,-12./(delta1+6.))*1/a.GHz 

    def Ec_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1): #This module calculates the characteristic frecuency and give the value in Hz
     A1=pow(2.,1./2.)*a.e*pow(a.me,1.)/(128. * pow(mt.pi,1./2.)* pow(a.mp,3./2.)*pow(sigmaT,2.0))
     A2=pow(3./(32.*mt.pi*a.mp), -4./(delta1+6))
     return A1*A2*pow(1+z, (delta1-6.)/(delta1+6.))*pow(xiBf,-3.0/2.0)*pow(eta,-(3.*delta1 + 10.)/(2.*(delta1+6.)) ) * pow(E1, -4./(delta1+6.)) * pow(1+xf,-2.) * pow(t,-(2.*delta1)/(delta1+6.))*1/a.KeV

    def Fmax_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1):#This module calculates the cut off frecuency and give the value in Hz
     A1= pow(2.,1./2) *a.me*a.sigmaT/(12.*pow(mt.pi,1./2.) *pow(a.mp,1./2.)*a.e)
     A2=pow(3./(32.*mt.pi*a.mp), (2.-delta1)/(delta1+6))
     return A1*A2*pow(1+z, (12.- 2.*delta1)/(delta1+6.))*pow(xiBf,1.0/2.0)*pow(eta, (3.*delta1+2.)/(2.*(delta1+6.))) * pow(D,-2.0)*pow(E1, 8./(delta1+6.))*pow(t, -3*(2.-delta1)/(delta1+6.))*1/a.mJy



########################## Fit Model ##########################

    td_jet = pm.Deterministic('tdj', tdx_jet(xiBf,xief,E_off,alpha,eta,D,z,tr6,delta,dtheta))
    #E_m_jet = pm.Deterministic('emj', Em_bb_jet(xiBf,xief,gama0b,E,alpha,eta,D,z,t,delta,dtheta,theta_j)*a.GHz)
    #E_c_jet = pm.Deterministic('ecj', Ec_bb_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)*a.KeV)
    #Fmax_jet = pm.Deterministic('fmj', Fmax_bb_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)*a.mJy)
    #E_c_R_jet = pm.Deterministic('ecrj', Ec_ab_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta)*a.KeV)
    #E_m_R_jet = pm.Deterministic('emrj', Em_ab_jet(xiBf,xief,gama0b,E,alpha,eta,D,z,t,delta)*a.GHz)
    #Fmax_R_jet = pm.Deterministic('fmrj', Fmax_ab_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta)*a.mJy)
    td_coc = pm.Deterministic('tdc',tdx_coc(xiBf,xief,E0,alpha,eta,D,z,tr6,delta,dtheta))
    #E_m_coc=pm.Deterministic('emc',Em_bb_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta,dtheta)*a.GHz)
    #Fmax_coc=pm.Deterministic('fmc',Fmax_bb_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)*a.mJy)
    #E_c_coc=pm.Deterministic('ecc',Ec_bb_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)*a.KeV)
    #E_m_R_coc=pm.Deterministic('emrc',Em_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1)*a.GHz)
    #E_c_R_coc=pm.Deterministic('ecrc',Ec_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1)*a.KeV)
    #Fmax_R_coc=pm.Deterministic('emrc',Fmax_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1)*a.mJy)

    if((tr6<td_jet).any()):
     Egammam_sc = 6 * a.GHz
     E_m_jet=pm.Deterministic('E_m_jet',Em_bb_jet(xiBf,xief,gama0b,E_off,alpha,eta,D,z,tr6,delta,dtheta,theta_j)*a.GHz)
     E_c_jet=pm.Deterministic('E_c_jet',Ec_bb_jet(xiBf,xief,E_off,alpha,eta,D,z,tr6,delta,dtheta,theta_j)*a.KeV)
     Fmax_jet=pm.Deterministic('F_max_jet',Fmax_bb_jet(xiBf,xief,E_off,alpha,eta,D,z,tr6,delta,dtheta,theta_j)*a.mJy*1)
     
     x = np.array([Em_bb_jet(xiBf,xief,gama0b,E_off,alpha,eta,D,z,tr6,delta,dtheta,theta_j)*a.GHz,
                   Ec_bb_jet(xiBf,xief,E_off,alpha,eta,D,z,tr6,delta,dtheta,theta_j)*a.KeV,
                   Fmax_bb_jet(xiBf,xief,E_off,alpha,eta,D,z,tr6,delta,dtheta,theta_j)*a.mJy*1])
                   
     np.save('values', x)
     
     Fm_sc_Radio6=pm.Deterministic('Fm_sc_Radio6',Fmax_jet*pow(Egammam_sc/E_m_jet,-(alpha-1.)/2.)*1./a.micJy)
     Fm_sc_Radio6_obs = pm.Normal('Fm_sc_Radio6_obs', mu=Fm_sc_Radio6, sd=sigma, observed=F_r6g)
     
     np.save("values2", np.array([Fmax_jet*pow(Egammam_sc/E_m_jet,-(alpha-1.)/2.)*1./a.micJy])
     
    else:
     Egammam_sc_R= 6 * a.GHz
     E_m_R_jet=pm.Deterministic('E_m_R_jet',Em_ab_jet(xiBf,xief,gama0b,E_off,alpha,eta,D,z,tr6,delta)*a.GHz)
     E_c_R_jet=pm.Deterministic('E_c_R_jet',Ec_ab_jet(xiBf,xief,E_off,alpha,eta,D,z,tr6,delta,dtheta)*a.KeV)
     Fmax_R_jet=pm.Deterministic('F_max_R_jet',Fmax_ab_jet(xiBf,xief,E_off,alpha,eta,D,z,tr6,delta,dtheta)*a.mJy*1.)
     Fm_sc_Radio6=pm.Deterministic('Fm_sc_Radio6',Fmax_R_jet*pow(Egammam_sc_R/E_m_R_jet,-(alpha-1.)/2.)*1./a.micJy)
     Fm_sc_Radio6_obs = pm.Normal('Fm_sc_Radio6_obs', mu=Fm_sc_Radio6, sd=sigma, observed=F_r6g)
    
   
    if((tr6<td_coc).any()):
     Egammam_sc = 6 * a.GHz
     E_m_coc=pm.Deterministic('E_m_coc',Em_bb_coc(xiBf,xief,gama0b,E0,alpha,eta,D,z,tr6,delta,dtheta)*a.GHz)
    
     E_c_coc=pm.Deterministic('E_c_coc',Ec_bb_coc(xiBf,xief,E0,alpha,eta,D,z,tr6,delta,dtheta)*a.KeV)
     Fmax_coc=pm.Deterministic('Fmax_coc',Fmax_bb_coc(xiBf,xief,E0,alpha,eta,D,z,tr6,delta,dtheta)*a.mJy)
     Fm_sc_Radio6_c=pm.Deterministic('Fm_sc_Radio6_c',Fmax_coc*pow(Egammam_sc/E_m_coc,-(alpha-1.)/2.)*1./a.micJy)
     Fm_sc_Radio6_obs = pm.Normal('Fm_sc_Radio6_obs_c', mu=Fm_sc_Radio6_c, sd=sigma, observed=F_r6g)
    else:
     Egammam_sc_R= 6 * a.GHz
     E_m_R_coc=pm.Deterministic('E_m_R_coc',Em_ab_coc(xiBf,xief,gama0b,E0,alpha,eta,D,z,tr6,delta)*a.GHz)
     E_c_R_coc=pm.Deterministic('E_c_R_coc',Ec_ab_coc(xiBf,xief,gama0b,E0,alpha,eta,D,z,tr6,delta)*a.KeV)
     Fmax_R_coc=pm.Deterministic('Fmax_R_coc',Fmax_ab_coc(xiBf,xief,gama0b,E0,alpha,eta,D,z,tr6,delta)*a.mJy)
     Fm_sc_Radio6_c=pm.Deterministic('Fm_sc_Radio6_c',Fmax_R_coc*pow(Egammam_sc_R/E_m_R_coc,-(alpha-1.)/2.) * 1./a.micJy)
     Fm_sc_Radio6_obs = pm.Normal('Fm_sc_Radio6_obs_c', mu=Fm_sc_Radio6_c, sd=sigma, observed=F_r6g)
    

    
########################## Step Definition ##########################

    step1 = pm.NUTS(target_accept=0.99)

########################## Sampling ##########################

    trace = pm.sample(niter, step=[step1], init="adapt_diag",tune=tune, random_seed = 123)
    aa    = pm.backends.tracetab.trace_to_dataframe(trace, varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
    aa.to_csv(path_or_buf = "Output/Radio6GHz/output.csv", sep="\t")
    summary=pm.stats.summary(trace,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
    summary.to_csv(path_or_buf = "Output/Radio6GHz/summary.csv", sep="\t")
    print(summary)

with pm.Model() as Radio3GHz_model:

########################## Prior Information ##########################

    eta0 =pm.Normal(r'$n \ (10^{-4}cm^{-3})$',1.0,0.1)  #1
    E_off0 = pm.Normal(r'$E_{off} \ (10^{49}erg)$',3.,0.1) #5
    E_0 = pm.Normal(r'$\tilde{E} \ (10^{49}erg)$', 5, 0.1)#3
    dtheta1 = pm.Uniform(r'$\Delta\theta \ (deg)$',15.5, 16.5)#16
    dtheta1a = pm.Uniform(r'$\theta_j \ (deg)$',2.5,3.5) #3
    alpha = pm.Normal(r'$p$',2.23,0.05)#2.2
    delta_1 =pm.Normal(r'$\alpha_S$',2.3,0.1)#2.3
    xiBf0 = pm.Normal(r'$\epsilon_B\ (10^{-4})$',2.,0.1) #2
    xief0 = pm.Normal(r'$\epsilon_e \ (10^{-1}) $',1.0,0.1) #1
    sigma = pm.HalfNormal(r'$\sigma$',10)

########################## Relationships  ##########################
    dtheta = pm.Deterministic('delta_theta', dtheta1 * mt.pi/180.)
    theta_j = pm.Deterministic('theta_j',dtheta1a * mt.pi/180.) 
    delta = pm.Deterministic('delta',4. + delta_1)
    eta=pm.Deterministic('eta',eta0*1e-4*pow(a.cm,-3))
    E0 = pm.Deterministic('E0',E_0*1e49*a.erg)  
    E_off=pm.Deterministic('E_off',E_off0*1e49*a.erg)
    xiBf=pm.Deterministic('xiBf',xiBf0*1e-3)
    xief=pm.Deterministic('xief',xief0*1e-1)
    xf = pm.Deterministic('xf',(-1.0+pow(1.0+4.0*xief/xiBf,1.0/2.0))/2.0)
 
########################## Function Definitions ##########################

    def tdx_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta):
     A1=pow(3./(2.*mt.pi*a.mp), 1./3.)
     return 0.5*kk*A1*pow(1+z,1.)* pow(eta, -1./3. ) * pow(E, 1./3.) * pow(dtheta,2.)*1/a.day

    def Gammax_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):
     A1=pow(32.*mt.pi*a.mp/3., -1./2.) 
     return A1 *pow(1+z,3./2.)* pow(eta,-1./2.)* pow(E,1./2.)* pow(theta_j,-1.) * pow(dtheta,3.) * pow(t,-3./2.)

    def gammam_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):
     A1=pow(32.*mt.pi*a.mp/3., -1./2.)*a.mp/a.me*(alpha-2.)/(alpha-1.)
     return A1*pow(1+z,3./2.)*xief* pow(eta, -1./2.)* pow(E,1./2.)*pow(dtheta,3.)* pow(theta_j,-1.) * pow(t,-3./2.)

    def gammac_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):
     A1=pow(3.*mt.pi/(32.*a.mp), 1./2.) * a.me/a.sigmaT
     nn=1.
     xf=(-1.0+pow(1.0+4.0*nn*xief/xiBf,1.0/2.0))/2.0
     return A1* pow(1+xf,-1.0) * pow(1+z,-1./2.) *pow(xiBf,-1.) * pow(eta,-1./2.)* pow(E,-1./2.)*pow(dtheta,-1.)* pow(theta_j,1.) * pow(t,1./2.)

    def xff_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):
     gam_m_jet=gammam_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)
     gam_c_jet=gammac_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)
     nn=pow(gam_c_jet/gam_m_jet,2.-alpha)
     return  (-1.0+pow(1.0+4.0*nn*xief/xiBf,1.0/2.0))/2.0

    def Em_bb_jet(xiBf,xief,gama0b,E,alpha,eta,D,z,t,delta,dtheta,theta_j): #This module calculates the characteristic frecuency and give the value in Hz
     A1=3.*pow(2.,1./2)*a.e*pow(a.mp,3./2.)/(8.*pow(mt.pi,3./2)*pow(a.me,3.)) * pow((alpha-2)/(alpha-1),2.0)
     return A1*pow(1+z,2.) *pow(xief,2.0)* pow(xiBf,1.0/2.0)*pow(eta, -1./2.) * pow(E,1.0)*pow(dtheta, 4.)* pow(theta_j,-2.) * pow(t,-3.)*1/a.GHz*1/1.48

    def Ec_bb_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):#This module calculates the cut off frecuency and give the value in Hz
     A1=3.*mt.pi*a.e*pow(a.me,1.)/(pow(a.sigmaT,2.0)*pow(32.*mt.pi*a.mp,1./2.))
     xf=xff_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)
     return A1*pow(1+z,-2.)*pow(1+xf,-2.0) * pow(xiBf,-3.0/2.0) * pow(eta,-1./2.) *pow(E,-1.)*pow(dtheta,-4.)* pow(theta_j,2.)*pow(t,1.)*1/a.KeV *1.48 #

    def Fmax_bb_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j):#This module calculates the cut off frecuency and give the value in Hz
     A1= 64.* a.sigmaT*a.me*pow(32.*mt.pi*a.mp,3./2.)/(27.*a.e)
     return A1*pow(1+z,-4.) * pow(xiBf,1.0/2.0) * pow(eta,5./2.) * pow(D,-2.)* pow(E,-1.)*pow(dtheta,-18.)* pow(theta_j,2.) *pow(t,6.)*1./a.mJy*1/pow(1.48,2.)*1./(2.*mt.pi)

    def Em_ab_jet(xiBf,xief,gama0b,E,alpha,eta,D,z,t,delta): #This module calculates the characteristic frecuency and give the value in Hz
     A1= pow(8.,1./2)*a.e*pow(a.mp,5./2.)/(pow(mt.pi,1./2)*pow(a.me,3.)) * pow((alpha-2)/(alpha-1),2.0)*pow(3./(2.*mt.pi*a.mp),2./3.)
     return A1*pow(1+z,1.) *pow(xief,2.0)* pow(xiBf,1.0/2.0)*pow(eta, -1./6.) * pow(E,2./3.)* pow(t,-2.)*1/a.GHz*1/1.48

    def Ec_ab_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta):#This module calculates the cut off frecuency and give the value in Hz
     A1=18.*mt.pi*a.e*a.me/(pow(a.sigmaT,2.0)*pow(32.*mt.pi*a.mp,3./2.))* pow(3./(2.*mt.pi*a.mp),-2./3.)
     xf=xff_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)
     return A1*pow(1+z,-1.)*pow(1+xf,-2.0) * pow(xiBf,-3.0/2.0) * pow(eta,-5./6.) *pow(E,-2./3.)*1/a.KeV *1.48 *1.5 #

    def Fmax_ab_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta):#This module calculates the cut off frecuency and give the value in Hz
     A1= a.sigmaT*a.me*pow(32.*mt.pi*a.mp,1./2.)/(9.*a.e)*pow(3./(2.*mt.pi*a.mp),5./3.)
     return pow(kk,2.0)* A1*pow(1+z,4.) * pow(xiBf,1.0/2.0) * pow(eta,-1./6.) * pow(D,-2.)* pow(E,5./3.)*pow(t,-2.)*1./a.mJy*1/pow(1.48,2.)*1./(2.*mt.pi)*0.4

    def gammam_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):
     A1=pow(3./(32.*mt.pi*a.mp), 1/(delta+8))*a.mp/a.me*(alpha-2.)/(alpha-1.)
     return A1*xief*pow(1+z,3./(delta+8.))* pow(eta, -1./(delta+8) )* pow(E1,1./(delta+8)) * pow(t,-3./(delta+8))

    def gammac_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):
     A1=pow(3./(32.*mt.pi*a.mp), -3/(delta+8)) * 3.*a.me/(16.*a.mp*a.sigmaT)
     nn=1.
     xf=(-1.0+pow(1.0+4.0*nn*xief/xiBf,1.0/2.0))/2.0
     return A1* pow(xiBf,-1.)*pow(1+xf,-1.0) * pow(1+z,(delta-1.)/(delta+8.)) * pow(eta, -(delta+5.)/(delta+8) )* pow(E1,-3./(delta+8))* pow(t,(1.-delta)/(delta+8))

    def xff_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):
     gam_m_coc=gammam_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)
     gam_c_coc=gammac_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)
     nn=pow(gam_c_coc/gam_m_coc,2.-alpha)
     return  (-1.0+pow(1.0+4.0*nn*xief/xiBf,1.0/2.0))/2.0

    def tdx_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta1):
     A1=pow(3./(32.*mt.pi*a.mp), 1./3.)
     dtheta1=10.*3.1416/180.
     return 0.5*kk*A1*pow(1+z,1.)* pow(eta, -1./3. ) * pow(E1, 1./3.) * pow(dtheta1,(delta+6.)/3.)*1/a.day

    def Em_bb_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta,dtheta): #This module calculates the characteristic frecuency and give the value in Hz
     A1=pow(2.,1./2)*a.e*pow(a.mp,5./2.)/(pow(mt.pi,1./2)*pow(a.me,3.)) * pow((alpha-2)/(alpha-1),2.0)
     A2=pow(3./(32.*mt.pi*a.mp),4./(delta+8.))
     return 3.* A1*A2*pow(1+z,(4.-delta)/(delta+8)) *pow(xief,2.0)* pow(xiBf,1.0/2.0)*pow(eta, delta/(2*(delta+8))) * pow(E1, 4./(delta+8))* pow(t,-12./(delta+8.))*1/a.GHz*1/1.48

    def Ec_bb_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):#This module calculates the cut off frecuency and give the value in Hz
     A1=pow(2.,1./2.)*a.e*pow(a.me,1.)/(128. * pow(mt.pi,1./2.)* pow(a.mp,3./2.)*pow(a.sigmaT,2.0))
     A2=pow(3./(32.*mt.pi*a.mp), -4./(delta+8))
     xf=xff_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)
     t0=0.
     return A1*A2*pow(1+z,(delta-4.)/(delta+8.))*pow(1+xf,-2.0) * pow(xiBf,-3.0/2.0) * pow(eta,-(16.+3*delta)/(2*(delta+8))) *pow(E1,-4./(delta+8))*pow(t-t0,-2.*(delta+2.)/(delta+8))*1/a.KeV *1.48#
    
    def Fmax_bb_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta):#This module calculates the cut off frecuency and give the value in Hz
     A1= pow(2.,1./2) *a.me*a.sigmaT/(12.*pow(mt.pi,1./2.) *pow(a.mp,1./2.)*a.e)
     A2=pow(3./(32.*mt.pi*a.mp), -delta/(delta+8))
     return A1*A2*pow(1+z,-4.*(delta+2.)/(delta+8.)) * pow(xiBf,1.0/2.0) * pow(eta,(3.*delta+8.)/(2*(delta+8))) * pow(D,-2.)* pow(E1,8./(delta+8))*pow(t,3.*delta/(delta+8))*1./a.mJy*1/pow(1.48,2.)

    def Em_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1): #This module calculates the characteristic frecuency and give the value in Hz
     A1=pow(2.,1./2)*a.e*pow(a.mp,5./2.)/(pow(mt.pi,1./2)*pow(a.me,3.)) * pow((alpha-2)/(alpha-1),2.0)
     A2=pow(3./(32.*mt.pi*a.mp),4./(delta1+6.))
     return 3.*A1*A2*pow(1+z,(6-delta1)/(delta1+6.))*pow(xief,2.)*pow(xiBf,1.0/2.0)*pow(eta, (delta1-2.)/(2.*(delta1+6.))) * pow(E1, 4./(delta1+6.)) * pow(t,-12./(delta1+6.))*1/a.GHz 

    def Ec_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1): #This module calculates the characteristic frecuency and give the value in Hz
     A1=pow(2.,1./2.)*a.e*pow(a.me,1.)/(128. * pow(mt.pi,1./2.)* pow(a.mp,3./2.)*pow(sigmaT,2.0))
     A2=pow(3./(32.*mt.pi*a.mp), -4./(delta1+6))
     return A1*A2*pow(1+z, (delta1-6.)/(delta1+6.))*pow(xiBf,-3.0/2.0)*pow(eta,-(3.*delta1 + 10.)/(2.*(delta1+6.)) ) * pow(E1, -4./(delta1+6.)) * pow(1+xf,-2.) * pow(t,-(2.*delta1)/(delta1+6.))*1/a.KeV

    def Fmax_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1):#This module calculates the cut off frecuency and give the value in Hz
     A1= pow(2.,1./2) *a.me*a.sigmaT/(12.*pow(mt.pi,1./2.) *pow(a.mp,1./2.)*a.e)
     A2=pow(3./(32.*mt.pi*a.mp), (2.-delta1)/(delta1+6))
     return A1*A2*pow(1+z, (12.- 2.*delta1)/(delta1+6.))*pow(xiBf,1.0/2.0)*pow(eta, (3.*delta1+2.)/(2.*(delta1+6.))) * pow(D,-2.0)*pow(E1, 8./(delta1+6.))*pow(t, -3*(2.-delta1)/(delta1+6.))*1/a.mJy



########################## Fit Model ##########################

    td_jet = pm.Deterministic('tdj', tdx_jet(xiBf,xief,E_off,alpha,eta,D,z,tr3,delta,dtheta))
    #E_m_jet = pm.Deterministic('emj', Em_bb_jet(xiBf,xief,gama0b,E,alpha,eta,D,z,t,delta,dtheta,theta_j)*a.GHz)
    #E_c_jet = pm.Deterministic('ecj', Ec_bb_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)*a.KeV)
    #Fmax_jet = pm.Deterministic('fmj', Fmax_bb_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta,theta_j)*a.mJy)
    #E_c_R_jet = pm.Deterministic('ecrj', Ec_ab_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta)*a.KeV)
    #E_m_R_jet = pm.Deterministic('emrj', Em_ab_jet(xiBf,xief,gama0b,E,alpha,eta,D,z,t,delta)*a.GHz)
    #Fmax_R_jet = pm.Deterministic('fmrj', Fmax_ab_jet(xiBf,xief,E,alpha,eta,D,z,t,delta,dtheta)*a.mJy)
    td_coc = pm.Deterministic('tdc',tdx_coc(xiBf,xief,E0,alpha,eta,D,z,tr3,delta,dtheta))
    #E_m_coc=pm.Deterministic('emc',Em_bb_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta,dtheta)*a.GHz)
    #Fmax_coc=pm.Deterministic('fmc',Fmax_bb_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)*a.mJy)
    #E_c_coc=pm.Deterministic('ecc',Ec_bb_coc(xiBf,xief,E1,alpha,eta,D,z,t,delta,dtheta)*a.KeV)
    #E_m_R_coc=pm.Deterministic('emrc',Em_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1)*a.GHz)
    #E_c_R_coc=pm.Deterministic('ecrc',Ec_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1)*a.KeV)
    #Fmax_R_coc=pm.Deterministic('emrc',Fmax_ab_coc(xiBf,xief,gama0b,E1,alpha,eta,D,z,t,delta1)*a.mJy)



    if((tr3<td_jet).any()):
     Egammam_sc = 3 * a.GHz
     E_m_jet=pm.Deterministic('E_m_jet',Em_bb_jet(xiBf,xief,gama0b,E_off,alpha,eta,D,z,tr3,delta,dtheta,theta_j)*a.GHz)
     E_c_jet=pm.Deterministic('E_c_jet',Ec_bb_jet(xiBf,xief,E_off,alpha,eta,D,z,tr3,delta,dtheta,theta_j)*a.KeV)
     Fmax_jet=pm.Deterministic('F_max_jet',Fmax_bb_jet(xiBf,xief,E_off,alpha,eta,D,z,tr3,delta,dtheta,theta_j)*a.mJy*1)
     Fm_sc_Radio3=pm.Deterministic('Fm_sc_Radio3',Fmax_jet*pow(Egammam_sc/E_m_jet,-(alpha-1.)/2.)*1./a.micJy)
     Fm_sc_Radio3_obs = pm.Normal('Fm_sc_Radio3_obs', mu=Fm_sc_Radio3, sd=sigma, observed=F_r3g)
    else:
     Egammam_sc_R=3 * a.GHz
     E_m_R_jet=pm.Deterministic('E_m_R_jet',Em_ab_jet(xiBf,xief,gama0b,E_off,alpha,eta,D,z,tr3,delta)*a.GHz)
     E_c_R_jet=pm.Deterministic('E_c_R_jet',Ec_ab_jet(xiBf,xief,E_off,alpha,eta,D,z,tr3,delta,dtheta)*a.KeV)
     Fmax_R_jet=pm.Deterministic('F_max_R_jet',Fmax_ab_jet(xiBf,xief,E_off,alpha,eta,D,z,tr3,delta,dtheta)*a.mJy*1.)
     Fm_sc_Radio3=pm.Deterministic('Fm_sc_Radio3',Fmax_R_jet*pow(Egammam_sc_R/E_m_R_jet,-(alpha-1.)/2.)*1./a.micJy)
     Fm_sc_Radio3_obs = pm.Normal('Fm_sc_Radio3_obs', mu=Fm_sc_Radio3, sd=sigma, observed=F_r3g)
    
   
    if((tr3<td_coc).any()):
     Egammam_sc = 3 * a.GHz
     E_m_coc=pm.Deterministic('E_m_coc',Em_bb_coc(xiBf,xief,gama0b,E0,alpha,eta,D,z,tr3,delta,dtheta)*a.GHz)
     E_c_coc=pm.Deterministic('E_c_coc',Ec_bb_coc(xiBf,xief,E0,alpha,eta,D,z,tr3,delta,dtheta)*a.KeV)
     Fmax_coc=pm.Deterministic('Fmax_coc',Fmax_bb_coc(xiBf,xief,E0,alpha,eta,D,z,tr3,delta,dtheta)*a.mJy)
     Fm_sc_Radio3_c=pm.Deterministic('Fm_sc_Radio3_c',Fmax_coc*pow(Egammam_sc/E_m_coc,-(alpha-1.)/2.)*1./a.micJy)
     Fm_sc_Radio3_obs = pm.Normal('Fm_sc_Radio3_obs_c', mu=Fm_sc_Radio3_c, sd=sigma, observed=F_r3g)
    else:
     Egammam_sc_R= 3 * a.GHz
     E_m_R_coc=pm.Deterministic('E_m_R_coc',Em_ab_coc(xiBf,xief,gama0b,E0,alpha,eta,D,z,tr3,delta)*a.GHz)
     E_c_R_coc=pm.Deterministic('E_c_R_coc',Ec_ab_coc(xiBf,xief,gama0b,E0,alpha,eta,D,z,tr3,delta)*a.KeV)
     Fmax_R_coc=pm.Deterministic('Fmax_R_coc',Fmax_ab_coc(xiBf,xief,gama0b,E0,alpha,eta,D,z,tr3,delta)*a.mJy)
     Fm_sc_Radio3_c=pm.Deterministic('Fm_sc_Radio3_c',Fmax_R_coc*pow(Egammam_sc_R/E_m_R_coc,-(alpha-1.)/2.) * 1./a.micJy)
     Fm_sc_Radio3_obs = pm.Normal('Fm_sc_Radio3_obs_c', mu=Fm_sc_Radio3_c, sd=sigma, observed=F_r3g)
    

    
########################## Step Definition ##########################

    step1 = pm.NUTS(target_accept=0.99)

########################## Sampling ##########################

    trace = pm.sample(niter, step=[step1], init="adapt_diag",tune=tune, random_seed = 123)
    aa    = pm.backends.tracetab.trace_to_dataframe(trace, varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
    aa.to_csv(path_or_buf = "Output/Radio3GHz/output.csv", sep="\t")
    summary=pm.stats.summary(trace,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
    summary.to_csv(path_or_buf = "Output/Radio3GHz/summary.csv", sep="\t")
    print(summary)
        
    
with open('Output/X-Ray/Trace/my_model.pkl', 'wb') as buff:
    pickle.dump({'model': X_ray_model, 'trace_xray': trace}, buff)

with open('Output/Radio6GHz/Trace/my_model.pkl', 'wb') as buff:
    pickle.dump({'model': Radio6GHz_model, 'trace_radio6ghz': trace}, buff)

with open('Output/Radio3GHz/Trace/my_model.pkl', 'wb') as buff:
    pickle.dump({'model': Radio3GHz_model, 'trace_radio3ghz': trace}, buff)
