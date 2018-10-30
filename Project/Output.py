import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import pandas as pan
import corner
import numpy as np
import a
import math as mt 
import warnings
import pickle

plt.style.use('seaborn-darkgrid')

with warnings.catch_warnings():
  warnings.filterwarnings("ignore", category=FutureWarning)
  warnings.filterwarnings("ignore", category= DeprecationWarning)


######################################## Global Variables ######################################

E_j = r'$E_{off} \ (10^{49}erg)$'
E_c = r'$\tilde{E} \ (10^{49}erg)$'
n = r'$n \ (10^{-4}cm^{-3})$'
eb = r'$\epsilon_B\ (10^{-4})$'
ee = r'$\epsilon_e \ (10^{-1}) $'
theta = r'$\theta_j \ (deg)$'
dtheta = r'$\Delta\theta \ (deg)$'
p = r'$p$'
alpha =r'$\alpha_S$'
order = [E_j,E_c, n, p, theta, dtheta, eb, ee, alpha]
ndim = 9



######################################## Dataframe Creations ####################################

######################################## X-Ray  ####################################

x_ray = pan.read_csv("Output/X-Ray/output.csv",sep="\t")
x_ray = x_ray.drop(x_ray.columns[0], axis=1)
x_ray = x_ray.reindex(columns=order)

######################################## Optical ###################################
"""
optical = pan.read_csv("Output/Optical/output.csv",sep="\t")
optical = optical.drop(optical.columns[0], axis=1)
optical = optical.reindex(columns=order)

######################################## Radio  ####################################
"""
radio3GHz = pan.read_csv("Output/Radio3GHz/output.csv",sep="\t")
radio3GHz = radio3GHz.drop(radio3GHz.columns[0], axis=1)
radio3GHz = radio3GHz.reindex(columns=order)

radio6GHz = pan.read_csv("Output/Radio6GHz/output.csv",sep="\t")
radio6GHz = radio6GHz.drop(radio6GHz.columns[0], axis=1)
radio6GHz = radio6GHz.reindex(columns=order)

######################################## Summaries ################################

summary_x_ray = pan.read_csv("Output/X-Ray/summary.csv", sep="\t")
summary_x_ray = summary_x_ray.rename(index= {0:summary_x_ray.loc[0,'Unnamed: 0'],1:summary_x_ray.loc[1,'Unnamed: 0'],2:summary_x_ray.loc[2,'Unnamed: 0'],
  3:summary_x_ray.loc[3,'Unnamed: 0'],4:summary_x_ray.loc[4,'Unnamed: 0'],5:summary_x_ray.loc[5,'Unnamed: 0'],
  6:summary_x_ray.loc[6,'Unnamed: 0'],7:summary_x_ray.loc[7,'Unnamed: 0'],8:summary_x_ray.loc[8,'Unnamed: 0']})
summary_x_ray = summary_x_ray.drop(summary_x_ray.columns[0], axis=1)
summary_x_ray = summary_x_ray.reindex(index=order)
"""
summary_optical = pan.read_csv("Output/Optical/summary.csv", sep="\t")
summary_optical = summary_optical.rename(index= {0:summary_optical.loc[0,'Unnamed: 0'],1:summary_optical.loc[1,'Unnamed: 0'],2:summary_optical.loc[2,'Unnamed: 0'],
  3:summary_optical.loc[3,'Unnamed: 0'],4:summary_optical.loc[4,'Unnamed: 0'],5:summary_optical.loc[5,'Unnamed: 0'],
  6:summary_optical.loc[6,'Unnamed: 0']})
summary_optical = summary_optical.drop(summary_optical.columns[0], axis=1)
summary_optical = summary_optical.reindex(index=order)
"""
summary_radio3GHz = pan.read_csv("Output/Radio3GHz/summary.csv", sep="\t")
summary_radio3GHz = summary_radio3GHz.rename(index= {0:summary_radio3GHz.loc[0,'Unnamed: 0'],1:summary_radio3GHz.loc[1,'Unnamed: 0'],2:summary_radio3GHz.loc[2,'Unnamed: 0'],
  3:summary_radio3GHz.loc[3,'Unnamed: 0'],4:summary_radio3GHz.loc[4,'Unnamed: 0'],5:summary_radio3GHz.loc[5,'Unnamed: 0'],
  6:summary_radio3GHz.loc[6,'Unnamed: 0'],7:summary_radio3GHz.loc[7,'Unnamed: 0'],8:summary_radio3GHz.loc[8,'Unnamed: 0']})
summary_radio3GHz = summary_radio3GHz.drop(summary_radio3GHz.columns[0], axis=1)
summary_radio3GHz = summary_radio3GHz.reindex(index=order)

summary_radio6GHz = pan.read_csv("Output/Radio6GHz/summary.csv", sep="\t")
summary_radio6GHz = summary_radio6GHz.rename(index= {0:summary_radio6GHz.loc[0,'Unnamed: 0'],1:summary_radio6GHz.loc[1,'Unnamed: 0'],2:summary_radio6GHz.loc[2,'Unnamed: 0'],
  3:summary_radio6GHz.loc[3,'Unnamed: 0'],4:summary_radio6GHz.loc[4,'Unnamed: 0'],5:summary_radio6GHz.loc[5,'Unnamed: 0'],
  6:summary_radio6GHz.loc[6,'Unnamed: 0'],7:summary_radio6GHz.loc[7,'Unnamed: 0'],8:summary_radio6GHz.loc[8,'Unnamed: 0']})
summary_radio6GHz = summary_radio6GHz.drop(summary_radio6GHz.columns[0], axis=1)
summary_radio6GHz = summary_radio6GHz.reindex(index=order)


######################################## Trace Retrieval  ###############################
with open('Output/X-Ray/Trace/my_model.pkl', 'rb') as buff:
    data = pickle.load(buff)  
X_ray_model, trace_xray = data['model'], data['trace_xray']

with open('Output/Radio6GHz/Trace/my_model.pkl', 'rb') as buff:
    data = pickle.load(buff)  
Radio6GHz_model, trace_radio6ghz = data['model'], data['trace_radio6ghz']

with open('Output/Radio3GHz/Trace/my_model.pkl', 'rb') as buff:
    data = pickle.load(buff)  
Radio3GHz_model, trace_radio3ghz = data['model'], data['trace_radio3ghz']

"""
with open('Output/Optical/Trace/mmy_model.pkl', 'rb') as buff:
    data = pickle.load(buff)  
Optical_model, trace_optical = data['model'], data['trace_optical']
"""
######################################## Cornerplots ####################################

x_ray_cornerplot = corner.corner(x_ray, quantiles=[0.15, 0.50, 0.85],
 range=[0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95], label_kwargs={"fontsize": 15},
 show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 12})

value1=summary_x_ray.loc[:,"mean"]

axes = np.array(x_ray_cornerplot.axes).reshape((ndim, ndim)) #Extract the axes

for i in range(ndim):     #Loop over the diagonal
  ax = axes[i, i]
  ax.axvline(value1[i], color="g")

for yi in range(ndim):    #Loop over the histograms
  for xi in range(yi):
   ax = axes[yi, xi]
   ax.axvline(value1[xi], color="g")
   ax.axhline(value1[yi], color="g")
plt.savefig("Output/Plots/Corner/Cornerplot_X_ray.png")
plt.close()
"""
optical_cornerplot = corner.corner(optical, quantiles=[0.15, 0.50, 0.85],
  range=[0.95,0.95,0.95,0.95,0.95,0.95,0.95], label_kwargs={"fontsize": 15}, 
  show_titles=True,title_fmt='.3f', title_kwargs={"fontsize": 12})

value1=summary_optical.loc[:,"mean"]

axes = np.array(optical_cornerplot.axes).reshape((ndim, ndim)) #Extract the axes
for i in range(ndim):     #Loop over the diagonal
  ax = axes[i, i]
  ax.axvline(value1[i], color="g")

for yi in range(ndim):    #Loop over the histograms
  for xi in range(yi):
   ax = axes[yi, xi]
   ax.axvline(value1[xi], color="g")
   ax.axhline(value1[yi], color="g")
plt.savefig("Output/Plots/Corner/Cornerplot_Optical.png")
plt.close()
"""
radio3ghz_cornerplot = corner.corner(radio3GHz, quantiles=[0.15, 0.50, 0.85],
  range=[0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95], label_kwargs={"fontsize": 15},
  show_titles=True,title_fmt='.3f', title_kwargs={"fontsize": 12})

value1=summary_radio3GHz.loc[:,"mean"]

axes = np.array(radio3ghz_cornerplot.axes).reshape((ndim, ndim)) #Extract the axes

for i in range(ndim):     #Loop over the diagonal
  ax = axes[i, i]
  ax.axvline(value1[i], color="g")

for yi in range(ndim):    #Loop over the histograms
  for xi in range(yi):
   ax = axes[yi, xi]
   ax.axvline(value1[xi], color="g")
   ax.axhline(value1[yi], color="g")
plt.savefig("Output/Plots/Corner/Cornerplot_Radio3GHz.png")
plt.close()

radio6ghz_cornerplot = corner.corner(radio6GHz, quantiles=[0.15, 0.50, 0.85],
  range=[0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95], label_kwargs={"fontsize": 15},
  show_titles=True,title_fmt='.3f', title_kwargs={"fontsize": 12})

value1=summary_radio6GHz.loc[:,"mean"]

axes = np.array(radio6ghz_cornerplot.axes).reshape((ndim, ndim)) #Extract the axes
for i in range(ndim):     #Loop over the diagonal
  ax = axes[i, i]
  ax.axvline(value1[i], color="g")

for yi in range(ndim):    #Loop over the histograms
  for xi in range(yi):
   ax = axes[yi, xi]
   ax.axvline(value1[xi], color="g")
   ax.axhline(value1[yi], color="g")
plt.savefig("Output/Plots/Corner/Cornerplot_Radio6GHz.png")
plt.close()

######################################## Tracerplots #################################
x_ray_traceplot = pm.plots.traceplot(trace_xray,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
plt.savefig("Output/Plots/Trace/Traceplot_X-Ray.png")
plt.close()
"""
optical_traceplot = pm.plots.traceplot(trace_optical)
plt.savefig("Output/Plots/Trace/Traceplot_Optical.png")
plt.close()
"""
radio3ghz_traceplot = pm.plots.traceplot(trace_radio3ghz,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
plt.savefig("Output/Plots/Trace/Traceplot_Radio3GHz.png")
plt.close()

radio6ghz_traceplot = pm.plots.traceplot(trace_radio6ghz,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
plt.savefig("Output/Plots/Trace/Traceplot_Radio6GHz.png")
plt.close()

######################################## Posterior ##################################
x_ray_posterior = pm.plots.plot_posterior(trace_xray,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
plt.savefig("Output/Plots/Posterior/Posteior_X-Ray.png")
plt.close()
"""
optical_posterior = pm.plots.plot_posterior(trace_optical)
plt.savefig("Output/Plots/Posterior/Posterior_Optical.png")
plt.close()
"""
radio3ghz_posterior = pm.plots.plot_posterior(trace_radio3ghz,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
plt.savefig("Output/Plots/Posterior/Posterior_Radio3GHz.png")
plt.close()

radio6ghz_posterior = pm.plots.plot_posterior(trace_radio6ghz,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
plt.savefig("Output/Plots/Posterior/Posterior_Radio6GHz.png")
plt.close()

######################################## Forest ####################################
x_ray_forestplot = pm.plots.forestplot(trace_xray,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
plt.savefig("Output/Plots/Forest/Forestplot_X-Ray.png")
plt.close()
"""
optical_forestplot = pm.plots.forestplot(trace_optical)
plt.savefig("Output/Plots/Forest/Forestplot_Optical.png")
plt.close()
"""
radio3ghz_forestplot = pm.plots.forestplot(trace_radio3ghz,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
plt.savefig("Output/Plots/Forest/Forestplot_Radio3GHz.png")
plt.close()

radio6ghz_forestplot = pm.plots.forestplot(trace_radio6ghz,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
plt.savefig("Output/Plots/Forest/Forestplot_Radio6GHz.png")
plt.close()


######################################## Auto Correlation ############################
x_ray_autocorrplot = pm.plots.autocorrplot(trace_xray,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
plt.savefig("Output/Plots/AutoCorrelation/AutoCorrelation_X-Ray.png")
plt.close()
"""
optical_autocorrplot = pm.plots.autocorrplot(trace_optical)
plt.savefig("Output/Plots/AutoCorrelation/AutoCorrelation_Optical.png")
plt.close()
"""
radio3ghz_autocorrplot = pm.plots.autocorrplot(trace_radio3ghz,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
plt.savefig("Output/Plots/AutoCorrelation/AutoCorrelation_Radio3GHz.png")
plt.close()

radio6ghz_autocorrplot = pm.plots.autocorrplot(trace_radio6ghz,varnames={r'$\tilde{E} \ (10^{49}erg)$',r'$E_{off} \ (10^{49}erg)$',r'$\Delta\theta \ (deg)$',r'$n \ (10^{-4}cm^{-3})$',r'$\theta_j \ (deg)$',r'$p$',r'$\alpha_S$',r'$\epsilon_B\ (10^{-4})$',r'$\epsilon_e \ (10^{-1}) $'})
plt.savefig("Output/Plots/AutoCorrelation/AutoCorrelation_Radio6GHz.png")
plt.close()
