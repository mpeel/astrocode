from scipy import optimize
from scipy import odr
import numpy as np

def linfit(x, param):
	return param[0]*x+param[1]

def compute_residuals_linfit(param, x, y):
	model = linfit(x, param)
	residual = y - model
	return residual

def linfit2(x, A, B):
	return A*x+B

def linfit3(param, x):
	return param[0]*x+param[1]

def plot_tt(vals1,vals2,outputname,sigma=np.empty(0),sigma_x=np.empty(0),leastsq=False,freq1=0,freq2=0,xlabel='',ylabel='',xyline=False):
	if len(vals1) == 0 or len(vals2) == 0:
		print('No data! Aborting')
		return [[1.0,0.0], [0.0,0.0]]
	# print(sigma)
	# Do a fit
	params = [1.0,0]
	if leastsq:
		# least-squares fit without uncertainties
		param_est, cov_x, infodict, mesg_result, ret_value = optimize.leastsq(compute_residuals_linfit, params, args=(vals1, vals2),full_output=True)
		sigma_param_est = np.sqrt(np.diagonal(cov_x))
	elif sigma_x.size:
		# Do an odr fit
		odr_model = odr.Model(linfit3)
		dataset = odr.Data(vals1, vals2, wd=1.0/sigma_x**2, we=1.0/sigma**2)
		odr_run = odr.ODR(dataset, odr_model, beta0=params)
		out = odr_run.run()
		param_est = out.beta
		sigma_param_est = out.sd_beta
	elif sigma.size:
		# Do a curve fit to use the uncertainties
		param_est, cov_x = optimize.curve_fit(linfit2, vals1, vals2, params, sigma=sigma)
		sigma_param_est = np.sqrt(np.diagonal(cov_x))
	else:
		# Do a curve fit without uncertainties
		param_est, cov_x = optimize.curve_fit(linfit2, vals1, vals2, params)
		sigma_param_est = np.sqrt(np.diagonal(cov_x))

	#Plot the data and the results
	if sigma_x.size:
		plt.errorbar(vals1,vals2,yerr=sigma,xerr=sigma_x,fmt='.')
	elif sigma.size:
		plt.errorbar(vals1,vals2,yerr=sigma,fmt='.')
	else:
		plt.plot(vals1,vals2,'.')
	xvals=np.arange(np.min(vals1),np.max(vals1),(np.max(vals1)-np.min(vals1))/100.0)
	if np.isfinite(sigma_param_est[0]) and np.isfinite(sigma_param_est[1]):
		mesg_fit = (
		r'$A={:5.3e}\pm{:3.2e}$'.format(
			param_est[0], sigma_param_est[0]) + ','
		r'$B={:5.3e}\pm{:3.2e}$'.format(
			param_est[1], sigma_param_est[1]))# + ','
		if freq1 != freq2:
			beta = -np.log(param_est[0])/np.log(freq1/freq2)
			s_beta = sigma_param_est[0]/param_est[0] / np.abs(np.log(freq1/freq2))
			mesg_fit = mesg_fit + ', beta = ${:5.3f}\pm{:3.2f}$'.format(beta,s_beta)
			# print(beta,s_beta)
		plt.plot(xvals,linfit(xvals,param_est),'g',label="Fit: " + mesg_fit)
	else:
		plt.plot(xvals,linfit(xvals,param_est),'g')
	if xyline:
		plt.plot(vals1,vals1,label='X=Y')
	plt.legend(prop={'size':8})
	if xlabel != '':
		plt.xlabel(xlabel)
	if ylabel != '':
		plt.ylabel(ylabel)
	plt.savefig(outputname)
	plt.clf()
	plt.close()
	return [param_est, sigma_param_est]

def calc_std_over_n(vals):
	return np.std(vals)/np.sqrt(len(vals))
