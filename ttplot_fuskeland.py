# Code by Federica
import healpy as hp
import numpy as np
import pylab as plt
from scipy import stats
from scipy.odr import *
import math
import numpy.ma as ma
from fastcc.fastcc import fastcc

nsigma = 0.0

# Define a linear function to fit the data with.
def linear_func(p, x):
	m, c = p
	return m*x + c

def linear_func_zerooffset(p, x):
	m = p
	return m*x

# Define a quadratic function to fit the data with.
def quad_func(p, x):
	a, b, c = p
	return a*x**2 + b*x + c

def linear_fit_zerooffset(d1, d2, err_d1, err_d2):
	# Create a model for fitting.
	linear_model = Model(linear_func_zerooffset)
	# Create a RealData object using our initiated data from above.
	data = RealData(d1, d2, sx=err_d1, sy=err_d2)
	# Set up ODR with the model and data.
	odr = ODR(data, linear_model, beta0=[0.])
	# Run the regression.
	out = odr.run()
	# Use the in-built pprint method to give us results.
	m = out.beta[0]
	err_m =  out.sd_beta[0]
	q = 0
	err_q = 0
	return(m, q, err_m, err_q)

def linear_fit(d1, d2, err_d1, err_d2):
	# Create a model for fitting.
	linear_model = Model(linear_func)
	# Create a RealData object using our initiated data from above.
	data = RealData(d1, d2, sx=err_d1, sy=err_d2)
	# Set up ODR with the model and data.
	odr = ODR(data, linear_model, beta0=[0., 0.])
	# Run the regression.
	out = odr.run()
	# Use the in-built pprint method to give us results.
	q = out.beta[1]
	err_q = out.sd_beta[1]
	m = out.beta[0]
	err_m =  out.sd_beta[0]
	print(m,q)
	return(m, q, err_m, err_q)

def linear_fit_cc(d1, d2, err_d1, err_d2, freq1, freq2, strfreq1, strfreq2, detector1, detector2, thresh_beta):
	dbeta = np.inf
	i = 0

	#First linear fit
	m, q, err_m, err_q = linear_fit(d1, d2, err_d1, err_d2)
	beta0 = np.log(m)/np.log(freq2/freq1)
	errbeta0 = 1./(np.log(freq2/freq1))/m*err_m
	print('Initial beta and error: ', beta0, errbeta0)
	alpha = beta0+2.

	while dbeta > thresh_beta:
		#Apply CC
		if (strfreq1 == 'Q11' or  strfreq1 =='Q13' or strfreq1 == 'Q11p' or  strfreq1 =='Q13p'):
			cc1 = fastcc(strfreq1, alpha, detector = detector1)
		else:
			cc1 = fastcc(strfreq1, alpha)

		if (strfreq2 == 'Q11' or  strfreq2 =='Q13' or strfreq2 == 'Q11p' or  strfreq2 =='Q13p') :
			cc2 = fastcc(strfreq2, alpha, detector = detector2)
		else:
			cc2 = fastcc(strfreq2, alpha)

		print('Color correction 1 : ', strfreq1, detector1, alpha, cc1)
		print('Color correction 2 : ', strfreq2, detector2, alpha, cc2)

		#Make linear fit
		m, q, err_m, err_q = linear_fit(d1*cc1, d2*cc2, err_d1*cc1, err_d2*cc2)
		#Compute spectral index
		beta1 = np.log(m)/np.log(freq2/freq1)
		aplha = beta1+2.
		dbeta = np.abs(beta0-beta1)
		beta0 = beta1
		i = i+1
		print('Iterations number: ', i)
		print('Beta : ', beta0)

	beta_final = beta0
	errbeta_final = 1./(np.log(freq2/freq1))/m*err_m
	print('Final beta and error: ', beta_final, errbeta_final)

	return(beta_final, errbeta_final, q, err_q, d1*cc1, d2*cc2, err_d1*cc1, err_d2*cc2)

def linear_fit_zerooffset_cc(d1, d2, err_d1, err_d2, freq1, freq2, strfreq1, strfreq2, detector1, detector2, thresh_beta):
	dbeta = np.inf
	i = 0

	#First linear fit
	m, q, err_m, err_q = linear_fit_zerooffset(d1, d2, err_d1, err_d2)
	beta0 = np.log(m)/np.log(freq2/freq1)
	errbeta0 = 1./(np.log(freq2/freq1))/m*err_m
	print('Initial beta and error: ', beta0, errbeta0)
	alpha = beta0+2.

	while dbeta > thresh_beta:

		#Apply CC
		if (strfreq1 == 'Q11' or  strfreq1 =='Q13' or strfreq1 == 'Q11p' or  strfreq1 =='Q13p'):
			cc1 = fastcc(strfreq1, alpha, detector = detector1)
		else:
			cc1 = fastcc(strfreq1, alpha)

		if (strfreq2 == 'Q11' or  strfreq2 =='Q13' or strfreq2 == 'Q11p' or  strfreq2 =='Q13p') :
			cc2 = fastcc(strfreq2, alpha, detector = detector2)
		else:
			cc2 = fastcc(strfreq2, alpha)
		print('Color correction 1 : ', strfreq1, detector1, alpha, cc1)
		print('Color correction 2 : ', strfreq2, detector2, alpha, cc2)

		#Make linear fit
		m, q, err_m, err_q = linear_fit_zerooffset(d1*cc1, d2*cc2, err_d1*cc1, err_d2*cc2)
		#Compute spectral index
		beta1 = np.log(m)/np.log(freq2/freq1)
		aplha = beta1+2.
		dbeta = np.abs(beta0-beta1)
		beta0 = beta1
		i = i+1
		print('Iterations number: ', i)
		print('Beta : ', beta0)

	beta_final = beta0
	errbeta_final = 1./(np.log(freq2/freq1))/m*err_m
	print('Final beta and error: ', beta_final, errbeta_final)

	return(beta_final, errbeta_final, q, err_q, d1*cc1, d2*cc2, err_d1*cc1, err_d2*cc2)


def quad_fit(d1, d2, err_d1, err_d2):
	# Create a model for fitting.
	quad_model = Model(quad_func)
	# Create a RealData object using our initiated data from above.
	data = RealData(d1, d2, sx=err_d1, sy=err_d2)
	# Set up ODR with the model and data.
	odr = ODR(data, quad_model, beta0=[0., 0., 0.])
	# Run the regression.
	out = odr.run()
	# Use the in-built pprint method to give us results.
	c = out.beta[2]
	err_c = out.sd_beta[2]
	b = out.beta[1]
	err_b = out.sd_beta[1]
	a = out.beta[0]
	err_a =  out.sd_beta[0]
	fit = np.array([a, b, c])
	err_fit = np.array([err_a, err_b, err_c])
	return(fit, err_fit)


def red_chi2(datax, sigmax, datay, sigmay, p):
	chi = np.sum((datay-linear_func(p,datax))**2/(sigmay**2+(p[0]*sigmax)**2))
	chi_red = chi/(np.size(datay)-np.size(p))
	return(chi_red)

def p_fusk14(mask_in, map1, var_map1, freq1, map2, var_map2, freq2, nsigma, path_out, label):
	alpha_deg = np.arange(0,90,step=5)
	alpha = alpha_deg*np.pi/180.
	beta = np.zeros(np.size(alpha))
	sigma_beta = np.zeros(np.size(alpha))
	q = np.zeros(np.size(alpha))
	sigmaq = np.zeros(np.size(alpha))
	chi2 = np.zeros(np.size(alpha))

	#fig = plt.figure(figsize=(20, 6))

	for i in range(np.size(alpha)):

		QcosUsin1 = map1[1,:]*np.cos(2.*alpha[i])+map1[2,:]*np.sin(2.*alpha[i])
		QcosUsin2 = map2[1,:]*np.cos(2.*alpha[i])+map2[2,:]*np.sin(2.*alpha[i])

		#hp.mollview(QcosUsin1, 'QcosUsin1')
		#hp.mollview(QcosUsin2, 'QcosUsin2')
		#plt.show()
		err1 = np.sqrt(var_map1[1,:]*(np.cos(2.*alpha[i]))**2+var_map1[2,:]*(np.sin(2.*alpha[i]))**2)
		err2 = np.sqrt(var_map2[1,:]*(np.cos(2.*alpha[i]))**2+var_map2[2,:]*(np.sin(2.*alpha[i])**2))
		SN1 = (np.abs(QcosUsin1)/err1 > nsigma)
		SN2 = (np.abs(QcosUsin2/err2) > nsigma)
		goodpoints = np.all([(mask_in == 1.), SN1, SN2,
							 (map1[1,:] != hp.pixelfunc.UNSEEN), (map1[2,:] != hp.pixelfunc.UNSEEN),
							 (map2[1,:] != hp.pixelfunc.UNSEEN), (map2[2,:] != hp.pixelfunc.UNSEEN),
							 (map1[1,:] != 0.), (map1[2,:] != 0.),
							 (map2[1,:] != 0.), (map2[2,:] != 0.),
							 (var_map1[1,:] != 0.), (var_map1[2,:] != 0.),
							 (var_map2[1,:] != 0.), (var_map2[2,:] != 0.)], axis=0)
		#goodpoints = np.all([(mask == 1.), SN1, SN2, (np.abs(QcosUsin1)/err1 > nsigma), (np.abs(QcosUsin2)/err2 > nsigma)], axis=0)
		beta[i], sigma_beta[i], q[i], sigmaq[i] = make_tt_plot_goodpoints(QcosUsin1[goodpoints], QcosUsin2[goodpoints], err1[goodpoints], err2[goodpoints], freq1, freq2)
		sigma_beta[i] = np.abs(sigma_beta[i])
		chi2[i] = red_chi2(QcosUsin1[goodpoints], err1[goodpoints], QcosUsin2[goodpoints], err2[goodpoints], [(freq1/freq2)**beta[i], q[i]])

		if (np.isnan(beta[i]) or beta[i] > 0.):
			beta[i] = 0.
			sigma_beta[i] = np.inf
			q[i] = 0.
			sigmaq[i] = np.inf
			chi2[i] = np.inf

		#label(alpha_deg[i])

		print(beta[i], sigma_beta[i], chi2[i])
		#if (i== np.size(alpha_deg)-1):
			#plt.legend()
			#plt.savefig(path_out+'FSK_TTplot_'+label+'.png')
			#plt.close()
	#Mask arrays
	mask = beta == 0.0
	if(np.sum(mask) != np.size(alpha_deg)):

	#if(np.sum(mask) > 0):
		beta_ma = ma.masked_array(beta, mask=mask)
		sigma_beta_ma = ma.masked_array(sigma_beta, mask=mask)
		q_ma = ma.masked_array(q, mask=mask)
		sigmaq_ma = ma.masked_array(sigmaq, mask=mask)
	#if(np.sum(mask) != np.size(alpha_deg)):
		beta_tot = np.sum(beta_ma/(sigma_beta_ma**2))/np.sum(1./sigma_beta_ma**2)
		#sigma_beta_tot =np.sqrt(1./np.sum(1./sigma_beta_ma**2))
		q_tot = np.sum(q_ma/(sigmaq_ma**2))/np.sum(1./sigmaq_ma**2)
		#sigma_q_tot =np.sqrt(1./np.sum(1./(sigmaq_ma**2)))
		min_err_beta = (sigma_beta == np.min(sigma_beta_ma))
		if(np.sum(min_err_beta) != 1):
			print('sum(min_err_beta)', sum(min_err_beta))
			where_first_min_err_beta = np.where(min_err_beta)[0][0]
			min_err_beta[:] = 0
			min_err_beta[where_first_min_err_beta] = 1
			min_err_beta = min_err_beta.astype('int')
			print('sum(min_err_beta)', sum(min_err_beta))
		sigma_beta2 = np.std(beta_ma) #(np.abs(np.max(beta_ma)-np.min(beta_ma))/2.)

		if(1):
			plt.errorbar(alpha_deg, beta, xerr=None, yerr=sigma_beta)
			plt.xlabel(r'$\alpha$ [deg]', fontsize=12)
			plt.ylabel(r'$\beta$', fontsize=12)
			#plt.plot(alpha_deg, np.ones(np.size(alpha))*(-3), color='red')
			plt.plot(alpha_deg, np.ones(np.size(alpha))*beta_tot, color='black')
			plt.axhline(beta_tot, color='black', label=r'$<\beta>$='+str("{0:.2f}".format(beta_tot))+'$\pm$'+str("{0:.2f}".format(sigma_beta2)))
			plt.axhspan(beta_tot-sigma_beta[min_err_beta], beta_tot+sigma_beta[min_err_beta], alpha=0.25, color='black', label=r'$1\sigma$ min(err)')
			plt.axhspan(beta_tot-sigma_beta2, beta_tot+sigma_beta2, alpha=0.15, color='black', label=r'$1\sigma$')
			plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=None,useLocale=None,useMathText=True)

			plt.legend()
			print(sigma_beta)
			print(sigma_beta[min_err_beta])
			#plt.show()
			#plt.close()

	else:
		beta_tot = hp.pixelfunc.UNSEEN
		q_tot = hp.pixelfunc.UNSEEN
		min_err_beta = 0
		sigma_beta2 = hp.pixelfunc.UNSEEN
		sigma_beta3 = hp.pixelfunc.UNSEEN

	sigma_beta2 = np.max([np.abs(sigma_beta[min_err_beta]), sigma_beta2])
	sigma_beta3 = np.sqrt(sigma_beta2**2+sigma_beta[min_err_beta]**2) #(np.abs(np.max(beta_ma)-np.min(beta_ma))/2.)
	print('RESULT: beta_tot, np.min(sigma_beta)', beta_tot, sigma_beta[min_err_beta])
	return(beta_tot, sigma_beta[min_err_beta], sigma_beta2, sigma_beta3, q_tot, sigmaq[min_err_beta],chi2[min_err_beta])


def p_fusk14_cc(mask_in, map1, var_map1, freq1, str_freq1, detector1, map2, var_map2, freq2, str_freq2, detector2, nsigma, path_out, label):

	alpha_deg = np.arange(0,90,step=5)
	alpha = alpha_deg*np.pi/180.
	beta = np.zeros(np.size(alpha))
	sigma_beta = np.zeros(np.size(alpha))
	q = np.zeros(np.size(alpha))
	sigmaq = np.zeros(np.size(alpha))
	chi2 = np.zeros(np.size(alpha))

	#fig = plt.figure(figsize=(20, 6))

	for i in range(np.size(alpha)):

		QcosUsin1 = map1[1,:]*np.cos(2.*alpha[i])+map1[2,:]*np.sin(2.*alpha[i])
		QcosUsin2 = map2[1,:]*np.cos(2.*alpha[i])+map2[2,:]*np.sin(2.*alpha[i])

		#hp.mollview(QcosUsin1, 'QcosUsin1')
		#hp.mollview(QcosUsin2, 'QcosUsin2')
		#plt.show()
		err1 = np.sqrt(var_map1[1,:]*(np.cos(2.*alpha[i]))**2+var_map1[2,:]*(np.sin(2.*alpha[i]))**2)
		err2 = np.sqrt(var_map2[1,:]*(np.cos(2.*alpha[i]))**2+var_map2[2,:]*(np.sin(2.*alpha[i])**2))
		SN1 = (np.abs(QcosUsin1)/err1 > nsigma)
		SN2 = (np.abs(QcosUsin2/err2) > nsigma)
		goodpoints = np.all([(mask_in == 1.), SN1, SN2,
							 (map1[1,:] != hp.pixelfunc.UNSEEN), (map1[2,:] != hp.pixelfunc.UNSEEN),
							 (map2[1,:] != hp.pixelfunc.UNSEEN), (map2[2,:] != hp.pixelfunc.UNSEEN),
							 (map1[1,:] != 0.), (map1[2,:] != 0.),
							 (map2[1,:] != 0.), (map2[2,:] != 0.),
							 (var_map1[1,:] != 0.), (var_map1[2,:] != 0.),
							 (var_map2[1,:] != 0.), (var_map2[2,:] != 0.)], axis=0)
		#goodpoints = np.all([(mask == 1.), SN1, SN2, (np.abs(QcosUsin1)/err1 > nsigma), (np.abs(QcosUsin2)/err2 > nsigma)], axis=0)
		beta[i], sigma_beta[i], q[i], sigmaq[i], dx, dy, ex, ey = linear_fit_cc(QcosUsin1[goodpoints], QcosUsin2[goodpoints], err1[goodpoints], err2[goodpoints], freq1, freq2, str_freq1, str_freq2, detector1, detector2, 0.001)
		sigma_beta[i] = np.abs(sigma_beta[i])
		chi2[i] = red_chi2(QcosUsin1[goodpoints], err1[goodpoints], QcosUsin2[goodpoints], err2[goodpoints], [(freq1/freq2)**beta[i], q[i]])

		if (np.isnan(beta[i]) or beta[i] > 0.):
			beta[i] = 0.
			sigma_beta[i] = np.inf
			q[i] = 0.
			sigmaq[i] = np.inf
			chi2[i] = np.inf

		#label(alpha_deg[i])

		#print(beta[i], sigma_beta[i], chi2[i])
		#if (i== np.size(alpha_deg)-1):
			#plt.legend()
			#plt.savefig(path_out+'FSK_TTplot_'+label+'.png')
			#plt.close()
	#Mask arrays
	mask = beta == 0.0
	if(np.sum(mask) != np.size(alpha_deg)):

	#if(np.sum(mask) > 0):
		beta_ma = ma.masked_array(beta, mask=mask)
		sigma_beta_ma = ma.masked_array(sigma_beta, mask=mask)
		q_ma = ma.masked_array(q, mask=mask)
		sigmaq_ma = ma.masked_array(sigmaq, mask=mask)
	#if(np.sum(mask) != np.size(alpha_deg)):
		beta_tot = np.sum(beta_ma/(sigma_beta_ma**2))/np.sum(1./sigma_beta_ma**2)
		#sigma_beta_tot =np.sqrt(1./np.sum(1./sigma_beta_ma**2))
		q_tot = np.sum(q_ma/(sigmaq_ma**2))/np.sum(1./sigmaq_ma**2)
		#sigma_q_tot =np.sqrt(1./np.sum(1./(sigmaq_ma**2)))
		min_err_beta = (sigma_beta == np.min(sigma_beta_ma))
		#print( np.sum(min_err_beta) )
		if(np.sum(min_err_beta) != 1):
			#print('sum(min_err_beta)', sum(min_err_beta))
			where_first_min_err_beta = np.where(min_err_beta)[0][0]
			min_err_beta[:] = 0
			min_err_beta[where_first_min_err_beta] = 1
			min_err_beta = min_err_beta.astype('int')
			#print('sum(min_err_beta)', sum(min_err_beta))
		sigma_beta2 = np.std(beta_ma) # (np.abs(np.max(beta_ma)-np.min(beta_ma))/2.)

		if(1):
			plt.errorbar(alpha_deg, beta, xerr=None, yerr=sigma_beta)
			plt.xlabel(r'$\alpha$ [deg]', fontsize=16)
			plt.ylabel(r'$\beta$', fontsize=16)
			#plt.plot(alpha_deg, np.ones(np.size(alpha))*(-3), color='red')
			plt.plot(alpha_deg, np.ones(np.size(alpha))*beta_tot, color='black')
			plt.axhline(beta_tot, color='black', label=r'$<\beta>$='+str("{0:.2f}".format(beta_tot))+'$\pm$'+str("{0:.2f}".format(sigma_beta2)))
			plt.axhspan(beta_tot-sigma_beta[min_err_beta], beta_tot+sigma_beta[min_err_beta], alpha=0.25, color='black', label=r'$1\sigma$ min(err)')
			plt.axhspan(beta_tot-sigma_beta2, beta_tot+sigma_beta2, alpha=0.15, color='black', label=r'$1\sigma$')
			plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=None,useLocale=None,useMathText=True)

			plt.legend(prop={'size': 16})
			plt.savefig(path_out+'/'+label+'_fusk.png')
			#print(sigma_beta)
			#print(sigma_beta[min_err_beta])
			#plt.show()
			#plt.close()

	else:
		beta_tot = hp.pixelfunc.UNSEEN
		q_tot = hp.pixelfunc.UNSEEN
		min_err_beta = 0
		sigma_beta2 =  hp.pixelfunc.UNSEEN
		sigma_beta3 = hp.pixelfunc.UNSEEN

	sigma_beta2 = np.max([np.abs(sigma_beta[min_err_beta]), sigma_beta2])
	sigma_beta3 = np.sqrt(sigma_beta2**2+sigma_beta[min_err_beta]**2) #(np.abs(np.max(beta_ma)-np.min(beta_ma))/2.)

	print('RESULT: beta_tot, np.min(sigma_beta)', beta_tot, sigma_beta[min_err_beta])
	return(beta_tot, sigma_beta[min_err_beta], sigma_beta2, sigma_beta3, q_tot, sigmaq[min_err_beta],chi2[min_err_beta])

def p_fusk14_posterior(mask_in, map1, var_map1, freq1, map2, var_map2, freq2, nsigma, path_out, lab_reg, lab_map1, lab_map2): #1x 2y
	alpha_deg = np.arange(0,90,step=5)
	alpha = alpha_deg*np.pi/180.
	beta = np.zeros(np.size(alpha))
	sigma_beta = np.zeros(np.size(alpha))
	q = np.zeros(np.size(alpha))
	sigmaq = np.zeros(np.size(alpha))
	chi2 = np.zeros(np.size(alpha))

	betas = np.arange(-6,-1,0.01)
	emms = (freq2/freq1)**betas
	posterior = np.zeros([np.size(alpha), np.size(betas)])

	for i in range(np.size(alpha)):

		QcosUsin1 = map1[1,:]*np.cos(2.*alpha[i])+map1[2,:]*np.sin(2.*alpha[i])
		QcosUsin2 = map2[1,:]*np.cos(2.*alpha[i])+map2[2,:]*np.sin(2.*alpha[i])

		#hp.mollview(QcosUsin1, 'QcosUsin1')
		#hp.mollview(QcosUsin2, 'QcosUsin2')
		#plt.show()
		err1 = np.sqrt(var_map1[1,:]*(np.cos(2.*alpha[i]))**2+var_map1[2,:]*(np.sin(2.*alpha[i]))**2)
		err2 = np.sqrt(var_map2[1,:]*(np.cos(2.*alpha[i]))**2+var_map2[2,:]*(np.sin(2.*alpha[i])**2))
		SN1 = (np.abs(QcosUsin1)/err1 > nsigma)
		SN2 = (np.abs(QcosUsin2/err2) > nsigma)
		goodpoints = np.all([(mask_in == 1.), SN1, SN2,
							 (map1[1,:] != hp.pixelfunc.UNSEEN), (map1[2,:] != hp.pixelfunc.UNSEEN),
							 (map2[1,:] != hp.pixelfunc.UNSEEN), (map2[2,:] != hp.pixelfunc.UNSEEN),
							 (map1[1,:] != 0.), (map1[2,:] != 0.),
							 (map2[1,:] != 0.), (map2[2,:] != 0.),
							 (var_map1[1,:] != 0.), (var_map1[2,:] != 0.),
							 (var_map2[1,:] != 0.), (var_map2[2,:] != 0.)], axis=0)
						 #goodpoints = np.all([(mask == 1.), SN1, SN2, (np.abs(QcosUsin1)/err1 > nsigma), (np.abs(QcosUsin2)/err2 > nsigma)], axis=0)
		#chi2[i] = red_chi2(QcosUsin1[goodpoints], err1[goodpoints], QcosUsin2[goodpoints], err2[goodpoints], [(freq1/freq2)**beta[i], q[i]])

		#compute the posterior for various input m
		plt.errorbar(x=QcosUsin1[goodpoints], y=QcosUsin2[goodpoints], xerr=err1[goodpoints], yerr=err2[goodpoints], fmt='o')
		for m in range(np.size(emms)):
			sigmamap = np.ones(np.size(QcosUsin1))*hp.pixelfunc.UNSEEN
			sigma_n2 = err2[goodpoints]**2+emms[m]**2*err1[goodpoints]**2
			q = np.mean(QcosUsin2[goodpoints]-emms[m]*QcosUsin1[goodpoints])
			#posterior[i,m] = -0.5*np.sum(np.log(2.*np.pi*sigma_n2)+((QcosUsin2[goodpoints]-emms[m]*QcosUsin1[goodpoints]-q)**2/(sigma_n2)))
			posterior[i,m] = -0.5*np.sum(((QcosUsin2[goodpoints]-emms[m]*QcosUsin1[goodpoints]-q)**2/(sigma_n2)))

			if(m%5 == 0):
				plt.plot(QcosUsin1[goodpoints], emms[m]*QcosUsin1[goodpoints]+q, label = r'$\beta=$'+str(betas[m])[0:5])
			if(m == 0 and i==0):
				#print('sum1', np.sum(np.log(2.*np.pi*sigma_n2)))
				#print('sum2', np.sum(((QcosUsin2[goodpoints]-emms[m]*QcosUsin1[goodpoints]-q)**2/(sigma_n2))))
				sigmamap[goodpoints] = sigma_n2
				#hp.mollview(sigmamap, title='sigmamap')
				sigmamap[goodpoints] = QcosUsin2[goodpoints]
				#hp.mollview(sigmamap, title='err')
				sigmamap[goodpoints] = QcosUsin1[goodpoints]
				#hp.mollview(sigmamap, title='err ref')
				#plt.show()
				plt.close()
		plt.title(r'$\alpha$'+str(alpha[i])[0:4])
		plt.legend()
		plt.savefig(path_out+'posterior_ttplot_FSK_'+lab_reg+'_'+lab_map1+'GHz_'+lab_map2+'GHz.png')
		#plt.show()
		plt.close()
	return(alpha*180./np.pi, betas, posterior)


def make_tt_plot_goodpoints(p1, p2, err_p1, err_p2, freq1, freq2):
	lab = ['I','Q','U']
	# Create a model for fitting.
	linear_model = Model(linear_func)
	# Create a RealData object using our initiated data from above.
	data = RealData(p1, p2, sx=err_p1, sy=err_p2)
	# Set up ODR with the model and data.
	odr = ODR(data, linear_model, beta0=[0.5, 0.])
	# Run the regression.
	out = odr.run()
	# Use the in-built pprint method to give us results.
	#out.pprint()
	line = linear_func(out.beta, p1)
	sigma_q = out.sd_beta[1]
	q = out.beta[1]
	if(out.beta[0] > 0.):
		beta = np.log(out.beta[0])/np.log(freq2/freq1)
		sigma_beta = out.sd_beta[0]/out.beta[0]/np.log(freq2/freq1)
	else:
		beta = np.nan
		sigma_beta = np.inf
	if(0):
		plt.errorbar(p1, p2, xerr=err_p1, yerr=err_p2, linestyle="None", fmt='.r', ecolor='b')
		plt.plot(p1, line, label='b='+str(beta)+'+-'+str(sigma_beta))
		plt.legend()
		#plt.show()
		plt.close()

	return(beta, sigma_beta, q, sigma_q)
