import numpy as np
import scipy as sp
from math import pi

def get_spectrum_constants():
	const = {
		'h': 6.626e-34,
		'k': 1.381e-23,
		'c': 2.997e8,
		'pi': pi,
		'dust_optical_depth_freq': 1198.8,
		'tcmb': 2.7255
	}
	return const

def synchrotron(const, freq, basefreq, sync_amp, sync_spec):
	return sync_amp * (freq/basefreq)**sync_spec

def syncshifted_comm(nu, sync_amplitude, sync_shift, sync_freq, sync_amp):
	nu_0s = 0.408
	nu_p0s = 0.0154

	# shift and interpolate
	synchrotron = sync_amplitude * (nu_0s/nu)**2 * ((np.interp((nu*nu_p0s/sync_shift),sync_freq, sync_amp)) / (np.interp((nu_0s*nu_p0s/sync_shift),sync_freq, sync_amp)))

	# print synchrotron
	# if outside range of synchrotron model, then set to 0
	if sync_shift != 0 and isinstance(synchrotron,list):
		synchrotron[nu < min(sync_freq)*2] = 0.0
		synchrotron[nu > max(sync_freq)-100.] = 0.0

	# if outside range of spinning dust model, then set to 0
	# IF (sync_shift NE 0) THEN BEGIN
	# badvals = where(S LT 0.,nbadvals)
	# IF (nbadvals GT 0) THEN S[badvals]=0.
	# ENDIF

	return synchrotron


def syncshifted_pol_comm(nu, sync_amplitude, sync_shift, sync_freq, sync_amp):
	nu_0s = 30.0
	nu_p0s = 0.0154

	# shift and interpolate
	synchrotron = sync_amplitude * (nu_0s/nu)**2 * ((np.interp((nu*nu_p0s/sync_shift),sync_freq, sync_amp)) / (np.interp((nu_0s*nu_p0s/sync_shift),sync_freq, sync_amp)))

	# print synchrotron

	# if outside range of synchrotron model, then set to 0
	if sync_shift != 0:
		synchrotron[nu < min(sync_freq)*2] = 0.0
		synchrotron[nu > max(sync_freq)-100.] = 0.0

	# if outside range of spinning dust model, then set to 0
	# IF (sync_shift NE 0) THEN BEGIN
	# badvals = where(S LT 0.,nbadvals)
	# IF (nbadvals GT 0) THEN S[badvals]=0.
	# ENDIF

	return synchrotron


def freefree(const, freq, EM, Te, solid_angle, equation=1,comm=0):

	if equation == 1:
		# new equations from Draine (2011) book!
		g_ff = np.log(np.exp(5.960 - (np.sqrt(3.0)/np.pi)*np.log(freq*(Te/10000.0)**(-3.0/2.0))) + 2.71828)
		tau_ff = 5.468e-2 * Te**(-1.5) * freq**(-2.0) * EM * g_ff
		# tau_ff = 1.772d-2 * (Te)^(-1.5) * (freq)^(-2d) * EM * g_ff
	elif equation == 2:
		# previous equations, from old/standard text books!
		g_ff = np.log(4.955e-2 / freq) + 1.5*np.log(Te)
		tau_ff = 3.014e-2 * Te**(-1.5) * freq**(-2.0) * EM * g_ff
	elif equation == 3:
		# old Altenhoff approximation, also in many text books including
		# Scheffler book
		tau_ff = 8.235e-2 * Te**(-1.35) * freq**(-2.1) * EM

	# finally get brightness temperature from the optical depth
	T_ff = Te * (1.0 - np.exp(-tau_ff))

	# deal with very small tau values in the exponential
	# if tau_ff < 1.0e-10:
	# 	T_ff = Te * (tau_ff - (-tau_ff)**2.0 / 2.0 - (-tau_ff^3.0)/6.0)

	if comm == 1:
		S = 1e6 * T_ff
	else:
		S = 2.0 * 1381.0 * T_ff * (freq*1e9)**2 / const['c']**2  * solid_angle

	return S

def spinningdust(spd_freq, spd_em, solid_angle, freq, amp, shift):

	# shift and interpolate
	# interp = sp.interpolate.interp1d(spd_freq, spd_em)
	# spinningdust = interp(freq)
	spinningdust = np.interp(freq, spd_freq, spd_em)
	# ; if outside range of spinning dust model, then set to 0
	# IF (spdustshift NE 0.) THEN BEGIN
	# badvals = where(nu LT min(spinningdust_freq)*2. OR nu GT (max(spinningdust_freq)-100.), nbadvals)
	# IF (nbadvals GT 0) THEN spinningdust[badvals] = 0.
	# ENDIF

	S = amp * spinningdust * solid_angle * 1.0e20
	return S

	# ; if outside range of spinning dust model, then set to 0
	# IF (spdustshift NE 0) THEN BEGIN
	# badvals = where(S LT 0.,nbadvals)
	# IF (nbadvals GT 0) THEN S[badvals]=0.
	# ENDIF

def spinningdust_comm(nu, spinningdust_amp, spdustshift, spinningdust_em_comm, spinningdust_freq_comm, component):
	nu_p0 = 30.0

	if component == 2:
		nu_0 = 41.0
	elif component == 3:
		nu_0 = 30.0
	else:
		nu_0 = 22.8

	 # shift and interpolate
	spinningdust = spinningdust_amp * (nu_0/nu)**2 * ((np.interp((nu*nu_p0/spdustshift),spinningdust_freq_comm, spinningdust_em_comm)) / (np.interp((nu_0*nu_p0/spdustshift),spinningdust_freq_comm, spinningdust_em_comm)))

	if spdustshift != 0 and isinstance(spinningdust,list):
		spinningdust[nu < min(spinningdust_freq_comm)] = 0.0
		spinningdust[nu > max(spinningdust_freq_comm)-100.] = 0.0

	return spinningdust

def cmb(const, freq, amp, solid_angle):

	xx = const['h'] * freq*1.0e9 / (const['k'] * const['tcmb'])
	xx2 = const['h'] * freq*1.0e9 / (const['k'] * (const['tcmb']+amp))
	S = (2.0e26 * const['h'] * (freq*1e9)**3 * solid_angle  / const['c']**2) * ((1.0/(np.exp(xx2)-1.0)) - (1.0/(np.exp(xx)-1.0)))
	return S

def cmb_comm(const, freq, amp):

	xx = const['h'] * freq*1.0e9 / (const['k'] * const['tcmb'])
	S = amp / ((np.exp(xx)-1.0)**2 / (xx**2*np.exp(xx)))
	return S

def thermaldust(const, freq, amp, index, temperature, optical_depth_freq, solid_angle):

	xx = const['h'] * freq*1.0e9 / (const['k'] * temperature)

	# fit optical depth at 250 um (1198.8 GHz) - dust_optical_depth_freq
	thermaldust = 2.0 * const['h'] * (freq*1e9)**3/(const['c'])**2 * (freq/optical_depth_freq)**index / (np.exp(xx)-1.0)
	S = amp * thermaldust * solid_angle *1e26
	return S

def thermaldust_comm(const, freq, amp, index, temperature):

	xx = const['h'] / (const['k'] * temperature)
	nu_0 = 545.0

	thermaldust = amp * (freq/nu_0)**(index+1.0) * (np.exp(xx*nu_0*1e9)-1.0)/(np.exp(xx*freq*1e9)-1.0)
	return thermaldust

def paraobla(freq, coeff1, coeff2, coeff3):
	S = np.exp(coeff1 + coeff2*np.log(freq) + coeff3*np.log(freq)**2 )
	return S

def planckcorr(const, nu_ghz):
	x = const['h'] * np.asarray(nu_ghz) * 1.0e9 / (const['k'] * const['tcmb'])
	value = (np.exp(x)-1.0)**2.0 / (x**2. * np.exp(x))
	return value

# Function to calculate the conversion factors between different units.
#
# Mike Peel, 1 October 2014 - Forked from haperflux.pro
# Mike Peel, 15 February 2016 - converted from IDL to python
# Mike Peel, 21 September 2017 - add special case of 'none'
# Mike Peel, 21 September 2018 - also handle 'Kcmb' format (used in new HFI maps)
def convertunits(const, units_in, units_out, frequency, pix_area=1.0):
	unitslist = ['K','mK','uK','K_RJ','mK_RJ','uK_RJ','K_CMB','mK_CMB','uK_CMB', 'MJy/sr', 'Jy/pix', 'none']
	# get conversion from the input units to Jy/pix
	if (units_in == 'K' or units_in == 'K_RJ' or units_in == 'KRJ'):
		factor_in = 2.*1381.*(frequency*1.0e9)**2/(const['c'])**2 * pix_area
	elif (units_in == 'mK' or units_in == 'mK_RJ' or units_in == 'mKRJ'):
		factor_in = 2.*1381.*(frequency*1.0e9)**2/(const['c'])**2 * pix_area / 1.0e3
	elif (units_in == 'uK' or units_in == 'uK_RJ' or units_in == 'uKRJ'):
		factor_in = 2.*1381.*(frequency*1.0e9)**2/(const['c'])**2 * pix_area / 1.0e6
	elif (units_in == 'K_CMB' or units_in == 'KCMB' or units_in == 'Kcmb'):
		factor_in = 2.*1381.*(frequency*1.0e9)**2/(const['c'])**2 * pix_area / planckcorr(const, frequency)
	elif (units_in == 'mK_CMB' or units_in == 'mKCMB' or units_in == 'mKCmb'):
		factor_in = 2.*1381.*(frequency*1.0e9)**2/(const['c'])**2 * pix_area / 1.0e3 / planckcorr(const, frequency)
	elif (units_in == 'uK_CMB' or units_in == 'uKCMB' or units_in == 'uKCmb'):
		factor_in = 2.*1381.*(frequency*1.0e9)**2/(const['c'])**2 * pix_area / 1.0e6 / planckcorr(const, frequency)
	elif (units_in == 'MJy/sr' or units_in == "MJY/SR" or units_in == "MjySr"):
		factor_in = pix_area * 1.0e6
	elif (units_in == 'Jy/pixel' or units_in == 'JY/PIXEL' or units_in == 'JY/PIX' or units_in == 'JyPix' or units_in == 'Jy/Pix' or units_in == 'none'):
		factor_in = 1.0
	else:
		print('Invalid unit conversion specified for convertunits (units_in='+str(units_in)+'). ***Returning 1.*** Please use one of the following available units:')
		print(unitslist)
		return 1.0

	if (units_in == 'none'):
		factor_out = 1.0
	elif (units_out == 'K' or units_out == 'K_RJ' or units_out == 'KRJ'):
		factor_out = 2.*1381.*(frequency*1.0e9)**2/(const['c'])**2 * pix_area
	elif (units_out == 'mK' or units_out == 'mK_RJ' or units_out == 'mKRJ'):
		factor_out = 2.*1381.*(frequency*1.0e9)**2/(const['c'])**2 * pix_area / 1.0e3
	elif (units_out == 'uK' or units_out == 'uK_RJ' or units_out == 'uKRJ'):
		factor_out = 2.*1381.*(frequency*1.0e9)**2/(const['c'])**2 * pix_area / 1.0e6
	elif (units_out == 'K_CMB' or units_out == 'KCMB'):
		factor_out = 2.*1381.*(frequency*1.0e9)**2/(const['c'])**2 * pix_area / planckcorr(const, frequency)
	elif (units_out == 'mK_CMB' or units_out == 'mKCMB'):
		factor_out = 2.*1381.*(frequency*1.0e9)**2/(const['c'])**2 * pix_area / 1.0e3 / planckcorr(const, frequency)
	elif (units_out == 'uK_CMB' or units_out == 'uKCMB'):
		factor_out = 2.*1381.*(frequency*1.0e9)**2/(const['c'])**2 * pix_area / 1.0e6 / planckcorr(const, frequency)
	elif (units_out == 'MJy/sr' or units_out == "MJY/SR" or units_out == "MjySr"):
		factor_out = pix_area * 1.0e6
	elif (units_out == 'Jy/pixel' or units_out == 'JY/PIXEL' or units_out == 'JY/PIX' or units_out == 'JyPix' or units_out == 'Jy/Pix'):
		factor_out = 1.0
	else:
		print('Invalid unit conversion specified for convertunits (units_out='+str(units_out)+'). ***Returning 1.*** Please use one of the following available units:')
		print(unitslist)
		return 1.0

	# Return the ratio of the factors
	return factor_in/factor_out
