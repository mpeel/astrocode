import numpy as np
# Debiasing, code from Federica
# Equation 19 of https://arxiv.org/pdf/1410.4436.pdf
def debias_p_mas(Q, U, sigmaQ, sigmaU):
	p = np.sqrt(Q**2+U**2)
	b = np.sqrt((Q**2+sigmaU**2+U**2+sigmaQ**2))/p
	pmas = p-b**2*((1.-np.exp(-p**2/b**2))/2*p)
	sigmap = np.sqrt(0.5*(sigmaQ**2+sigmaU**2))
	SN = (pmas/sigmap > 3.8)
	# print(SN)
	return [pmas, sigmap, SN]

# Debiasing, code from Federica
def debias_p_as(Q, U, sigmaQ, sigmaU):
	p = np.sqrt(Q**2+U**2)
	sigmap = np.sqrt((sigmaQ*Q)**2+(sigmaU*U)**2)/p
	sigmaa = np.sqrt(0.5*(sigmaQ**2+sigmaU**2))
	mask = np.ones(len(p))
	mask[p < sigmap] = 0
	pas = np.zeros(len(p))
	pas[mask == 1] = np.sqrt(p[mask==1]**2-sigmap[mask==1]**2)
	# pas[mask == 0] = 0.0
	SN = (pas/sigmaa > nsigma)
	#print(SN)
	return [pas, sigmap, SN]

# Was Q, U, just changed to U, Q -- MP, 4 June 2021
# Subsequently changed to -U, Q -- MP, October 2021
def calc_polang(Q, U):
	return 0.5*np.arctan2(-U, Q) * 180 / np.pi

def calc_polang_unc(Q, U, Qerr, Uerr):
	unc_map = np.sqrt((Qerr**2)*(-0.5*U/(Q**2.0+U**2.0))**2.0 + (Uerr**2)*(-0.5*Q/(Q**2.0+U**2.0))**2.0) * 180 / np.pi
	unc_map[~np.isfinite(unc_map)] = 1000.0
	return unc_map

# Required to handle points around +- 90°, see Alberto email 19 October 2021
def dodiff(map1, map2):
	# return np.arctan(np.tan((map1-map2)*np.pi/180.0))*(180.0/np.pi)
	return 0.5 * np.arctan(np.tan(2*(map1-map2)*np.pi/180.0))*(180.0/np.pi)
