import numpy as np
import pyscf
from pyscf import gto, dft, lib, scf
from pyscf.dft import numint
from pyscf.dft import r_numint
import matplotlib.pyplot as plt
import pylibxc
import math
import sys

if len(sys.argv) != 4:
        print("Usage: python my_script.py molecule_name charge spin correlation exchange")


    # Retrieve the command-line arguments
molecule_inp = sys.argv[1]
charge_inp = int(sys.argv[2])
mult_inp = int(sys.argv[3])
corr_f = sys.argv[4]
ex_f = sys.argv[5]
#dir_path = os.getcwd()
#Script to calculate score 1

#User input
'''
molecule_inp = input("Enter the name of the molecule: ")
charge_inp = int(input("Enter the charge: "))
mult_inp = int(input("Enter the number of unpaired electrons: "))
corr_f = input("Enter correlation functional: ")
ex_f = input("Enter exchange functional: ")
'''
#path_to_xyz = dir_path+"/"+molecule_inp+".xyz"
path_to_xyz = "/export/zimmerman/vkchem/python_try/CH3F/He_19Jun/W4-11/BLYP/norm1/W4-singl-score1blyp/"+molecule_inp+"/struc.xyz"

#Molecule input
mol = gto.M(
    verbose = 0,
    atom = path_to_xyz,
    charge = charge_inp,
    spin = mult_inp,
    basis = 'augccpvtz')

#DFt on grid
mf = dft.RKS(mol)
mf.grids.level = 9
mf.kernel()
dm = mf.make_rdm1()

# Use default mesh grids and weights
coords = mf.grids.coords
weights = mf.grids.weights
ao_value = numint.eval_ao(mol, coords, deriv=1)
# The first row of rho is electron density, the rest three rows are electron
# density gradients which are needed for GGA functional
rho = numint.eval_rho(mol, ao_value, dm, xctype='GGA')
print("Shape of rho: ",rho.shape)

rho1 = np.array(rho)
rho2 = np.transpose(rho1)
print("Shape of rho transpose: ", rho2.shape)

#libxc quantites 
lib_rho = []
lib_sig = []
for i in range(len(rho2)):
    lib_rho.append(rho2[i][0])
    lib_sig.append((rho2[i][1])**2 + (rho2[i][2])**2 + (rho2[i][3])**2)
    
rho_libxc = np.array(lib_rho)
sig_libxc = np.array(lib_sig)
print("Rho (Libxc) shape: ",rho_libxc.shape)
print("Sigma (LibXC) shape: ",sig_libxc.shape)
    
# Build functional
func = pylibxc.LibXCFunctional(corr_f, "unpolarized")

# Create input
inp = {}
inp["rho"] = rho_libxc
inp["sigma"] = sig_libxc

# Compute
ret = func.compute(inp, do_fxc= True)

print("Output of LibXC for ", corr_f," correlation functional")
print(ret)

#Weigner seitz
def rs(rho):
    return ((4*np.pi*rho)/3)**(-1/3)

#LDA exchange energy density
def eunif(rho):
    return ((-(3 / (4 * np.pi))) * ((rho * 3 * np.pi**2)**(1 / 3)))

def drho_drs(r):
    return ((-9/(4 * np.pi)) * r**(-4))

def d2rho_drs2(r):
    return ((9/np.pi) * r**(-5))

def dex_drho(rho):
    return ((-3/(4 * np.pi)) * (3 * (np.pi)**2)**(1/3) * (1/3) * (rho)**(-2/3))

def dex_drs(r):
    return((-3/(4 * np.pi)) * (3 * (np.pi)**2)**(1/3) * (3/(4 * np.pi))**(1/3) * (-1) * (r)**(-2))

def d2ex_drs2(r):
    return((-3/(4 * np.pi)) * (3 * (np.pi)**2)**(1/3) * (3/(4 * np.pi))**(1/3) * (2) * (r)**(-3))

r_s = []
e_unifx = []
dexdrho = []
for b in range(len(rho2)):
    r_s.append(rs(rho2[b][0]))
    e_unifx.append(eunif(rho2[b][0]))
    dexdrho.append(dex_drho(rho2[b][0]))
    
drhodrs = []
d2rhodrs2 = []
dexdrs = []
d2exdrs2 = []
for c in range(len(rho2)):
    drhodrs.append(drho_drs(r_s[c]))
    d2rhodrs2.append(d2rho_drs2(r_s[c]))
    dexdrs.append(dex_drs(r_s[c]))
    d2exdrs2.append(d2ex_drs2(r_s[c]))
    
dec_drho = [((ret["vrho"][d] - ret["zk"][d])/rho2[d][0]) for d in range(len(rho2))]

d2ec_drho2 = [((ret["v2rho2"][e] - (2*dec_drho[e]))/rho2[e][0]) for e in range(len(rho2))]

dec_drs = [dec_drho[f]*drhodrs[f] for f in range(len(rho2))]

d2ec_drs2 = [(d2ec_drho2[g] * (drhodrs[g])**2) + (dec_drho[g] * d2rhodrs2[g]) for g in range(len(rho2))]

fc = np.zeros(len(rho2))
for h in range(len(rho2)):
    fc[h] = ret["zk"][h]/e_unifx[h]
    
    
dfc_drs = [((e_unifx[a]*dec_drs[a]) - (ret["zk"][a]*dexdrs[a]))/((e_unifx[a])**2) for a in range(len(rho2))]

d2fc_drs2 = [(((d2ec_drs2[j] * e_unifx[j]) - (d2exdrs2[j] * ret["zk"][j]))*(e_unifx[j]) - (2 * dexdrs[j])*((dec_drs[j] * e_unifx[j]) - (dexdrs[j] * ret["zk"][j])))/((e_unifx[j])**3) for j in range(len(rho2))]

# Build functional
func2 = pylibxc.LibXCFunctional(ex_f, "unpolarized")

# Create input
inp2 = {}
inp2["rho"] = rho_libxc
inp2["sigma"] = sig_libxc

# Compute
ret2 = func2.compute(inp2, do_fxc= True)

print("LibXC ", ex_f, "exchange functional output")

print(ret2)

f_x = np.zeros(len(rho2))
for k in range(len(rho2)):
    f_x[k] = ret2["zk"][k]/e_unifx[k]
    
f_xc = [f_x[l]+fc[l] for l in range(len(rho2))]

ones = []
for w in range(len(rho2)):
    ones.append(1)
    
n_el = np.einsum('i,i,i->', ones, rho[0], weights)
print("Number of electrons in ", molecule_inp, ":", n_el)

print("Condition 4 (see Burke preprint 2023)")
vk4 = []
for q in range(len(rho2)):
    if ret["zk"][q] > 0:
        vk4.append(ret["zk"][q][0])
    else:
        vk4.append(0)
        
vk4_f = np.einsum('i,i,i->', vk4, rho[0], weights)

print(molecule_inp,"/",corr_f,"/augccpvtz correlation energy non-positivity score 2 semi-normalised: ", vk4_f/n_el)

print("Condition 7")
vk7 = []
for q in range(len(rho2)):
    if f_xc[q] > 2.27:
        vk7.append(f_xc[q] - 2.27)
    else:
        vk7.append(0)
        
vk7_f = np.einsum('i,i,i->', vk7, rho[0], weights)

print(molecule_inp,"/",ex_f,corr_f,"/augccpvtz LO extension to Exc score 2 semi-normalised:: ", vk7_f/n_el)

print("Condition 10")

vk10 = []
for q in range(len(rho2)):
    if dfc_drs[q] < 0:
        vk10.append(dfc_drs[q][0])
    else:
        vk10.append(0)
        
vk10_f = np.einsum('i,i,i->', vk10, rho[0], weights)
print(molecule_inp,"/",corr_f,"/augccpvtz Ec scaling inequality score 2 semi-normalised:: ", vk10_f/n_el)

print("Condition 13")

rs_sf, fc_sf = zip(*sorted(zip(r_s, fc)))

# Remove NaN and inf from the list using filter and lambda
cleaned_list = list(filter(lambda x: not (isinstance(x, float) and (math.isnan(x) or math.isinf(x))), fc_sf))


fcinf = cleaned_list[-1]

vk13 = []
for q in range(len(rho2)):
    if dfc_drs[q] > (fcinf - fc[q])/r_s[q]:
        vk13.append(dfc_drs[q][0] - ((fcinf - fc[q])/r_s[q]))
    else:
        vk13.append(0)
        
vk13_f = np.einsum('i,i,i->', vk13, rho[0], weights)
print(molecule_inp,"/",corr_f,"/augccpvtz Tc upper bound score 2 semi-normalised:: ", vk13_f/n_el)

print("Condition 17")

cond_17 = []
for t in range(len(rho2)):
    cond_17.append(f_xc[t] + ((r_s[t])*dfc_drs[t]))
    
vk17 = []
for q in range(len(rho2)):
    if cond_17[q] > 2.27:
        vk17.append(cond_17[q][0] - 2.27)
    else:
        vk17.append(0)
        
vk17_f = np.einsum('i,i,i->', vk17, rho[0], weights)
print(molecule_inp,"/",ex_f,corr_f,"/augccpvtz Lieb-Oxford score 2 semi-normalised:: ", vk17_f/n_el)

print("Condition 15")

r_s_2 = [(r_s[i])**2 for i in range(len(rho2))]

cond15 = []
for y in range(len(rho2)):
    cond15.append((2 * r_s[y] * dfc_drs[y]) + (r_s_2[y] * d2fc_drs2[y]))
    
vk15 = []
for q in range(len(rho2)):
    if cond15[q] < 0:
        vk15.append(cond15[q][0])
    else:
        vk15.append(0)
        
vk15_f = np.einsum('i,i,i->', vk15, rho[0], weights)
print(molecule_inp,"/",corr_f,"/augccpvtz Uc monotonicity score 2 semi-normalised:: ", vk15_f/n_el)