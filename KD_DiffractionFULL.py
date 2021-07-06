'''
This code computes the KD diffraction pattern
following the double slit (full) code
with a point source.
Everything in SI units.
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate as integ
import math
import cmath
import time
import sys

start_time = time.time()

def propagate(Psi_in,x_in,dx,x_out,l):
	global lamdB
	Psi_out=np.zeros_like(x_out,dtype="complex")
	for i in range(len(x_out)):
		for j in range(len(x_in)):
			Kij=cmath.exp(1j*2*math.pi*math.sqrt((x_out[i]-x_in[j])**2+l**2)/lamdB)
			Psi_out[i]+=Kij*Psi_in[j]*dx
	return np.array(Psi_out,dtype="complex")

m=9.10938356e-31
q=1.602e-19
hbar=1.0545718e-34
c=299792458
eps0=8.85e-12

E=380 # electron energy in eV
v=math.sqrt(2*(E*q)/m) # electron velocity ~ 1.1e7 m/s
lamdB=2*math.pi*hbar/(m*v) # deBroglie wavelength

D=125e-6 # Laser beam waist
lamL=532e-9 # laser wavelength
k=2*math.pi/lamL
wL=k*c
I=0.5e14
t=D/v # total interaction time

w2=2e-6 # second collimating slit width
l0=24e-2 # distance from incoherent source slit to collimating slit
l1=5.5e-2 # distance from second slit to grating
l2=24e-2 # distance from grating to screen

#V0=0
V0=q**2*I/(2*m*eps0*c*wL**2) # Ponderomotive potential amplitude

dx01=10e-9
x01=np.arange(-w2/2,w2/2,dx01) # collimating second slit x-coordinate
	
xmax1=5e-6
dx1=10e-9
x1=np.arange(-xmax1,xmax1,dx1) # grating (laser) x-coordinate
	
xmax2=1e-3
dx2=1e-6
x2=np.arange(-xmax2,xmax2,dx2) # detection screen x-coordinate
	
Vpond=np.array([V0*math.cos(k*x)**2 for x in x1]) # ponderomotive potential

widths=np.linspace(2e-6,200e-6,50) # incoherent source slit
cmax=np.zeros_like(widths) # initializing central maximum position (shift)

j=0
for w1 in widths:
	dx0=w1/10
	dchi=dx0
	x0=np.arange(-w1/2,w1/2,dx0) # incoherent source slit x-coordinate
	
	P_detect=np.zeros([len(x2),len(x0)]) # initializing the detection probability matrix (cols represent each incoherent source position)
	
	# this section computes the propagation of the source wavefunction from the source to the one on the screen
	
	for i in range(len(x0)):
		Psi_inc=np.zeros_like(x0) # initializing wavefunction at incoherent source slit
		Psi_slit=np.zeros_like(x01,dtype="complex") # initializign wavefunction at collimating slit
		Psi=np.zeros_like(x1,dtype="complex") # initializing wavefunction before laser interaction
		Psi_detect=np.zeros_like(x2,dtype="complex") # initializing wavefunction at detection screen
		
		# this section computes the source wavefunction (wf) using a incoherent slit
		
		Psi_inc[i]=1
		
		if i==0 and w1==max(widths):# or i==len(x0)-1:
			plt.figure(1)
			plt.subplot(2,1,1)
			plt.stem(x0*1e6,Psi_inc)
			plt.xlabel=('x0 microns')
			plt.ylabel=('Wavefunction')
		
		# this section computes the wf propagation from the first (incoherent source) slit of width w1 to the second (collimating) slit of width w2
		
		Psi_slit=propagate(Psi_inc,x0,dx0,x01,l0)
		
		# this section computes the wf propagation from the second (collimating) slit to the laser
		
		Psi=propagate(Psi_slit,x01,dx01,x1,l1)
		
		# this section computes the wf after laser interaction
		
		Psi_grate=np.array([Psi[i]*cmath.exp(-1j*Vpond[i]*t/hbar) for i in range(len(x1))],dtype="complex")
		
		# this section computes the wf propagation from the laser to the screen
		
		Psi_detect=propagate(Psi_grate,x1,dx1,x2,l2)
		
		P_detect[:,i]=(np.conj(Psi_detect)*Psi_detect).real
		P_detect[:,i]/=integ.simps(P_detect[:,i],x2,dx2) # normalizing the probability distribution
		
		if i==0 and w1==max(widths):# or i==len(x0)-1:
			plt.figure(1)
			plt.subplot(2,1,2)
			plt.plot(x2*1e6,P_detect[:,i])
			plt.xlabel=('x2 microns')
			plt.ylabel=('P_detect')
			plt.savefig('first_delta.png')
	
	cmax[j]=x2[np.where(P_detect[:,1]==max(P_detect[:,1]))] # storing the central maximum (due to first delta) shift
	#print(cmax[j])
	j+=1
	
	# this section sums over the incoherent source
	
	P_final=np.array(P_detect.sum(axis=1)*dx0)

	if w1==max(widths):
		plt.figure(2)
		plt.plot(x2*1e6,P_final)
		plt.xlabel=('x2 microns')
		plt.ylabel=('Final probability')
		plt.savefig('final_prob.png')
	
#print(widths)
#print(cmax)

plt.figure(3)
plt.plot(widths*1e6/2,cmax)
plt.xlabel=('w1/2 microns')
plt.ylabel=('Shifting')
plt.savefig('shifting.png')

print(time.time() - start_time, "seconds")
