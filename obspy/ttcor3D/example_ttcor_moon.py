'''
Example of application of ttcor routines computing P and S wave crust and topography corrections
using Moon models (VPREMOON) and pycrust models of the Moon
primary authors (August 2018):
Raphael F. Garcia (raphael.garcia@isae.fr)
and Gabriel Bena 
'''

import os
import sys
import obspy
import matplotlib.pyplot as plt
import numpy as np
import pyshtools
import ttcor_crust as ttcor
from obspy.taup import TauPyModel

#####
## Simple use of ttcor computing corrections assuming P and S ray parameters,
## and moho and topography radii are known at source and receiver
# set mantle density
rho_m = 3396.0
# set crust density from pycrust density model assuming 11% porosity
rho_c = 2605.4
# set quake depth
quake_depth=float(10.0)
# Assume vertical incidence
p_ray_param=float(0.0)
s_ray_param=float(0.0)
# Assume average radius (value consistent with VPREMOON value
topo_r0=float(1731.1)
# load seismoloigal model (VPREMOON + small inner core to be managed by obspy taup) 
thickref=float(28.0)
model_nd='./example_Moon_data/VPREMOON'+str(int(thickref))+'.nd'
# set Moho radii at source and receiver
moho_rad_stat = topo_r0 - float(thickref) - 10.0
moho_rad_quake = topo_r0 - float(thickref) + 10.0
# set topography radii at source and receiver
topo_rad_stat = topo_r0 + 1.0
topo_rad_quake = topo_r0 - 1.0
# topo_rad_quake necessary if quake_depth=0.0
# because in that case interpreted as an impact on the surface

# compute P and S corrections at vertical incidence angle
[corrections]=ttcor.ttcor_corrections_simple(quake_depth,p_ray_param,s_ray_param,moho_rad_stat,moho_rad_quake,rho_c,rho_m,topo_rad_stat,topo_rad_quake,topo_r0,model_nd)
# output values
print('vertical incidence angle 3D crust + topo corrections')
print('(to be added to 1D travel times)')
print('P cor. at source, S cor. at source,P cor. at receiver, S cor. at receiver')
print(corrections)
print(' ')

## test figures at vertical incidence angle, varying topography and moho depth
moho_rad_stat = topo_r0 - float(thickref)
moho_rad_quake = topo_r0 - float(thickref)
topo_rad_stat = topo_r0
topo_rad_quake = topo_r0
# computation of topography corrections
topo_tab=np.linspace(-10,10,21)
Pcor=0.0*topo_tab
Scor=0.0*topo_tab
for i in range(21):
    topo_rad_stat = topo_r0 + topo_tab[i]
    [corrections]=ttcor.ttcor_corrections_simple(quake_depth,p_ray_param,s_ray_param,moho_rad_stat,moho_rad_quake,rho_c,rho_m,topo_rad_stat,topo_rad_quake,topo_r0,model_nd)
    Pcor[i]=corrections[0,2]
    Scor[i]=corrections[0,3]
# plot figure of station corrections as a function of topography
plt.figure(1)
plt.plot(topo_tab,Pcor,label='P cor.',marker='o',color='b')
plt.plot(topo_tab,Scor,label='S cor.',marker='*',color='r')
plt.title('P and S topography corrections at vertical incidence')
plt.ylabel('travel time correction (in s)')
plt.xlabel('topography (in km)')
plt.legend()
plt.show(block=False)
topo_rad_stat = topo_r0
# computation of Moho corrections
moho_diff_tab=np.linspace(-20,20,21)
Pcor=0.0*moho_diff_tab
Scor=0.0*moho_diff_tab
for i in range(21):
    moho_rad_stat = topo_r0 - float(thickref) + moho_diff_tab[i]
    [corrections]=ttcor.ttcor_corrections_simple(quake_depth,p_ray_param,s_ray_param,moho_rad_stat,moho_rad_quake,rho_c,rho_m,topo_rad_stat,topo_rad_quake,topo_r0,model_nd)
    Pcor[i]=corrections[0,2]
    Scor[i]=corrections[0,3]
# plot figure of station corrections as a function of topography
plt.figure(2)
plt.plot(topo_r0 - float(thickref) + moho_diff_tab,Pcor,label='P cor.',marker='o',color='b')
plt.plot(topo_r0 - float(thickref) + moho_diff_tab,Scor,label='S cor.',marker='*',color='r')
plt.title('P and S Moho corrections at vertical incidence')
plt.ylabel('travel time correction (in s)')
plt.xlabel('Moho radius (in km)')
plt.legend()
plt.show(block=False)
##
#####

#####
## Use of ttcor with
## 1- topography and Moho models from pycrust (geodetic model)
## 2- taup computation of travel times and ray parameters in a seismic model
## !!! CONSISTENCY MUST BE ENSURED BETWEEN PARAMETERS OF GEODETIC AND SEISMIC MODELS !!!
## !!! CRUSTAL THICKNESS IN SEISMIC MODEL = AVERAGE CRUSTAL THICKNESS OF GEDODETIC MODEL
## !!! MANTLE DENSITY BELOW CRUST IN SEISMIC MODEL = MANTLE DENSITY OF GEODETIC MODEL

# set planetary flattening parameter (Earth = 1.0/298.257, Moon = 1.0/900.0, Mars = 0.00589 )
flat = 1.0 / 900.0
# set mantle density
rho_m = 3396.0
# set crust density from pycrust density model assuming 11% porosity
rho_c = 2605.4
# set quake depth
quake_depth=float(0.0) # Assume impact on the surface
# strong impact recorded by 4 stations of Apollo network
# (date : 7205130846, VPREMOON location from Garcia et al., 2011)
quake_lat=float(1.55)
quake_lon=float(-16.91)
# Apollo 12 station location (from LRO relocations by Wagner et al., 2017)
stat_lat=float(3.01)
stat_lon=float(-23.42)
# load topography model
# import Moon topography from LOLA (file available in pycrust)
degmax = 900
topofile = './example_Moon_data/LOLA1500p.sh'
topo = pyshtools.SHCoeffs.from_file(topofile, degmax)
# load moho shape model (computed by pycrust for an average crustal thickness of 28 km and rho_c, rho_m)
moho_pad = pyshtools.SHCoeffs.from_file('./example_Moon_data/mohosave'+str(int(thickref))+'.sh', topo.lmax)
# load seismoloigal model (VPREMOON + small inner core to be managed by obspy taup) 
thickref=float(28.0)
model_nd='./example_Moon_data/VPREMOON'+str(int(thickref))+'.nd'
# load taup tables
model_npz='./example_Moon_data/VPREMOON'+str(int(thickref))+'.npz'
# AND, if not already done, compute (may be long computation) with
#obspy.taup.taup_create.build_taup_model(model_nd,'./example_Moon_data')
model=TauPyModel(model_npz)
## compute ray parameters
# extract average radius from topography model (in meter here)
topo.r0 = topo.coeffs[0, 0, 0]
# compute distance
dist=obspy.geodetics.base.gps2dist_azimuth(stat_lat,stat_lon,quake_lat,quake_lon,topo.r0,flat)
d=dist[0]*180.0/(topo.r0*np.pi)
# compute body waves with taup
arrivals_P=model.get_travel_times(quake_depth ,d,["P","p"])
arrivals_S=model.get_travel_times(quake_depth ,d,["S","s"])
arrival_p=arrivals_P[0]
arrival_s=arrivals_S[0]
# extract ray parameters
p_ray_param=arrival_p.ray_param
s_ray_param=arrival_s.ray_param

# compute P and S corrections at Apollo 12 station
[corrections]=ttcor.ttcor_corrections(stat_lat,stat_lon,quake_lat,quake_lon,quake_depth,p_ray_param,s_ray_param,moho_pad,rho_c,rho_m,topo,model_nd)

# compute P and S corrections and travel times at Apollo 12 station
[corrections2,times3D,times1D]=ttcor.ttcor_complete(stat_lat,stat_lon,quake_lat,quake_lon,quake_depth,moho_pad,rho_c,rho_m,topo,flat,model_nd,model_npz)

# check outputs of ttcor.ttcor_corrections  and ttcor.ttcor_complete are identical
print('Corrections from ttcor.ttcor_corrections')
print(corrections)
print('Corrections from ttcor.ttcor_complete')
print(corrections2)
print(' ')
print('3D and 1D P travel times')
print(times3D[0,0])
print(times1D[0,0])
print('3D and 1D S travel times')
print(times3D[0,1])
print(times1D[0,1])

#####


