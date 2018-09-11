'''
Module gathering the routines that will used
to compute topography and crust corrections of P and S direct waves
primary authors (August 2018):
Raphael F. Garcia (raphael.garcia@isae.fr)
and Gabriel Bena 
'''

def ttcor_corrections_simple(quake_depth,p_ray_param,s_ray_param,moho_rad_stat,moho_rad_quake,rho_c,rho_m,topo_rad_stat,topo_rad_quake,topo_r0,model_nd):

  '''
  Function computing
  the crustal corrections on the receiver's and source's
  end for both P and S waves from ray parameters
  using radii of Moho and topography at station and quake positions and average radius of the planet

  Usage
  -----
  [corrections]=ttcor_complete(quake_depth,p_ray_param,s_ray_param,moho_rad_stat,moho_rad_quake,rho_c,rho_m,topo_rad_stat,topo_rad_quake,topo_r0,model_nd):


  Returns
  -----
  corrections (in s): the crustal and topography corrections in a array:
      [P wave correction for the source's end, S wave correction for the source's end,
      P wave correction for the receiver's end, S wave correction for the receiver's end]
      These corrections must be added to the theoretical time in order to obtain the 3D theoretical time

  Parameters
  ------
  quake_depth: quake depth (in km)
  p_ray_param: ray parameter of P wave computed into model_nd model (in s/radian)
  s_ray_param: ray parameter of S wave computed into model_nd model (in s/radian)

  moho_rad_stat: radius of Moho (crust/mantle interface) at station position (in km)
  moho_rad_quake: radius of Moho (crust/mantle interface) at quake position (in km)

  rho_c : Crust volumic mass (kg/m/m/m) used to compute the Moho shape moho_pad

  rho_m : Mantle volumic mass (kg/m/m/m) used to compute the Moho shape moho_pad

  topo_rad_stat: radius of topography at station position (in km)
  topo_rad_quake: radius of topography at quake position (in km)
  topo_r0: average radius of the planet (geodetic altitude 0) (in km)

  model_nd: Text form model of the moon (Named discontinuites *.nd format)


  '''
  import numpy as np

# Extract 1D seismological model 
  [modcrust,thickave]=extract_modcrust(model_nd,rho_m)

  geoparam=np.zeros((1,7))
  geoparam[0,3]=0.0
  geoparam[0,4]=0.0
  geoparam[0,0]=0.0
  geoparam[0,1]=0.0
  geoparam[0,2]=quake_depth
  geoparam[0,5]=p_ray_param*np.pi/180
  geoparam[0,6]=s_ray_param*np.pi/180

  topo_source_radius = topo_rad_quake
  topo_receiver_radius = topo_rad_stat
  moho_source_radius = moho_rad_quake
  moho_receiver_radius = moho_rad_stat

# lower radius for computation of correction
# should be smaller than the minimum radius of Moho discontinuity 
  rd = topo_r0 - 200.0
  outtimes=np.zeros((1,4))

  modcrustradref = build_crust3D(modcrust,topo_r0-thickave/1e3,topo_r0,topo_r0,rd)

  modcrustradrec = build_crust3D(modcrust,moho_receiver_radius,
                                  topo_receiver_radius,topo_r0,rd)
  modcrustradsour = build_crust3D(modcrust,moho_source_radius,
                                  topo_source_radius,topo_r0,rd)
  [ttcorP_crust3D_sour,ttcorS_crust3D_sour]=crust_topo_cor_sour(modcrustradsour,modcrustradref,geoparam[0,:],topo_r0)
  [ttcorP_crust3D_rec,ttcorS_crust3D_rec]=crust_topo_cor_rec(modcrustradrec,modcrustradref,geoparam[0,:],topo_r0)
  outtimes[0,:]=[ttcorP_crust3D_sour, ttcorS_crust3D_sour, ttcorP_crust3D_rec, ttcorS_crust3D_rec]

  return [outtimes]

def ttcor_corrections(stat_lat,stat_lon,quake_lat,quake_lon,quake_depth,p_ray_param,s_ray_param,moho_pad,rho_c,rho_m,topo,model_nd):

  '''
  Function computing
  the crustal corrections on the receiver's and source's
  end for both P and S waves from ray parameters
  using moho shape from pycrust
  and topography of the planet in the same format (SH coefficients)

  IMPORTANT : consistency between seismic model (model_nd)
              and the parameters used to create the geodetic estimate of Moho shape (moho_pad)
              MUST BE ENSURED (same crustal thickness, same mantle density below Moho)
              This code is checking only consistency of crustal thickness

  Usage
  -----
  [corrections]=ttcor_complete(stat_lat,stat_lon,quake_lat,quake_lon,
                                                                        quake_depth,p_ray_param,s_ray_param,moho_pad,rho_c,rho_m,topo,model_nd)

  Returns
  -----
  corrections (in s): the crustal and topography corrections in a array:
      [P wave correction for the source's end, S wave correction for the source's end,
      P wave correction for the receiver's end, S wave correction for the receiver's end]
      These corrections must be added to the theoretical time in order to obtain the 3D theoretical time

  Parameters
  ------
  stat_lat: station latitude (in deg)
  stat_lon: station longitude (in deg) 
  quake_lat: quake latitude (in deg)
  quake_lon: quake longitude (in deg)
  quake_depth: quake depth (in km)
  p_ray_param: ray parameter of P wave computed into model_nd model (in s/radian)
  s_ray_param: ray parameter of S wave computed into model_nd model (in s/radian)
  

  moho_pad: SHCoeffs class instance containing the radius of the
        crust-mantle interface.(obtained with pyMoho)
        moho_pad= pyshtools.SHCoeffs.from_array(moho.to_array(lmax=topo.lmax))

  rho_c : Crust volumic mass (kg/m/m/m) used to compute the Moho shape moho_pad

  rho_m : Mantle volumic mass (kg/m/m/m) used to compute the Moho shape moho_pad

  topo: SHCoeffs class instance
        Spherical harmonic coefficients of the surface relief.

  model_nd: Text form model of the moon (Named discontinuites *.nd format)

  model_npz: obspy model of the moon
          ( can be created from model_nd by using obspy.taup.taup_create.build_taup_model )

  '''
  import numpy as np
  import obspy
  from obspy.taup import TauPyModel
  from obspy.clients.iris import Client

# extract average radius from topography model
  topo.r0 = topo.coeffs[0, 0, 0]
# extract average radius of Moho
  moho_pad.r0 = moho_pad.coeffs[0, 0, 0]
  lmax = 900
  lmax_calc = 600
# Extract 1D seismological model 
  [modcrust,thickave]=extract_modcrust(model_nd,rho_m)

# average crustal thickness from geodetic model
  geod_thickave = topo.r0 - moho_pad.r0

# Check consistency between seismic and geodetic model at 1 km level
  if (abs(geod_thickave-thickave)>1000.0):
    print('!!! Seismic (1D model) and Geodetic (global average) crustal thicknesses are not consistent !!!')
    print('Seismic crustal thickness = ', thickave , ' meters')
    print('Geodetic global average crustal thickness = ', geod_thickave , ' meters')
    print('!!! 3D corrections computation not performed !!!')
    outtimes=np.zeros((1,4))
    return [outtimes]

  geoparam=np.zeros((1,7))
  geoparam[0,3]=stat_lat
  geoparam[0,4]=stat_lon
  geoparam[0,0]=quake_lat
  geoparam[0,1]=quake_lon
  geoparam[0,2]=quake_depth
  geoparam[0,5]=p_ray_param*np.pi/180
  geoparam[0,6]=s_ray_param*np.pi/180

  topo_source_radius = topo.expand(lat=float(quake_lat), lon=float(quake_lon))/1e3
  topo_receiver_radius = topo.expand(lat=float(stat_lat), lon=float(stat_lon))/1e3
  moho_source_radius = moho_pad.expand(lat=float(quake_lat), lon=float(quake_lon))/1e3
  moho_receiver_radius = moho_pad.expand(lat=float(stat_lat), lon=float(stat_lon))/1e3

# lower radius for computation of correction
# should be smaller than the minimum radius of Moho discontinuity 
  rd = topo.r0/1e3 - 200.0
  outtimes=np.zeros((1,4))

  modcrustradref = build_crust3D(modcrust,(topo.r0-thickave)/1e3,(topo.r0/1e3),topo.r0/1e3,rd)

  modcrustradrec = build_crust3D(modcrust,moho_receiver_radius,
                                  topo_receiver_radius,topo.r0/1e3,rd)
  modcrustradsour = build_crust3D(modcrust,moho_source_radius,
                                  topo_source_radius,topo.r0/1e3,rd)
  [ttcorP_crust3D_sour,ttcorS_crust3D_sour]=crust_topo_cor_sour(modcrustradsour,modcrustradref,geoparam[0,:],topo.r0/1e3)
  [ttcorP_crust3D_rec,ttcorS_crust3D_rec]=crust_topo_cor_rec(modcrustradrec,modcrustradref,geoparam[0,:],topo.r0/1e3)
  outtimes[0,:]=[ttcorP_crust3D_sour ,ttcorS_crust3D_sour,ttcorP_crust3D_rec,ttcorS_crust3D_rec]

  return [outtimes]


def ttcor_complete(stat_lat,stat_lon,quake_lat,quake_lon,quake_depth,moho_pad,rho_c,rho_m,topo,flat,model_nd,model_npz):

  '''
  Function gathering all the routines in one,returning
  the theoretical travel times (1D) predicted with obspy,
  the crustal corrections on the receiver's and source's
  end for both P and S waves and the new calculated travel times (3D)
  using moho shape from pycrust
  and topography of the planet in the same format (SH coefficients)

  IMPORTANT : consistency between seismic model (model_nd)
              and the parameters used to create the geodetic estimate of Moho shape (moho_pad)
              MUST BE ENSURED (same crustal thickness, same mantle density below Moho)
              This code is checking only consistency of crustal thickness

  Usage
  -----
  [corrections,new_travel_times,theoretical_times]=ttcor_complete(stat_lat,stat_lon,quake_lat,quake_lon,
                                                                        quake_depth,moho_pad,rho_c,rho_m,topo,flat,model_nd,model_npz)

  Returns
  -----
  corrections (in s): the crustal and topography corrections in a array:
      [P wave correction for the source's end, S wave correction for the source's end,
      P wave correction for the receiver's end, S wave correction for the receiver's end]
      Computed at vertical incidence if phases are not present

  new_times (in s): 3D travel times = 1D times + corrections
    [P time, S time], return zero if phases are not present

  theoretical_times (in s): 1D travel times calculated with obspy using the input model
    [P time, S time], return zero if phases are not present

  Parameters
  ------
  stat_lat: station latitude (in deg)
  stat_lon: station longitude (in deg) 
  quake_lat: quake latitude (in deg)
  quake_lon: quake longitude (in deg)
  quake_depth: quake depth (in km)

  moho_pad: SHCoeffs class instance containing the radius of the
        crust-mantle interface.(obtained with pyMoho)
        moho_pad= pyshtools.SHCoeffs.from_array(moho.to_array(lmax=topo.lmax))

  rho_c : Crust volumic mass (kg/m/m/m) used to compute the Moho shape moho_pad

  rho_m : Mantle volumic mass (kg/m/m/m) used to compute the Moho shape moho_pad

  topo: SHCoeffs class instance
        Spherical harmonic coefficients of the surface relief.

  flat: flattening parameter of the planet (Earth = 1.0/298.257, Moon = 1.0/900.0, Mars = 0.00589 )

  model_nd: Text form model of the moon (Named discontinuites *.nd format)

  model_npz: obspy model of the moon
          ( can be created from model_nd by using obspy.taup.taup_create.build_taup_model )

  '''
  import numpy as np
  import obspy
  from obspy.taup import TauPyModel
  from obspy.clients.iris import Client

# extract average radius from topography model
  topo.r0 = topo.coeffs[0, 0, 0]
# extract average radius of Moho
  moho_pad.r0 = moho_pad.coeffs[0, 0, 0]
  lmax = 900
  lmax_calc = 600
# Extract 1D seismological model 
  [modcrust,thickave]=extract_modcrust(model_nd,rho_m)

# average crustal thickness from geodetic model
  geod_thickave = topo.r0 - moho_pad.r0

  times_theo=np.zeros((1,2))

  model=TauPyModel(model_npz)
  dist=obspy.geodetics.base.gps2dist_azimuth(stat_lat,stat_lon,quake_lat,quake_lon,topo.r0,flat)
  d=dist[0]*180.0/(topo.r0*np.pi)
  arrivals_P=model.get_travel_times(quake_depth ,d,["P","p"])
  arrivals_S=model.get_travel_times(quake_depth ,d,["S","s"])
  if (len(arrivals_P) > 0):
    arrival_p=arrivals_P[0]
    arrival_p_time=arrival_p.time
    p_ray_param=arrival_p.ray_param
  else:
    arrival_p_time=0.0
    p_ray_param=0.0
  if (len(arrivals_S) > 0):
    arrival_s=arrivals_S[0]
    arrival_s_time=arrival_s.time
    s_ray_param=arrival_s.ray_param
  else:
    arrival_s_time=0.0
    s_ray_param=0.0
  times_theo[0,:]=[arrival_p_time,arrival_s_time]

# Check consistency between seismic and geodetic model at 1 km level
  if (abs(geod_thickave-thickave)>1000.0):
    print('!!! Seismic (1D model) and Geodetic (global average) crustal thicknesses are not consistent !!!')
    print('Seismic crustal thickness = ', thickave , ' meters')
    print('Geodetic global average crustal thickness = ', geod_thickave , ' meters')
    print('!!! 3D corrections computation not performed !!!')
    new_times=np.zeros((1,2))
    outtimes=np.zeros((1,4))
    return [outtimes,new_times,times_theo]

  geoparam=np.zeros((1,7))
  geoparam[0,3]=stat_lat
  geoparam[0,4]=stat_lon
  geoparam[0,0]=quake_lat
  geoparam[0,1]=quake_lon
  geoparam[0,2]=quake_depth
  geoparam[0,5]=p_ray_param*np.pi/180
  geoparam[0,6]=s_ray_param*np.pi/180

  topo_source_radius = topo.expand(lat=float(quake_lat), lon=float(quake_lon))/1e3
  topo_receiver_radius = topo.expand(lat=float(stat_lat), lon=float(stat_lon))/1e3
  moho_source_radius = moho_pad.expand(lat=float(quake_lat), lon=float(quake_lon))/1e3
  moho_receiver_radius = moho_pad.expand(lat=float(stat_lat), lon=float(stat_lon))/1e3

# lower radius for computation of correction
# should be smaller than the minimum radius of Moho discontinuity 
  rd = topo.r0/1e3 - 200.0
  new_times=np.zeros((1,2))
  outtimes=np.zeros((1,4))

  modcrustradref = build_crust3D(modcrust,(topo.r0-thickave)/1e3,(topo.r0/1e3),topo.r0/1e3,rd)

  modcrustradrec = build_crust3D(modcrust,moho_receiver_radius,
                                  topo_receiver_radius,topo.r0/1e3,rd)
  modcrustradsour = build_crust3D(modcrust,moho_source_radius,
                                  topo_source_radius,topo.r0/1e3,rd)
  [ttcorP_crust3D_sour,ttcorS_crust3D_sour]=crust_topo_cor_sour(modcrustradsour,modcrustradref,geoparam[0,:],topo.r0/1e3)
  [ttcorP_crust3D_rec,ttcorS_crust3D_rec]=crust_topo_cor_rec(modcrustradrec,modcrustradref,geoparam[0,:],topo.r0/1e3)
  outtimes[0,:]=[ttcorP_crust3D_sour,ttcorS_crust3D_sour,ttcorP_crust3D_rec,ttcorS_crust3D_rec]
  if (times_theo[0,0]>0.0000001):
    new_times[0,0]=times_theo[0,0]+ttcorP_crust3D_sour+ttcorP_crust3D_rec
  else:
    new_times[0,0]=0.0
  if (times_theo[0,1]>0.0000001):
    new_times[0,1]=times_theo[0,1]+ttcorS_crust3D_sour+ttcorS_crust3D_rec
  else:
    new_times[0,1]=0.0

  return [outtimes,new_times,times_theo]




def extract_modcrust(model_nd,rho_m) :

  '''
  Function creating the simplified model of crust
  used in the module, containing the
  different layers of the crust, based on a
  .nd model of the moon
  if values are not constant in layers,
  layers of constant average values are created
  mantle layer as constant values equal to the values just below the crust

  Usage
  -----
  modcrust= extract_modcrust('VPREMOON.nd')
  
  Return
  -----
  modcrust seismic model, format by layer : layer number, layer thickness, Vp, Vs, density
  Crustal thickness in meters

  Parameters
  -----
  model_nd = name of seismic model nd file
  rho_m = mantle density below Moho used to construct Moho shape (kg/m*m*m)

  '''
  import os
  import numpy as np

  if os.path.isfile(model_nd):
    f=open(model_nd,"r+",)
    lines = f.readlines()
    nclay=0
#    print(len(lines))
    for i in range(len(lines)):
      if (lines[i]=='mantle\n'):
        thickave=np.fromstring(lines[i-1], dtype=float, sep=' ')[0]*1000
        break
      elif (lines[i+1]=='mantle\n'):
        if(np.fromstring(lines[i], dtype=float, sep=' ')[0]==np.fromstring(lines[i+2], dtype=float, sep=' ')[0]):
          nclay+=1
      else :
        if(np.fromstring(lines[i], dtype=float, sep=' ')[0]<np.fromstring(lines[i+1], dtype=float, sep=' ')[0]):
          nclay+=1          
    modcrust = np.zeros((nclay,5))
    j=0
    for i in range(len(lines)):
      if (lines[i]=='mantle\n'):
        thickave=np.fromstring(lines[i-1], dtype=float, sep=' ')[0]*1000
        modcrust[j,0]=j+1
        modcrust[j,1]= 200.0
        modcrust[j,2:]=np.fromstring(lines[i+1], dtype=float, sep=' ')[1:4]
        modcrust[j,4]=rho_m/1000.0
        break
      elif (lines[i+1]=='mantle\n'):
        pass
      else :
        if(np.fromstring(lines[i], dtype=float, sep=' ')[0]<np.fromstring(lines[i+1], dtype=float, sep=' ')[0]):
          modcrust[j,0]=j+1
          modcrust[j,1]=np.fromstring(lines[i+1], dtype=float, sep=' ')[0]-np.fromstring(lines[i], dtype=float, sep=' ')[0]
          modcrust[j,2:]=(np.fromstring(lines[i+1], dtype=float, sep=' ')[1:4]+np.fromstring(lines[i], dtype=float, sep=' ')[1:4])/2
          j+=1
  else :
    print('!!! Seismic model file not available !!!')
    modcrust = np.zeros((4,5))    

  return [modcrust,thickave]


def build_crust3D(modcrust,mohorad,toporad,r0,rd):

  '''

  Function to construct crustal seismic structure
  creating an homotetic version of reference seismic model
  keeping intact the first layer of regolith

  Returns
  -----
  modcrustrad: a 3D model of the crust,
    same format than modcrust + 2 additional columns (bottom and top radii of the layer)

  Parameters:
  -----
  modcrust: reference model of the crust created by extract_modcrust from *.nd model

  mohorad: moho depth below r0 (in km)

  toporad: topography height above r0 (in km)

  r0: average radius of the moon (in km)

  rd: starting integration radius for computing 3D corrections (in km)

  '''
  import numpy as np

  nl = len(modcrust)
  modcrustrad = np.zeros((nl,7))
  modcrustrad[0,0:5] = modcrust[0,0:5]
  modcrustrad[0,5] = toporad-modcrust[0,1]
  modcrustrad[0,6] = toporad
  thick = 0.0
  thick3D=modcrustrad[0,5]-mohorad

  for i in range(1,nl-1):
    thick=thick+modcrust[i,1]

  rescale=thick3D/thick
  for i in range(1,nl-1):
    modcrustrad[i,0:5] = modcrust[i,0:5]
    modcrustrad[i,1] = modcrustrad[i,1] * rescale
    modcrustrad[i,6] = modcrustrad[i-1,5]
    modcrustrad[i,5] = modcrustrad[i,6] - modcrustrad[i,1]

  i=nl-1
  modcrustrad[i,0:5] = modcrust[i,0:5]
  modcrustrad[i,6] = modcrustrad[i-1,5]
  modcrustrad[i,5] = rd
  modcrustrad[i,1] = modcrustrad[i,6] - modcrustrad[i,5]

  return modcrustrad;


def crust_topo_cor_rec(modcrustrad,modcrustradref,geoparam,r0):

  '''
  function computing crustal and topographic correction at the receiver's end
  assuming modcrust is the crust model including topography

  Returns:
  -----
  ttcorP_crust3D: 3D time correction on the receiver's end for P wave
  ttcorS_crust3D: 3D time correction on the receiver's end for S wave
  These corrections must be added to 1D theoretical times in order to obtain 3D theoretical time
  These corrections contains both crustal thickness and topography effects
  (all differences between 3D and 1D models)

  Parameters:
  ------
  modcrustrad: 3D model of the crust built with build_crust3D
  modcrustradref: reference seismic model of the crust
  geoparam: geographic and geometric parameters :
            [quake lat, quake lon, quake depth, station lat, station lon, P ray parameter, S ray parameter]
  r0: average radius of the moon
  '''

  # ray parameters
  pp=geoparam[5]
  ps=geoparam[6]

  # compute in reference model
  nlr = len(modcrustradref)
  taupref=0.0
  tausref=0.0
  for i in range(nlr):
    [tpdum,tsdum]=compute_tau(modcrustradref[i,5],modcrustradref[i,6],modcrustradref[i,2],modcrustradref[i,3],pp,ps)
    taupref=taupref+tpdum
    tausref=tausref+tsdum

  # compute in 3D crust model
  nl = len(modcrustrad)
  taup=0.0
  taus=0.0
  for i in range(nl):
    [tpdum,tsdum]=compute_tau(modcrustrad[i,5],modcrustrad[i,6],modcrustrad[i,2],modcrustrad[i,3],pp,ps)
    taup=taup+tpdum
    taus=taus+tsdum

  ttcorP_crust3D=taup-taupref
  ttcorS_crust3D=taus-tausref

  return ttcorP_crust3D,ttcorS_crust3D;

def crust_topo_cor_sour(modcrustrad,modcrustradref,geoparam,r0):

  '''
  function computing crustal and topographic correction at the source's end
  assuming modcrust is the crust model including topography


  Returns:
  -----
  ttcorP_crust3D: 3D time correction on the source's end for P wave
  ttcorS_crust3D: 3D time correction on the source's end for S wave
  These corrections must be added to 1D theoretical times in order to obtain 3D theoretical time
  These corrections contains both crustal thickness and topography effects
  (all differences between 3D and 1D models)

  Parameters:
  ------
  modcrustrad: 3D model of the crust built with build_crust3D
  modcrustradref: reference seismic model of the crust
  geoparam: geographic and geometric parameters :
            [quake lat, quake lon, quake depth, station lat, station lon, P ray parameter, S ray parameter]
  r0: average radius of the moon
  '''

  # ray parameters
  pp=geoparam[5]
  ps=geoparam[6]


  radsour=r0-geoparam[2]
  # compute in reference model
  nlr = len(modcrustradref)
  taupref=0.0
  tausref=0.0
  flag = 0
  for i in range(nlr):
    if (flag<=1):
      #finding in what layer the quake happened and start integration from there
      if ((radsour-modcrustradref[i,5])>0) and ((radsour-modcrustradref[i,6])<=0):
        r1=modcrustradref[i,5]
        r2=radsour
        flag=1
      else:
        r1=modcrustradref[i,5]
        r2=modcrustradref[i,6]
    if (flag>0):
      [tpdum,tsdum]=compute_tau(r1,r2,modcrustradref[i,2],modcrustradref[i,3],pp,ps)
      taupref=taupref+tpdum
      tausref=tausref+tsdum

  # compute in 3D crust model
  if (geoparam[2]<0.1):
    # assume topography is used for impacts if the quake is on the surface
    radsour=modcrustrad[0,6]-geoparam[2]
  else:
    # radius of the source assumed to be computed from reference radius
    # for seismic events
    radsour=r0-geoparam[2]

  nl = len(modcrustrad)
  taup=0.0
  taus=0.0
  flag = 0
  for i in range(nl):
    if (flag<=1):
      if ((radsour-modcrustrad[i,5])>0) and ((radsour-modcrustrad[i,6])<=0):
        r1=modcrustrad[i,5]
        r2=radsour
        flag=1
      else:
        r1=modcrustrad[i,5]
        r2=modcrustrad[i,6]
    if (flag>0):
      [tpdum,tsdum]=compute_tau(r1,r2,modcrustrad[i,2],modcrustrad[i,3],pp,ps)
      taup=taup+tpdum
      taus=taus+tsdum

  ttcorP_crust3D=taup-taupref
  ttcorS_crust3D=taus-tausref

  return ttcorP_crust3D,ttcorS_crust3D;


def compute_tau(r1,r2,vp,vs,pp,ps):

  '''
  Function calculating the intercept time in between r1 and r2 (in km)
  for p and s waves of ray parameters pp and ps in s/deg
  assuming constant velocity vp and vs (km/s) in that layer

  Formula from "A Brievary of seismic tomography", chap 13.3, Guust Nolet
  '''
  import numpy as np

  # convert from s/deg to s/rad
  npp=pp*180.0/np.pi
  nps=ps*180.0/np.pi
  # test r2>r1
  sign=1.0
  if r2<r1:
    sign=r1
    r1=r2
    r2=sign
    sign=-1.0

  # compute integral (analytical from integrals.com)
  a=1/vp
  b=pp*pp*(vp*vp)
  taup = tau_int(a,b,r1,r2)*sign
  a=1/vs
  b=ps*ps*(vs*vs)
  taus = tau_int(a,b,r1,r2)*sign

  return taup,taus;

def tau_int(a,b,r1,r2):
  '''
  Function computing the tau integral in a constant velocity (c) layer
  in between r1 and r2
  '''
# analytical formula from integrals.com
# assume a= 1/c
# b=p*p*(c*c)
  import numpy as np

  x=r1
  tau1=a*(np.sqrt(x*x-b) + np.sqrt(b)*np.arctan(np.sqrt(b)/np.sqrt(x*x-b)))
  x=r2
  tau2=a*(np.sqrt(x*x-b) + np.sqrt(b)*np.arctan(np.sqrt(b)/np.sqrt(x*x-b)))
  tau=tau2-tau1

  return tau;

##### Other functions developped for internal purposes #######
##### Some are not fully validated  #########
##### Some are old versions of actual functions ######

def create_model(thickave):
  '''
  Funtion creating a new .nd moon model in a 'Models' file
  with a given average thickness of the crust
  based on the VPREMOON model

  '''
  f=open('Models/VPREMOON'+str(int(thickave))+'.nd',"w+")
  g=open('../VPREMOON_model_2011/VPREMOON.nd',"r+")
  lines=g.readlines()
  for i in range(5):
    f.write(lines[i])
  f.write(str(thickave)+' 5.5 3.3 2.762 6750 6750\n')
  f.write(lines[6])
  f.write(str(thickave)+' 7.54 4.34 3.312 6750 6750\n')
  for i in range(8,len(lines)):
    f.write(lines[i])
  f.close

def create_model_V(thickave,V_P):
  '''
  Funtion creating a new .nd moon model in a 'Models' file
  with a given average thickness of the crust
  and a given P wave velocity in the crust
  S wave velocity is set by V_S=0.6*V_P
  based on the VPREMOON model

  '''
  f=open('Models/Inversion_V/VPREMOON'+str(int(thickave))+'_'+str(round(V_P,1))+'.nd',"w+")
  g=open('../VPREMOON_model_2011/VPREMOON.nd',"r+")
  lines=g.readlines()
  for i in range(4):
    f.write(lines[i])
  f.write(str(12)+' ' +str(V_P)+ ' ' +str(round(0.6*V_P,3))+' 2.762 6750 6750\n')
  f.write(str(thickave)+' ' +str(V_P)+ ' ' +str(round(0.6*V_P,3))+' 2.762 6750 6750\n')

  f.write(lines[6])
  f.write(str(thickave)+' 7.54 4.34 3.312 6750 6750\n')
  for i in range(8,len(lines)):
    f.write(lines[i])
  f.close

def create_modcrust():
  import os
  import numpy as np

  crustfile = './input_example_pycrust_Moon_ttcor2.dat'

  if os.path.isfile(crustfile):
    f=open(crustfile,"r+",)
    lines = f.readlines()
    myarray = np.fromstring(lines[0], dtype=float, sep=',')
    thickave = myarray[0]
    rho_m = myarray[1]
    modcrust = np.zeros((4,5))
    modcrust[0,:] = np.fromstring(lines[1], dtype=float, sep=',')
    modcrust[1,:] = np.fromstring(lines[2], dtype=float, sep=',')
    modcrust[2,:] = np.fromstring(lines[3], dtype=float, sep=',')
    modcrust[3,:] = np.fromstring(lines[4], dtype=float, sep=',')
  return [modcrust,rho_m,thickave]

