# Dose rate calculator v0.6
#
# Code for calculating dose rates from SILAM netcdf files
#
# Usage: python doserate_v1_0.py <input_folder> <heights> <e>
# <input_folder> : path to folder where the input netcdf files (SILAM output files) are located
# <heights> : heights of the planes where the dose rate is to be calculated, 
# given in meters and as sequence separated by commas (without any spaces!)
# <e> : e -> output as function of energy, missing or something else: output not as function of energy
#
# output of the files: netcdf files with dose rates from in-air, dry-deposited and wet-deposited radioactive species
# the output files are written into the same folder as the input files
#
# The dose rate calculator finds all files ending with .nc and .nc4 in the input folder, and then looks for activities 
# given in Bq/m3 in variables starting with 'cnc', 'wd' and 'dd'.
#
# The dose rate calculator approximates the radioactive cloud to be semi-infinite in the xy-plane, which is a good 
# approximation for normal dispersion model output (horizontal cell size of the order of 1 km or more). The dose rates
# are integrated using cylindrical coordinates. 2D numerical integration is applied despite that the problem is strictly
# 1D, as the relevant integral lacks an analytical solution. In case faster computation times are needed, the code could
# be modified to use tabulated values for the redundant dimension.
#
# Vertical integration weights are calculated for 14 different energies (listed in the 'data' dictionary below), with 
# interpolation of the weights applied for all intermediate emission energies. The vertical integration weights are 
# calculated only at the start, or when a new file has different vertical levels than the previous one.
#
# Possible metastable daughters are checked for having same emission energies as the parent nuclide. If such energies are
# found, they are removed (as nuclide databases may double count the same emission for both the parent and metastable
# daughter).
#
# The air density is taken from a standard atmosphere, assuming sea level pressure at zero height. The use of Berger's
# coefficients, derived for a homogeneous air density, introduces errors when calculating the dose rate across
# a vertically thick and thus inhomogeneous portion of the atmosphere, but the attenuation across such a thick portion
# (so thick that it renders the air density significantly inhomogeneous) is substantial and should render the absolute
# value of the error small for most cases.
#
# In case the input data is a function of hybrid levels, it should contain a variable that maps the hybrid levels
# into height in meters as a function of a standard atmosphere (this is true for SILAM output).

import numpy as np
from netCDF4 import Dataset
import os, sys, datetime as dt

# nuclib module by STUK (the custom query the emission data from the STUK database)
from nuclib import custom_query

# data matrix based on 
# CALCULATION OF GAMMA-RAY DOSE RATE FROM AIRBORNE AND DEPOSITED ACTIVITY,
# CERC, P20/01N/17, August2017
# https://www.cerc.co.uk/environmental-software/assets/data/doc_techspec/P20_01.pdf

# {photon energy (MeV) : 
# [mu0 (1/m), Berger's a, Berger's b, mu_absorption*energy (Gy m**2), conversion fact. (Sv/Gy)]}
# mu0 is the attenuation coefficient at sea level
data = {0.01 : [0.623, 0.025, -0.0464, 7.43e-16, 0.00296],
          0.015 : [0.187, 0.0947, -0.0484, 3.12e-16, 0.0183],
          0.02 : [0.0893, 0.2652, -0.0463, 1.68e-16, 0.0543],
          0.03 : [0.0411, 1.055, -0.0192, 0.721e-16, 0.191],
          0.05 : [0.0253, 3.498, 0.0729, 0.323e-16, 0.557],
          0.065 : [0.0226, 4.209, 0.1169, 0.278e-16, 0.63],
          0.1 : [0.0195, 4.033, 0.1653, 0.371e-16, 0.765],
          0.2 : [0.0159, 2.678, 0.1678, 0.856e-16, 0.703],
          0.5 : [0.0112, 1.748, 0.1014, 2.38e-16, 0.689],
          1.0 : [0.00821, 1.269, 0.0559, 4.47e-16, 0.732],
          1.5 : [0.00668, 1.040, 0.0338, 6.12e-16, 0.765],
          2.0 : [0.00574, 0.891, 0.0215, 7.50e-16, 0.791],
          4.0 : [0.00398, 0.5879, 0.0022, 12.0e-16, 0.850],
          10 : [2.65e-3, 0.3113, -0.0194, 23.1e-16, 0.935]}

#computes the relative air density for a standard atmosphere from height in meters
def rel_air_dens(height):
    ad_0 = 1.225
    p_0 = 101325
    T_0 = 288.15
    R=287.04
    g=9.81

    ad = np.zeros(len(height))

    for i, h in enumerate(height):    
        if h <= 11000:
            p = p_0* (1-0.0065*(h/T_0))**5.2561
            T = T_0 - h*6.5e-3
            ad[i] = p/(R*T)
        else:
            T_11 = 216.65
            p_11 = 22632
            p = p_11 * np.exp(-g/(R*T_11)*(h-11000))
            ad[i] = p/(R*T_11)

    return ad/ad_0

# Calculation of integration weights for the activity concentrations in a column of cells.
# The activity is approximated to be infinite in the xy plane for a given column.
# A cylindrical coordinate system is used.
def calc_weights(height, thickness, ad, dr_h, inds_dr_h, mu0, a, b):
    weights = np.zeros(len(height))
        
    for i in range(len(height)):
        bottom = np.sum(thickness[:i])
        top = np.sum(thickness[:i+1])

        #mu0 is scaled with the average relative air density
        if i in inds_dr_h:
            ad_scaling = ad[i]
        else:
            i_dr_h = np.min(inds_dr_h)

            if dr_h > top:
                ad_scaling = np.dot(ad[i+1:i_dr_h], thickness[i+1:i_dr_h])
                ad_scaling += ad[i_dr_h]*(dr_h-np.sum(thickness[:i_dr_h]))
            elif dr_h < bottom:
                ad_scaling = np.dot(ad[i_dr_h+1:i], thickness[i_dr_h+1:i])
                ad_scaling += ad[i_dr_h]*(np.sum(thickness[:i_dr_h+1])-dr_h)
            else:
                ad_scaling = 0

            ad_scaling = (ad_scaling + 0.5*thickness[i]*ad[i])/abs(dr_h-height[i])

        mu = mu0*ad_scaling

        # very rough optimization of the integration step length
        if dr_h >= bottom and dr_h <= top:
            drho = 0.05/mu**0.33
            dz = 0.05/mu**0.33
        else:
            drho = abs(height[i]-dr_h)*0.001/mu**0.33
            dz = abs(height[i]-dr_h)*0.001/mu**0.33

        # 1500 m radius of the integration area in the xy plane
        rho_range = np.arange(0, 1500, drho) + drho/2.
        z_range = np.arange(bottom-dr_h, top-dr_h, dz) + dz/2.
        z, rho = np.meshgrid(z_range, rho_range)
        r2 = z**2 + rho**2
        r = r2**0.5
        weights[i] += np.sum((rho*drho*dz) * 0.5 * (1+a*mu*r*np.exp(b*mu*r)) * np.exp(-mu*r2**0.5)/r2)

    return weights

# weight for the deposition                                                                                                                                                     
# the deposited activity is approximated to be infinte and homogeneous in the xy plane
def calc_weight_depo(thickness, ad, dr_h, inds_dr_h, mu0, a, b):
    i_dr_h = np.min(inds_dr_h)
    mu = mu0*(np.dot(ad[:i_dr_h], thickness[:i_dr_h]) + (dr_h-np.sum(thickness[:i_dr_h]))*ad[i_dr_h])/dr_h
    drho = 0.05
    rho = np.arange(0, 1500, drho) + drho/2
    r2 = dr_h**2 + rho**2
    r = r2**0.5
    return np.sum((rho*drho) * 0.5 * (1+a*mu*r*np.exp(b*mu*r)) * np.exp(-mu*r2**0.5)/r2)

def convert_silam_name(species):
    species = list(species.replace('_','-').lower())
    species[0] = species[0].upper()
    if species[-1] == 'm':
        species[-1] = 'M'
    return ''.join(species)

def get_ems_data(species, species_list):
    species = convert_silam_name(species)

    energies = []
    fractions = []
    # Possible metastable daughters:
    metastable_daughters = []

    lines = custom_query(species, 'libLines', False, 'lineType,energy,emissionProb,daughterNuclideId')
    lines = ''.join(lines).split('\n')

    for line in lines:
        line = line.split(' ')
        if (line[0] == 'G' or line[0] == 'X') and line[3] != species and line[2] != 'None':
            energies.append(float(line[1])*1e-3)
            fractions.append(float(line[2]))
            if not species.endswith('M') and line[3]+'M' in species_list:
                metastable_daughters.append(line[3] + 'M')
    return energies, fractions, metastable_daughters

def read_branching(species, species_list):
    energies_in, fractions_in, metastable_daughters = get_ems_data(species, species_list)
    
    if len(metastable_daughters) == 0:
        # no possible metastable daughters
        return energies_in, fractions_in

    for daughter in metastable_daughters:
        # possible metastable daughters, whose emissions may be double counted in the
        # database, and need thus to be removed
        energies = []
        fractions = []
        if daughter in species_list:
            energies_d, _, _ = get_ems_data(daughter, species_list)

            for ie, energy in enumerate(energies_in):
                # if the emission energy of the metastable daughter equals the                                                                                                                                 
                # emission energy of the parent (in the database), it is assumed to 
                # be a double counting and thus not included in the emissions
                if not (energy in energies_d):
                    energies.append(energy)
                    fractions.append(fractions_in[ie])
            energies_in = energies
            fractions_in = fractions

    return energies, fractions

def init_netcdf(filename, times, heights, lats, lons, lat_unit, lon_unit, 
                time_unit, energy_dim_in_output, energies, nc_attributes):
    # write NetCDF file coordinate variables and metadata
    df_out = Dataset(filename, mode='w')
    # set global NetCDF attributes
    for key, value in nc_attributes.items():
        df_out.setncattr(key,value)

    if energy_dim_in_output:
        df_out.createDimension('energy', len(energies))
        energy = df_out.createVariable('energy', np.float32, ('energy'))
        energy[:] = energies
        energy.units = 'MeV'

    df_out.createDimension('time', len(times))
    df_out.createDimension('height', len(heights))
    df_out.createDimension('lat', len(lats))
    df_out.createDimension('lon', len(lons))
        
    time = df_out.createVariable('time', 'i', ('time',))
    height = df_out.createVariable('height', 'f', ('height',))
    lat = df_out.createVariable('lat', 'f', ('lat',))
    lon = df_out.createVariable('lon', 'f', ('lon',))

    # set units
    time.units = time_unit
    height.units = 'm'
    lat.units = lat_unit
    lon.units = lon_unit

    # set axes and other metadata
    lon._CoordinateAxisType = "Lon"
    lon.axis = "X"
    lon.standard_name = "longitude"
    lon.long_name = "longitude"
    lat._CoordinateAxisType = "Lat"
    lat.axis = "Y"
    lat.standard_name = "latitude"
    lon.long_name = "latitude"
    height.positive = "up"
    height.axis = "Z"
    height.standard_name = "height"
    height.long_name = "height"
    time.long_name = "time"
    time.axis = "T"
    time.standard_name = "time"
    time.calendar = "standard"
    
    time[:] = times
    height[:] = heights
    lat[:] = lats
    lon[:] = lons
    
    return df_out

def write_field(df_out, ncfile, field_name, vals, unit, energy_dim_in_output):
    print('writing field:', field_name)
    if energy_dim_in_output:
        field = df_out.createVariable(field_name, 'f', ('energy','time','height','lat','lon'), zlib=True)
    else:
        field = df_out.createVariable(field_name, 'f', ('time','height','lat','lon'), zlib=True)
    field.units = unit
    field[:] = vals
    return df_out

if __name__ == '__main__':

    input_folder = sys.argv[1]
    if not os.path.exists(input_folder):
        print('The folder', input_folder, 'does not exist')
        quit()

    # the output folder is here hard-coded to be the input folder (but it could be set as an argument instead)
    output_folder = input_folder

    # in case some other folder is selected for the output
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # the input files must end with .nc or .nc4
    ncfiles = np.sort([f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and (f.endswith('.nc') or f.endswith('.nc4'))])
    if len(ncfiles) == 0:
        print('No netCDF files found in folder', input_folder)
        quit()

    # the heights for the dose rate calculation in meters
    dr_calc_height_m = np.float32(np.array(sys.argv[2].split(',')))

    for dr_h in dr_calc_height_m:
        if dr_h < 0:
            print('negative height in the input data:' %dr_h)
            quit()

    energy_dim_in_output = False
    if len(sys.argv) > 3:
        if sys.argv[3] == 'e':
            energy_dim_in_output = True

    if energy_dim_in_output:
        print('output fields as functions of energy, time, height, lat, and lon')
    else:
        print('output fields as functions of time, height, lat, and lon')

    height_old = []
    variables_old = []

    for ncfile in ncfiles:
        print('new input file:', os.path.join(input_folder, ncfile))
        df = Dataset(os.path.join(input_folder, ncfile),mode='r')

        if 'height' in df.variables.keys():
            # The input data is a function of height in meters. The height refers to model layer midpoints.
            height = df.variables['height'][:]
        else:
            # If the model output is as function of hybrid levels, it should contain a 1D variable named 'hybrid' that contains
            # the heights of the hybrid levels based on a standard atmosphere.
            if 'hybrid' in df.variables.keys():
                print('no height dimension found in the input file, using hybrid converted to height in a standard atmosphere')
                height = df.variables['hybrid'][:]
            else:
                print('no height or hybrid dimension found in the input file, skipping')
                continue

        variables_cnc = [v for v in df.variables.keys() if v.startswith('cnc') and df.variables[v].units == 'Bq/m3']
        if len(variables_cnc) == 0:
            print('no radioactive species found in the input file, skipping')
            continue

        # the thicknesses of the model layers
        thickness = np.zeros(len(height))
        bottom = 0
        for i in range(len(height)):
            top = bottom+2*(height[i]-bottom)
            thickness[i] = top-bottom
            bottom=top

        if np.max(dr_calc_height_m) > np.sum(thickness):
            print('the maximum requested height, %f m, is outside the domain, skipping file' %np.max(dr_calc_height_m))
            continue

        lats = df.variables['lat'][:]
        lons = df.variables['lon'][:]
        time = df.variables['time'][:]

        lat_unit = df.variables['lat'].units
        lon_unit = df.variables['lon'].units
        time_unit = df.variables['time'].units
    
        filename_out, _ = os.path.splitext(ncfile)
        filename_out += '_dose_rate.nc4'
        filename_out = os.path.join(output_folder, filename_out)

        # If the file has different cell heights than the previous file (or if it is the first file),
        # the integration weights need to be computed.
        # For a single SILAM run, this is needs to be done only once
    
        if not (height in height_old and len(height) == len(height_old)):
            print('computing weights for vertical integration')
            weights_m = np.zeros((len(dr_calc_height_m), len(data), len(height)))
            weights_depo = np.zeros((len(dr_calc_height_m), len(data)))

            # relative air density                                                                                                                                                                                
            ad = rel_air_dens(height)

            for ih_out, dr_h in enumerate(dr_calc_height_m):

                inds_dr_h = []
                for i in range(len(height)):
                    bottom = np.sum(thickness[:i])
                    top = np.sum(thickness[:i+1])
                    if dr_h >= bottom and dr_h <= top:
                        inds_dr_h.append(i)

                for ie, energy in enumerate(data):
                    vals = data[energy]
                    weights_m[ih_out, ie] = calc_weights(height, thickness, ad, dr_h, inds_dr_h, vals[0], vals[1], vals[2])
                    weights_depo[ih_out, ie] = calc_weight_depo(thickness, ad, dr_h, inds_dr_h, vals[0], vals[1], vals[2])
            height_old = height

            print('done')

        # If the cell has different species than the previous file (or if it is the first file),
        # the interpolation weights of the energies need to be computed
        # For a single SILAM run, this is needs to be done only once
    
        if not variables_cnc == variables_old:
            print('computing weights for energy interpolation')
            species_list = []
            for var in variables_cnc:
                species_list.append(convert_silam_name(df.variables[var].substance_name))

            var_weights = np.zeros((len(variables_cnc), len(data)))
            energies_out = np.array(list(data.keys()))

            for ivar, var in enumerate(variables_cnc):
                species = convert_silam_name(df.variables[var].substance_name)
                var_energies, var_ratios = read_branching(species, species_list)
                if len(var_energies) == 0 or len(var_ratios) == 0:
                    print('***************************************')
                    print('WARNING!!!!')
                    print('no emission data for %s' % species)
                    print('***************************************')
                for ievar, evar in enumerate(var_energies):
                    ind = np.where(evar <= energies_out)[0][0]-1
                    if ind < 0:
                        continue
                    # exponential growth assumed between the interpolation values
                    frac = (np.exp(evar)-np.exp(energies_out[ind]))/(np.exp(energies_out[ind+1])-np.exp(energies_out[ind]))
                    var_weights[ivar,ind] += (1-frac) * var_ratios[ievar]
                    var_weights[ivar,ind+1] += frac * var_ratios[ievar]
            variables_old = variables_cnc
            print('done')

        # read SILAM file global attributes
        attrs = ("Conventions", "source", "_CoordinateModelRunDate", "grid_projection",
                 "pole_lat", "pole_lon")
        nc_attributes = {}
        for name in df.ncattrs():
            if name in attrs:
                nc_attributes[name] = getattr(df, name)
        # set history variable
        nc_attributes["history"] = "Created by the dose rate calculator v0.6 from {}".format(df.filepath())

        print('initializing output file:', filename_out)
        df_out = init_netcdf(filename_out, time, dr_calc_height_m, lats, lons, 
                             lat_unit, lon_unit, time_unit, energy_dim_in_output, 
                             energies_out, nc_attributes)

        for ivar, variable in enumerate(variables_cnc):
            calc = [True, False, False]
            cnc = df.variables[variable][:]
            
            if len(cnc.shape) != 4:
                print('concentrations need to be 4D (time, height/hybrid, lat, lon), skipping variable', variable)
                calc[0] = False

            if 'dd' + variable[3:] in df.variables.keys():
                calc[1] = True
                dd = df.variables['dd' + variable[3:]][:]
                if len(dd.shape) != 3:
                    print('depositions need to be 3D (time, lat, lon), skipping variable', 'dd' + variable[3:])
                    calc[1] = False
            else:
                print('No dry deposition of %s in the input file' %variable[3:])

            if 'wd' + variable[3:] in df.variables.keys():
                calc[2] = True
                wd = df.variables['wd' + variable[3:]][:]
                if len(wd.shape) != 3:
                    print('depositions need to be 3D (time, lat, lon), skipping variable', 'wd' + variable[3:])
                    calc[2] = False
            else:
                print('No wet deposition of %s in the input file' %variable[3:])

            if energy_dim_in_output:
                doserate_cnc = np.zeros((len(energies_out), len(time), len(dr_calc_height_m), len(lats), len(lons)))
                doserate_wd = np.zeros((len(energies_out), len(time), len(dr_calc_height_m), len(lats), len(lons)))
                doserate_dd = np.zeros((len(energies_out), len(time), len(dr_calc_height_m), len(lats), len(lons)))
            else:
                doserate_cnc = np.zeros((len(time), len(dr_calc_height_m), len(lats), len(lons)))
                doserate_wd = np.zeros((len(time), len(dr_calc_height_m), len(lats), len(lons)))
                doserate_dd = np.zeros((len(time), len(dr_calc_height_m), len(lats), len(lons)))
        
            for it in range(len(time)):
                for ih_out in range(len(dr_calc_height_m)):
                    for ie, energy in enumerate(energies_out):
                        if energy_dim_in_output:
                            if calc[0]:
                                doserate_cnc[ie, it, ih_out] += var_weights[ivar, ie]*data[energy][3]*data[energy][4]*np.dot(cnc[it].T, weights_m[ih_out, ie]).T
                            if calc[1]:
                                doserate_dd[ie, it, ih_out] += var_weights[ivar, ie]*data[energy][3]*data[energy][4]*dd[it]*weights_depo[ih_out, ie]
                            if calc[2]:
                                doserate_wd[ie, it, ih_out] += var_weights[ivar, ie]*data[energy][3]*data[energy][4]*wd[it]*weights_depo[ih_out, ie]
                        else:
                            if calc[0]:
                                doserate_cnc[it, ih_out] += var_weights[ivar, ie]*data[energy][3]*data[energy][4]*np.dot(cnc[it].T, weights_m[ih_out, ie]).T
                            if calc[1]:
                                doserate_dd[it, ih_out] += var_weights[ivar, ie]*data[energy][3]*data[energy][4]*dd[it]*weights_depo[ih_out, ie]
                            if calc[2]:
                                doserate_wd[it, ih_out] += var_weights[ivar, ie]*data[energy][3]*data[energy][4]*wd[it]*weights_depo[ih_out, ie]

            if calc[0]:
                df_out = write_field(df_out, filename_out, 'dose_rate_cloud' + variable[3:], doserate_cnc, 'Sv/s', energy_dim_in_output)
            if calc[1]:
                df_out = write_field(df_out, filename_out, 'dose_rate_wd' + variable[3:], doserate_dd, 'Sv/s', energy_dim_in_output)
            if calc[2]:
                df_out = write_field(df_out, filename_out, 'dose_rate_dd' + variable[3:], doserate_wd, 'Sv/s', energy_dim_in_output)

        df.close()
        df_out.close()
