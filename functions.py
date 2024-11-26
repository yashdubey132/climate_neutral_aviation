import pandas as pd
import xarray as xr
#import netCDF4, h5netcdf
import glob
from importlib import reload
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import uncertainties
from uncertainties import unumpy, ufloat
from fair.forward import fair_scm
from fair.inverse import inverse_fair_scm

TECH_1 = 'Zero-CO$_2$ fuels'
TECH_2 = 'No-emissions aircraft'

baseline_1 = 'Gold'
baseline_2 = 'Silver'
baseline_3 = 'Bronze'
baseline_4 = 'EWF'

#============================= MAKE SCENARIOS ===========================
# import data from Lee et al. (2021)
def import_lee():
    """
    Functions to import historical dataset from Lee et al. (2021)
    :return: ERF values from 1990-2018 for all aviation species,
             aviation values (e.g. fuel usage, km traveled for 1990-2018,
             emissions of all aviation species for 1990-2018,
             Sensitivity to emissions factors (constant)
    """
    leedf = pd.read_csv('Data/timeseries_lee2021.csv', sep=';') # import data from csv file
    aviation_2018 = leedf.iloc[2:31,1:12]
    aviation_2018.columns = leedf.iloc[0,1:12]
    aviation_2018.index = leedf.iloc[2:31,0]
    aviation_2018.index = pd.to_datetime(aviation_2018.index)
    emissions_2018 = leedf.iloc[2:31, 13:19]
    emissions_2018.columns = leedf.iloc[0, 13:19]
    emissions_2018.index = leedf.iloc[2:31, 0]
    emissions_2018.index = pd.to_datetime(aviation_2018.index)
    ERF_factors_2018 =  leedf.iloc[12, 20:29]
    ERF_factors_2018.index = leedf.iloc[0, 20:29]
    ERF_2018 = leedf.iloc[12:31,30:40] # isolate only ERF time series
    ERF_2018.columns = leedf.iloc[0, 30:40]  # set column names
    ERF_2018.index = leedf.iloc[12:31,0]
    ERF_2018.index = pd.to_datetime(ERF_2018.index)
    ERF_2018 = ERF_2018.apply(lambda x: x.str.replace(',', '.'))
    aviation_2018 = aviation_2018.apply(lambda x: x.str.replace(',', '.'))
    emissions_2018 = emissions_2018.apply(lambda x: x.str.replace(',', '.'))
    ERF_factors_2018 = ERF_factors_2018.apply(lambda x: x.replace(',', '.'))
    return ERF_2018, aviation_2018, emissions_2018, ERF_factors_2018

def make_aviation_CMIP(scenario):
    """
    Reads the .nc file with the emission pathways downloaded from https://esgf-node.llnl.gov/search/input4mips/.
    :param scenario: what scenario to read (e.g. SSP1_19, SSP2_45)
    :return: a data frame with the aviation emissions
    """
    aviation_emissions = xr.merge(
        [
            xr.open_dataset(x)
            .mean(dim=["level", "lat", "lon"])
            .groupby("time.year").mean("time")
            .drop(["lat_bnds", "lon_bnds"]) #, "time_bnds"])
            for x in glob.glob("Data/SSPs/"+scenario+"/*input4MIPs*.nc")
        ]
    )

    av_long = aviation_emissions.interp(year=np.arange(2015,2101), method = 'cubic').to_dataframe()
    av_long.index = pd.to_datetime(list(np.arange(2015,2101)), format='%Y')
    return av_long

def make_first_time_CMIP_aviation_scenarios():
    """
    Function to make .csv tables from .nc files of aviation emissions.
    :return: .csv tables with emissions
    """
    scenarios = ['SSP1_19','SSP1_26', 'SSP2_45', 'SSP3_70', 'SSP4_34', 'SSP4_60',
                 'SSP5_34', 'SSP5_85']
    [
        make_aviation_CMIP(scenario).to_csv('Data/inputCMIP_aviation_'+scenario+'.csv')
        for scenario in scenarios
    ]
    return

def make_aviation_Sharmina(scenario):
    """
    Function to calculate CO2 intensity changes and demand changes from the study by Sharmina et al. (2020) (https://doi.org/10.1080/14693062.2020.1831430)
    :param scenario: the scenario chosen
    :return: a dataframe with CO2 intensity and demand informations.
    """
    dates_av_CO2intensity = pd.to_datetime([2005, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100],
                                           format='%Y')
    if scenario == '1.5':
        av_CO2intensity_kg_pkm = [0.2050, 0.1990, 0.1761, 0.1375,0.1049,0.0750,0.0580,0.0531,0.0512,0.0499, 0.0490]
        av_demand_billion_pkm = [3955, 3696, 4425, 5377, 4915, 5070, 5791, 7026, 8557, 10155, 11765]
        av_energyintensity_GJ_pkm = [0.0055, 0.0056, 0.0048,	0.0032,	0.0026,	0.0023,	0.0019,	0.0016,	0.0014,	0.0013,	0.0012]
    elif scenario == '2':
        av_CO2intensity_kg_pkm = [0.2050,0.1990,0.1756,0.1244,0.0751,0.0453,0.0379,0.0432, 0.0455,0.0479,0.0469]
        av_demand_billion_pkm = [3955,3696,4437,6204,8383,10345,12331,13610,14149,13180,13322]
        av_energyintensity_GJ_pkm = [0.0055,0.0056,	0.0049,	0.0035,	0.0027,	0.0023,	0.0018,	0.0015,	0.0013,	0.0012,	0.0011]
    else:
        av_CO2intensity_kg_pkm = [0.2050, 0.1990, 0.1828, 0.1422, 0.0955, 0.0706, 0.0742, 0.0788, 0.0714, 0.0667, 0.0685]
        av_demand_billion_pkm = [3955, 3696, 4531, 6958, 10227, 13212, 16087, 19145, 22493, 26311, 29858]
        av_energyintensity_GJ_pkm = [0.0055,0.0056, 0.0051,	0.0042,	0.0035,	0.0031,	0.0027,	0.0024,	0.0022,	0.0020,	0.0019]
    av_sharmina = pd.DataFrame(data = zip(av_CO2intensity_kg_pkm, av_demand_billion_pkm), index=dates_av_CO2intensity,
                                columns=['CO2 intensity (kg/pkm)', 'demand (billion pkm)'])
    av_sharmina_long = av_sharmina.resample("AS").asfreq()
    av_sharmina_long = av_sharmina_long.astype(float).interpolate(method="cubic")
    av_sharmina_long['CO2 intensity change since 2005 (pkm)'] = (av_sharmina_long['CO2 intensity (kg/pkm)'].values
                                                            - av_sharmina_long['CO2 intensity (kg/pkm)']['2005'].values)\
                                                           /av_sharmina_long['CO2 intensity (kg/pkm)']['2005'].values
    av_sharmina_long['demand change since 2005'] = (av_sharmina_long['demand (billion pkm)'].values
                                                            - av_sharmina_long['demand (billion pkm)']['2005'].values)\
                                                    /av_sharmina_long['demand (billion pkm)']['2005'].values
    return av_sharmina_long

def make_SSP_CMIP_emissions(df, scenario):
    """
    Makes the input aviation emissions dataframe needed for futher analysis.
    :param df: historical emissions from Lee et al. 2021 (https://doi.org/10.1016/j.atmosenv.2020.117834)
    :param scenario: SSP-RCP scenario
    :return: dataframe with all emissions and distance
    """
    av_df = pd.read_csv('Data/inputCMIP_aviation_'+scenario+'.csv', index_col=0)
    av_df.index = pd.to_datetime(av_df.index, infer_datetime_format=True)
    if scenario == 'SSP1_19':
        av_SSP2 = make_aviation_Sharmina('1.5')
    elif scenario == 'SSP1_26':
        av_SSP2 = make_aviation_Sharmina('2')
    else:
        av_SSP2 = make_aviation_Sharmina('reference')
    fuel_per_CO2 = np.mean(pd.to_numeric(df['Fuel'])/pd.to_numeric(df['CO2']))
    kgCO2_km_hist = pd.to_numeric(df['CO2'])/(pd.to_numeric(df['distance'])) #in kgCO2/km flown
    correction_pkm = np.mean(kgCO2_km_hist['2005':'2018'].values/av_SSP2.loc['2005':'2018', 'CO2 intensity (kg/pkm)'].values)
    av_SSP2['CO2 intensity (kg/km)'] = correction_pkm* av_SSP2['CO2 intensity (kg/pkm)']
    av_SSP2['CO2 intensity change since 2005'] = (av_SSP2['CO2 intensity (kg/km)'].values
                                                            - kgCO2_km_hist['2005'].values)\
                                                           /kgCO2_km_hist['2005'].values
    CO2_toTgyr = pd.to_numeric(df['CO2']['2015'])/av_df['CO2_em_AIR_anthro']['2015']
    BC_toTgyr = pd.to_numeric(df['BC']['2015'])/av_df['BC_em_AIR_anthro']['2015']
    NOx_toTgyr = pd.to_numeric(df['NOx']['2015'] )/ av_df['NOx_em_AIR_anthro']['2015']
    SO2_toTgyr = pd.to_numeric(df['SO2']['2015'] )/ av_df['SO2_em_AIR_anthro']['2015']
    EI_H2O = 0.37 # TgH2O/TgCO2
    contrails_dist = 1.17  # conversion factor from km traveled to km of contrails and contrails-cirruses
    av_SSP_df = pd.DataFrame(index=pd.to_datetime(list(np.arange(1990,2101)), format = '%Y'), columns = ['CO2', 'BC', 'NOx', 'SO2', 'H2O', 'Fuel', 'Distance', 'Contrail'])
    av_SSP_df = av_SSP_df.fillna(0.)
    av_SSP_df['CO2'] = np.concatenate((pd.to_numeric(df.loc[:'2014','CO2']),av_df['CO2_em_AIR_anthro'].values * CO2_toTgyr.values))
    av_SSP_df['BC'] = np.concatenate((pd.to_numeric(df.loc[:'2014','BC']),av_df['BC_em_AIR_anthro'].values * BC_toTgyr.values))
    av_SSP_df['NOx'] = np.concatenate((pd.to_numeric(df.loc[:'2014','NOx']),av_df['NOx_em_AIR_anthro'].values * NOx_toTgyr.values))
    av_SSP_df['SO2'] = np.concatenate((pd.to_numeric(df.loc[:'2014','SO2']),av_df['SO2_em_AIR_anthro'].values * SO2_toTgyr.values))
    av_SSP_df['Fuel'] = np.concatenate((pd.to_numeric(df.loc[:'2014','Fuel']),av_SSP_df.loc['2015':,'CO2'].values*fuel_per_CO2))
    av_SSP_df['H2O']= av_SSP_df['CO2'].values * EI_H2O
    av_SSP_df['Distance'] = np.concatenate((pd.to_numeric(df.loc[:'2014','distance'])*10**9, #from billion km to km
                                            av_SSP_df.loc['2015':,'CO2'].values*10**9 #from TgCO2 to kgCO2
                                            /av_SSP2.loc['2015':,'CO2 intensity (kg/km)'].values)) # in kgCO2/km
    av_SSP_df['Contrail'] = av_SSP_df['Distance'].values*contrails_dist  # in km
    av_SSP2['Change in km since 2005'] = (av_SSP_df['Contrail']['2005':].values - av_SSP_df['Contrail']['2005'].values)/av_SSP_df['Contrail']['2005'].values
    return av_SSP_df, av_SSP2

def make_CO2aviation_hist():
    """
    Makes CO2 emissions from aviation from 1940-2018 from concentrations reported in Lee et al. 2021
    :return: historical aviation CO2 emissions and forcing due to CO2 emissions
    """
    CO2_C_1940_2018 = np.array([0.0042, 0.0078, 0.0113, 0.0149, 0.0187, 0.0227, 0.0269, 0.0314, 0.0362, 0.0413, 0.0468,
                          0.0527, 0.0590, 0.0658, 0.0731, 0.0810, 0.0894, 0.0986, 0.1085, 0.1192, 0.1308, 0.1437,
                          0.1579, 0.1724, 0.1870, 0.2024, 0.2193, 0.2409, 0.2657, 0.2907, 0.3143, 0.3386, 0.3647,
                          0.3916, 0.4162, 0.4404, 0.4643, 0.4908, 0.5185, 0.5475, 0.5762, 0.6038, 0.6319, 0.6598,
                          0.6898, 0.7213, 0.7558, 0.7924, 0.8315, 0.8725, 0.9130, 0.9507, 0.9872, 1.0216, 1.0596,
                          1.0997, 1.1434, 1.1892, 1.2361, 1.2853, 1.3382, 1.3869, 1.4357, 1.4843, 1.5381, 1.5956,
                          1.6530, 1.7123, 1.7704, 1.8227, 1.8803, 1.9401, 2.0004, 2.0633, 2.1291, 2.2002, 2.2737,
                          2.3496, 2.4281])
    CO2_C_1940_2018 += 278
    E1, F1, T1 = inverse_fair_scm(C=CO2_C_1940_2018, rt=0)
    return E1, F1


def calc_ERF_CO2(E, start_year=1990):
    """
    Calculate the ERF of CO2
    :param E: dataframe with future emissions
    :param start_year: start date of future emissions
    :return: forcing of CO2 emissions from start date
    """
    E_CO2_hist = make_CO2aviation_hist()[0]
    E_input = np.concatenate((E_CO2_hist[:start_year-1940], pd.to_numeric(E.loc[str(start_year):,'CO2']).values/(3.677*10**3)), axis = 0)
    C_CO2, F_CO2, T_CO2= fair_scm(
        emissions = E_input,
        useMultigas= False
    )
    return F_CO2[start_year-1940:]*10**3 #in mW/m2

def calculate_ERF(df, e_factors):
    """
    Function to calculate ERF from sensitivity to emissions reported in Lee et al. 2021
    :param df: dataframe with emissions
    :param e_factors: sensitivity to emissions reported in Lee et al. 2021
    :return: ERF of each species in each year
    """
    #f_CO2 = ufloat(0.035, 0.00057) # sensitivity to emissions for CO2
    index = df.index
    columns = e_factors.index
    # sensitivity to emissions for other species + uncertainties (as in Lee et al. 2021)
    erf_data = np.array([ufloat(34.44, 9.90), ufloat(-18.60,6.90), ufloat(-9.35,3.40), ufloat(-2.80,1.00), ufloat(5.46,8.10),
                     ufloat(100.67, 165.50), ufloat(-19.91,16.00), ufloat(0.0052, 0.0026), ufloat(9.36*10**(-10),6.57*10**(-10))])
    erf_factors = pd.DataFrame(index = columns, columns = ['ERF factors'],
                                 data = erf_data)
    ERF_df = pd.DataFrame(index=index, columns=columns)
    ERF_df = ERF_df.fillna(0.)
    #ERF_df['CO2'] = df['CO2'].values*f_CO2
    ERF_df['CO2'] = calc_ERF_CO2(df, start_year=df.index[0].year)
    ERF_df['O3 short'] = df['NOx'].values*erf_factors.loc['O3 short',:].values
    ERF_df['CH4'] = df['NOx'].values*erf_factors.loc['CH4',:].values
    ERF_df['O3 long'] = df['NOx'].values*erf_factors.loc['O3 long',:].values
    ERF_df['SWV'] = df['NOx'].values*erf_factors.loc['SWV',:].values
    ERF_df['netNOx'] = df['NOx'].values*erf_factors.loc['netNOx',:].values# here actually 6,4
    ERF_df['BC'] = df['BC'].values*erf_factors.loc['BC',:].values
    ERF_df['SO4'] = df['SO2'].values*erf_factors.loc['SO4',:].values
    ERF_df['H2O'] = df['H2O'].values*erf_factors.loc['H2O',:].values
    ERF_df['Contrails and C-C'] = df['Contrail'].values*erf_factors.loc['Contrails and C-C',:].values
    ERF_df['non-CO2'] = ERF_df.loc[:,['netNOx', 'BC', 'SO4', 'H2O', 'Contrails and C-C']].sum(axis=1)
    ERF_df['Tot'] = ERF_df.loc[:,['netNOx', 'BC', 'SO4', 'H2O', 'Contrails and C-C', 'CO2']].sum(axis=1)
    return ERF_df

#==================================== CALCULATE CDR ==============================
# MAKE baseline emissions
def make_gold_emissions(df, intermediate_goal = False, start_year=2050, intermediate_year = 2035):
    """
    Makes emissions time series corresponding to the Gold definition of climate neutrality
    (all emissions zero after start_year)
    :param df: aviation emissions
    :param start_year: start of climate neutrality
    :return: Gold baseline of emissions
    """
    av_gold_df = pd.DataFrame(index=pd.to_datetime(list(np.arange(1990,2101)), format = '%Y'), columns =  df.columns)
    av_gold_df = av_gold_df.fillna(0.)
    av_gold_df[:str(start_year - 1)] = df[:str(start_year - 1)]
    av_gold_df.loc['2020':str(start_year-1), 'CO2'] = float(df.loc[str(2019), 'CO2'])
    if intermediate_goal == True:
        for i in av_gold_df.columns:
            av_gold_df.loc[str(intermediate_year):str(start_year-1), 'CO2'] = np.linspace(float(av_gold_df.loc[str(intermediate_year), 'CO2']),
                                                                                          0.5*float(av_gold_df.loc[str(2005), 'CO2']),
                                                                                          start_year - intermediate_year)
    return av_gold_df

def make_bronze_emissions(df, intermediate_goal = False, start_year=2050, intermediate_year = 2035):
    """
    Makes emissions time series corresponding to the Bronze definition of climate neutrality
    (all non-CO2 emissions stable at their start_year levels, CO2 emissions zero after start_year)
    :param df: aviation emissions
    :param start_year: start of climate neutrality
    :return: Bronze baseline of emissions
    """
    av_bronze_df = df.copy()
    av_bronze_df[str(start_year):] = av_bronze_df.loc[str(start_year)]
    av_bronze_df.loc[str(start_year):, 'CO2'] = 0.
    av_bronze_df.loc['2020':str(start_year - 1), 'CO2'] = float(df.loc[str(2019), 'CO2'])
    if intermediate_goal == True:
        av_bronze_df.loc[str(intermediate_year):str(start_year-1), 'CO2'] = np.linspace(float(av_bronze_df.loc[str(intermediate_year), 'CO2']),
                                                                                        0.5*float(av_bronze_df.loc[str(2005), 'CO2']),
                                                                                        start_year - intermediate_year)
    return av_bronze_df

def make_target_scenarios(df1, df2, df3, erf_factors, start_year, baseline, intermediate_goal = False):
    """
    From emissions, calculate the emissions and ERF baseline corresponding to the definition of climate neutrality.
    :param df1: emissions in scenario 1
    :param df2: emissions in scenario 2
    :param df3: emissions in scenario 3
    :param erf_factors: ERF sensitivity to emissions from Lee et al. (2021)
    :param start_year: start of climate neutrality
    :param baseline: type of climate neutrality (Gold or Bronze)
    :return: baseline emissions and ERF for all three scenario under the corresponding climate neutrality definition
    """
    if start_year <= 2035:
        if baseline == baseline_1:
            baseline_1_df = make_gold_emissions(df1, intermediate_goal, start_year= 2050)
            baseline_2_df = make_gold_emissions(df2, intermediate_goal, start_year= 2050)
            baseline_3_df = make_gold_emissions(df3, intermediate_goal, start_year= 2050)
        elif baseline == baseline_3:
            baseline_1_df = make_bronze_emissions(df1, intermediate_goal, start_year=2050)
            baseline_2_df = make_bronze_emissions(df2, intermediate_goal, start_year=2050)
            baseline_3_df = make_bronze_emissions(df3, intermediate_goal, start_year=2050)
    else:
        if baseline == baseline_1:
            baseline_1_df = make_gold_emissions(df1, intermediate_goal, start_year)
            baseline_2_df = make_gold_emissions(df2, intermediate_goal, start_year)
            baseline_3_df = make_gold_emissions(df3, intermediate_goal, start_year)
        elif baseline == baseline_3:
            baseline_1_df = make_bronze_emissions(df1, intermediate_goal, start_year)
            baseline_2_df = make_bronze_emissions(df2, intermediate_goal, start_year)
            baseline_3_df = make_bronze_emissions(df3, intermediate_goal, start_year)
    ERF_1_df = calculate_ERF(baseline_1_df, erf_factors)
    ERF_2_df = calculate_ERF(baseline_2_df, erf_factors)
    ERF_3_df = calculate_ERF(baseline_3_df, erf_factors)
    return baseline_1_df, baseline_2_df, baseline_3_df, \
           ERF_1_df, ERF_2_df, ERF_3_df

#============================ GWP* ================================
def calculate_GWPstar_CDR(ERF, dt, start_date = 2018, init_date = 1990):
    """
    Equation to calculate GWP* as reported in Lee et al. 2021
    :param ERF: time series of ERF
    :param dt: delta t of choice
    :param start_date: start date of the policy (when CDR starts offsetting emissions)
    :return: time series of CDR rates
    """
    H = 100
    AGWP_CO2 = 8.8*10**(-2) # from Lee et al. 2021 (in mW yr / (m^2 Mt))
    CDR = []
    for i in np.arange((start_date - init_date), len(ERF)):
        if 0 < dt <= (start_date - init_date):
            tmp = ((ERF[i] - ERF[i - dt]) / dt) * H / AGWP_CO2
            #if tmp < 0:
            #    tmp = 0.
            CDR.append(tmp)
        else: # case needed when optimizing dt
            CDR.append(0.0)
    return np.array(CDR)

def make_CDR_metric(ERF_df, emissions_df, dt, start_date=2018, CO2only = False, metric ='GWP*', EF=2):
    """
    Equation to make dataframe with CDR rates offsetting each species of aviation
    :param ERF_df: dataframe with ERF values for each species
    :param emissions_df: dataframe with emissions for each species
    :param dt: delta t of choice
    :param start_date: start of policy (offsetting)
    :return: dataframe with CDR rates necessary to offset each emissions in each year
    """
    index = ERF_df.index
    columns = ERF_df.columns
    CDR_df = pd.DataFrame(index=index, columns=columns)
    CDR_df = CDR_df.fillna(0.)
    if metric == 'GWP100':
        GWP = pd.DataFrame(data = [
            0, # O3 short
            0, # CH4
            0, #O3 long
            0, # SWV
            114, # netNOx - Lee et al. 2021
            1166, #BC - Lee et al. 2021
            -226, # SO4 - Lee et al. 2021
            0.06 , #H20 - Lee et al. 2021
            11, #Contrails and C-C - Lee et al. 2021
           1, # CO2 # Lee et al. 2021
            0, # non-CO2
            0, # Tot
        ],
                           index = columns)
        CDR_df.loc[str(start_date):, 'netNOx'] = emissions_df.loc[str(start_date):, 'NOx']*GWP.loc['netNOx'].values
        CDR_df.loc[str(start_date):, 'BC'] = emissions_df.loc[str(start_date):, 'BC'] * GWP.loc['BC'].values
        CDR_df.loc[str(start_date):, 'SO4'] = emissions_df.loc[str(start_date):, 'SO2'] * GWP.loc['SO4'].values
        CDR_df.loc[str(start_date):, 'H2O'] = emissions_df.loc[str(start_date):, 'H2O'] * GWP.loc['H2O'].values
        CDR_df.loc[str(start_date):, 'Contrails and C-C'] = emissions_df.loc[str(start_date):, 'Contrail']/10**9 * GWP.loc['Contrails and C-C'].values
        CDR_df.loc[str(start_date):, 'CO2'] = emissions_df.loc[str(start_date):, 'CO2'] * GWP.loc['CO2'].values
        CDR_df.loc[str(start_date):, 'Tot'] = CDR_df.loc[:,['netNOx', 'BC', 'SO4', 'H2O', 'Contrails and C-C', 'CO2']].sum(axis=1)

    elif metric == 'EWF':
        for i in columns:
            if i == 'CO2':
                CDR_df.loc[str(start_date):, i] = emissions_df.loc[str(start_date):, i]
            elif i == 'Tot':
                CDR_df.loc[str(start_date):, i] = emissions_df.loc[str(start_date):, 'CO2']*EF
    else:
        for i in columns:
            if i == 'CO2':
                CDR_df.loc[str(start_date):, i] = emissions_df.loc[str(start_date):, i]
            else:
                if CO2only == False:
                    CDR_df.loc[str(start_date):,i] = calculate_GWPstar_CDR(ERF_df[i], dt, start_date)
        CDR_df.loc[str(start_date):, 'Tot'] = CDR_df.loc[:,['netNOx', 'BC', 'SO4', 'H2O', 'Contrails and C-C', 'CO2']].sum(axis=1)
    return CDR_df

def make_CDR_scenarios(ERF_df1, ERF_df2, ERF_df3, df1, df2, df3, dt=10, start_date = 2018, CO2only = False, metric ='GWP*', EF = None):
    """
    Make dataframe with CDR needed to offset each species for each scenario
    :param ERF_df1: dataframe of ERF for scenario 1
    :param ERF_df2: dataframe of ERF for scenario 2
    :param ERF_df3: dataframe of ERF for scenario 3
    :param df1: dataframe of emissions for scenario 1
    :param df2: dataframe of emissions for scenario 2
    :param df3: dataframe of emissions for scenario 3
    :param dt: delta t of choice
    :param start_date: start of the offsetting policy
    :return: three dataframe with CDR rates needed to offset each aviation species
    """
    CDRstar_1 = make_CDR_metric(ERF_df1, df1, dt, start_date, CO2only, metric, EF)
    CDRstar_2 = make_CDR_metric(ERF_df2, df2, dt, start_date, CO2only, metric, EF)
    CDRstar_3 = make_CDR_metric(ERF_df3, df3, dt, start_date, CO2only, metric, EF)
    return CDRstar_1, CDRstar_2, CDRstar_3

#=========================== USE FAIR ======================================
def test_CO2_Fair(E, CDR, ERF, E_ref = None, ERF_ref = None, start_year = 2018, end_year = 2100, ind_CDR_tot = 11, baseline ='zero'):
    """
    Calculate ERF and T of aviation by running Fair in CO2-only mode with non-CO2 forcings input externally.
    :param E: Emissions of which we want to know the ERF and T
    :param CDR: CDR rates to include in the simulation
    :param ERF: ERF time series (for calculation of the non-CO2 forcing)
    :param E_ref: baseline emissions according to climate neutrality definition
    :param ERF_ref: baseline ERF according to climate neutrality definition
    :param start_year: start of climate neutrality (CDR rates start being added)
    :param end_year: end year of the analysis
    :param ind_CDR_tot: index of total CDR in the CDR dataframe
    :param baseline: Gold, Silver or Bronze baselines
    :return: a Fair_outputs class object containing the climatic outcomes for aviation, aviation + CDR, and the baseline
    """

    class Fair_outputs:
        """
        Class to store Fair output nicely
        """

        def __init__(self, C_baseline, F_baseline, T_baseline, C_aviation, F_aviation, T_aviation,
                     C_avCDR, F_avCDR, T_avCDR, C_baseline_upper, F_baseline_upper, T_baseline_upper,
                     C_aviation_upper, F_aviation_upper, T_aviation_upper,
                     C_avCDR_upper, F_avCDR_upper, T_avCDR_upper, C_baseline_lower, F_baseline_lower, T_baseline_lower,
                     C_aviation_lower, F_aviation_lower, T_aviation_lower,
                     C_avCDR_lower, F_avCDR_lower, T_avCDR_lower):
            self.C_baseline = C_baseline
            self.F_baseline = F_baseline
            self.T_baseline = T_baseline
            self.C_aviation = C_aviation
            self.F_aviation = F_aviation
            self.T_aviation = T_aviation
            self.C_avCDR = C_avCDR
            self.F_avCDR = F_avCDR
            self.T_avCDR = T_avCDR
            self.C_baseline_upper = C_baseline_upper
            self.F_baseline_upper = F_baseline_upper
            self.T_baseline_upper = T_baseline_upper
            self.C_aviation_upper = C_aviation_upper
            self.F_aviation_upper = F_aviation_upper
            self.T_aviation_upper = T_aviation_upper
            self.C_avCDR_upper = C_avCDR_upper
            self.F_avCDR_upper = F_avCDR_upper
            self.T_avCDR_upper = T_avCDR_upper
            self.C_baseline_lower = C_baseline_lower
            self.F_baseline_lower = F_baseline_lower
            self.T_baseline_lower = T_baseline_lower
            self.C_aviation_lower = C_aviation_lower
            self.F_aviation_lower = F_aviation_lower
            self.T_aviation_lower = T_aviation_lower
            self.C_avCDR_lower = C_avCDR_lower
            self.F_avCDR_lower = F_avCDR_lower
            self.T_avCDR_lower = T_avCDR_lower
    index_start = start_year - 1940
    index_end = end_year - 1940
    # copy FaIR's RCP2.6
    E_CO2_hist = make_CO2aviation_hist()[0]
    E_CO2_aviation = np.concatenate((E_CO2_hist[:1990 - 1940], pd.to_numeric(E['CO2']).values/(3.677 * 10 ** 3)), axis=0)
    E_CO2_baseline = np.zeros(len(E_CO2_aviation))
    E_CO2_baseline[:index_start + 1] = E_CO2_aviation[:index_start + 1]
    E_CO2_avCDR = E_CO2_aviation.copy()
    E_CO2_avCDR_upper = E_CO2_aviation.copy()
    E_CO2_avCDR_lower = E_CO2_aviation.copy()
    E_CO2_avCDR[index_start:] -= unumpy.nominal_values(CDR.iloc[start_year-1990:,ind_CDR_tot]/(3.677*10**3))
    E_CO2_avCDR_upper[index_start:] -= (unumpy.nominal_values(CDR.iloc[start_year - 1990:, ind_CDR_tot]) +
                                        unumpy.std_devs(CDR.iloc[start_year - 1990:, ind_CDR_tot]))/ (3.677 * 10 ** 3)
    E_CO2_avCDR_lower[index_start:] -= (unumpy.nominal_values(CDR.iloc[start_year - 1990:, ind_CDR_tot]) -
                                        unumpy.std_devs(CDR.iloc[start_year - 1990:, ind_CDR_tot])) / (
                                                   3.677 * 10 ** 3)
    ERF_nonCO2 = np.zeros(len(E_CO2_aviation))
    ERF_nonCO2_upper = np.zeros(len(E_CO2_aviation))
    ERF_nonCO2_lower = np.zeros(len(E_CO2_aviation))
    ERF_nonCO2[1990-1940:] = unumpy.nominal_values(ERF.iloc[:,10] / 10 ** 3)
    ERF_nonCO2_upper[1990 - 1940:] = unumpy.nominal_values(ERF.iloc[:, 10] / 10 ** 3) + \
                                     unumpy.std_devs(ERF.iloc[:, 10] / 10 ** 3)
    ERF_nonCO2_lower[1990 - 1940:] = unumpy.nominal_values(ERF.iloc[:, 10] / 10 ** 3) - \
                                     unumpy.std_devs(ERF.iloc[:, 10] / 10 ** 3)
    ERF_nonCO2_baseline = ERF_nonCO2.copy()
    ERF_nonCO2_baseline_upper = ERF_nonCO2_upper.copy()
    ERF_nonCO2_baseline_lower = ERF_nonCO2_lower.copy()
    if baseline == 'zero':
        ERF_nonCO2_baseline[start_year - 1940:] = 0.
        ERF_nonCO2_baseline_upper[start_year - 1940:] = 0.
        ERF_nonCO2_baseline_lower[start_year - 1940:] = 0.
    elif baseline == 'SSP1_19' or baseline == baseline_1:
        E_CO2_baseline[E_ref.index[0].year - 1940:] = E_ref['CO2'].values / (3.677 * 10 ** 3)
        ERF_nonCO2_baseline[ERF_ref.index[0].year - 1940:] = unumpy.nominal_values(ERF_ref['non-CO2'] / 10 ** 3)
        ERF_nonCO2_baseline_upper[ERF_ref.index[0].year - 1940:] = unumpy.nominal_values(ERF_ref['non-CO2'] / 10 ** 3) +\
                                                                   unumpy.std_devs(ERF_ref['non-CO2'] / 10 ** 3)
        ERF_nonCO2_baseline_lower[ERF_ref.index[0].year - 1940:] = unumpy.nominal_values(ERF_ref['non-CO2'] / 10 ** 3) - \
                                                                   unumpy.std_devs(ERF_ref['non-CO2'] / 10 ** 3)
    else:
        ERF_nonCO2_baseline[start_year - 1940:] = ERF_nonCO2_baseline[start_year - 1940 - 1]
        ERF_nonCO2_baseline_upper[start_year - 1940:] = ERF_nonCO2_baseline_upper[start_year - 1940 - 1]
        ERF_nonCO2_baseline_lower[start_year - 1940:] = ERF_nonCO2_baseline_lower[start_year - 1940 - 1]
    # Running single simulations with best estimates
    C_baseline, F_baseline, T_baseline = fair_scm(
        emissions=E_CO2_baseline,
        useMultigas= False,
        other_rf= ERF_nonCO2_baseline
    )

    C_aviation, F_aviation, T_aviation = fair_scm(
        emissions=E_CO2_aviation,
        useMultigas=False,
        other_rf= ERF_nonCO2
    )
    C_avCDR, F_avCDR, T_avCDR = fair_scm(
        emissions=E_CO2_avCDR,
        useMultigas=False,
        other_rf= ERF_nonCO2
    )
    # Running single simulations with best estimates and upper confidence level
    C_baseline_upper, F_baseline_upper, T_baseline_upper = fair_scm(
        emissions=E_CO2_baseline,
        useMultigas= False,
        other_rf= ERF_nonCO2_baseline_upper
    )

    C_aviation_upper, F_aviation_upper, T_aviation_upper = fair_scm(
        emissions=E_CO2_aviation,
        useMultigas=False,
        other_rf= ERF_nonCO2_upper
    )
    C_avCDR_upper, F_avCDR_upper, T_avCDR_upper = fair_scm(
        emissions=E_CO2_avCDR,
        useMultigas=False,
        other_rf= ERF_nonCO2_upper
    )
    # Running single simulations with best estimates and lower confidence level
    C_baseline_lower, F_baseline_lower, T_baseline_lower = fair_scm(
        emissions=E_CO2_baseline,
        useMultigas= False,
        other_rf= ERF_nonCO2_baseline_lower
    )

    C_aviation_lower, F_aviation_lower, T_aviation_lower = fair_scm(
        emissions=E_CO2_aviation,
        useMultigas=False,
        other_rf= ERF_nonCO2_lower
    )
    C_avCDR_lower, F_avCDR_lower, T_avCDR_lower = fair_scm(
        emissions=E_CO2_avCDR,
        useMultigas=False,
        other_rf= ERF_nonCO2_lower
    )
    return Fair_outputs(C_baseline, F_baseline, T_baseline, C_aviation, F_aviation, T_aviation,
                     C_avCDR, F_avCDR, T_avCDR, C_baseline_upper, F_baseline_upper, T_baseline_upper,
                     C_aviation_upper, F_aviation_upper, T_aviation_upper,
                     C_avCDR_upper, F_avCDR_upper, T_avCDR_upper, C_baseline_lower, F_baseline_lower, T_baseline_lower,
                     C_aviation_lower, F_aviation_lower, T_aviation_lower,
                     C_avCDR_lower, F_avCDR_lower, T_avCDR_lower)


#=================================== MAKE NEW TECHS SCENARIOS ===========================================
def calculate_scaled_SAF(old, new, old_err, new_err, perc):
    """
    Function to calculate reduction in emissions through SAF for a blending by 100% when data are available only for less than 100% blendings
    :param old: old emissions per kg (with JetA1 fuels)
    :param new: new emissions per kg (with Zero-CO$_2$ fuels)
    :param old_err: old emissions std  (with Jet A1 fuels)
    :param new_err: old emissions std  (with Zero-CO$_2$ fuels)
    :param perc: percent of blending of SAF
    :return: percent reduction in emissions with Zero-CO$_2$ fuels
    """
    y = 1 - ((old-new)/old)/perc
    y_err = 1- ((old+old_err)-(new+new_err))/(old+old_err)/perc
    err = y_err - y
    if y < 0:
        y = 0
    return ufloat(y, np.absolute(err))

def f_shape(t, end_date, start_date,a = 0.15, b = 1000):
    """
    returns s-shaped curve
    :param t: time (in year)
    :param end_date: end year of time series
    :param start_date: start year of time series
    :param a: coefficient 1
    :param b: coefficient 2
    :return: y corresponding to a s-shaped function of time
    """
    return b / (1 + np.exp(-a*(t-(start_date+(end_date - start_date)/2))))

def lin_function(x,a,b):
    """
    linear function of time
    :param x: time
    :param a: coefficient 1
    :param b: coefficient 2
    :return: linear function of time
    """
    return a*x + b



def make_emissions_new_tech(E, start_date, shape_uptake, tech_type, transition_date=2050):
    """
    make emissions array with new technology
    :param E: reference emissions (with Jet-A1 fuel)
    :param start_date: start date of technology substitution
    :param shape_uptake: type of uptake curve
    :param tech_type: type of technology, i.e. Zero-CO$_2$ fuels or E-airplanes
    :return: E_techs class-type array with emissions
    """
    class E_techs:
        """
        Class to store Fair output nicely
        """

        def __init__(self, E_tech, c):
            self.best = pd.DataFrame(index=E_tech.index, columns = E_tech.columns, data = unumpy.nominal_values(E_tech))
            self.upper = pd.DataFrame(index=E_tech.index, columns = E_tech.columns,
                                      data = unumpy.nominal_values(E_tech) + unumpy.std_devs(E_tech))
            self.lower = pd.DataFrame(index=E_tech.index, columns = E_tech.columns,
                                      data = unumpy.nominal_values(E_tech) + unumpy.std_devs(E_tech))
            self.uncertainty = E_tech
            self.uptake = c

    if tech_type == 'Zero-CO$_2$ fuels':
        alpha_CO2 = ufloat(0., 0.) # assuming 100% CO2-free Zero-CO$_2$ fuels, even in life cycle (e.g. thanks to 100% renewable energy)
        alpha_BC =  calculate_scaled_SAF(18.8, 11.4, 0.025, 0.025, 0.41) # calculated from change in aromatics reported in Voigt et al. (2021), scaled to a 100% FT SAF # FINAL: 3.9+-6.7%
        alpha_SO2 = calculate_scaled_SAF(0.117, 0.057, 0.003, 0.002, 0.41) # calculated from sulfate content of Voigt et al. (2021) scaled to a 100% FT SAF # FINAL: -25+-11%
        alpha_contrails = ufloat(0.35, 0.15) # scaled from Voigt et al. 2021, Kärcher 2018, and Burkhardt et al. 2018
        #calculate_scaled_SAF(4.2*10**15, 2.0*10**15, 0.6*10**15, 0.2*10**15, 1)
        alpha_NOx = ufloat(0.9, 0.08) # reported by Jagtap, 2019, Braun-Unkhoff et al., 2017, Blakey et al., 2011
        alpha_H2O = calculate_scaled_SAF(13.67, 14.36, 0.14, 0.02, 0.41) # calculated from hydrogen content of Voigt et al. (2021) scaled to a 100% FT SAF # FINAL: 18+-68%
        alpha_fuel = ufloat(1.0, 0.0)
        beta = 0.25
    elif tech_type == 'H2':
        alpha_CO2 = ufloat(0., 0.)
        alpha_BC = ufloat(0., 0.)
        alpha_SO2 = ufloat(0., 0.)
        alpha_contrails = ufloat(0.4, 0.3)
        alpha_NOx = ufloat(1.1, 0.2)
        alpha_H2O = ufloat(0.7, 0.3)
        alpha_fuel = ufloat(1.0, 0.0)
        beta = 0.2
    else:
        alpha_CO2 = ufloat(0., 0.)
        alpha_BC = ufloat(0.0, 0.)
        alpha_SO2 = ufloat(0.0, 0.)
        alpha_contrails = ufloat(0.0, 0.0)
        alpha_NOx = ufloat(0.0, 0.0)
        alpha_H2O = ufloat(0.0, 0.0)
        alpha_fuel = ufloat(0.0, 0.0)
        beta = 0.2

    c = np.zeros(len(E['CO2']))
    if shape_uptake == 'linear':
        c[start_date - E.index.year[0]:] = lin_function(np.arange(0, E.index.year[-1] + 1 - start_date), 0.02, 0)
    elif shape_uptake == 'sigma':
        c[start_date - E.index.year[0]:] = f_shape(np.arange(0, E.index.year[-1] + 1 - start_date), 50, 0, 0.15, 1)
    elif shape_uptake == 'abrupt':
        c[start_date - E.index.year[0]:transition_date - E.index.year[0]] = f_shape(
            np.arange(0, transition_date - start_date),
            transition_date - start_date, 0, beta, 1)
        c[transition_date - E.index.year[0]:] = 1.

    E_tech = E.copy()
    E_tech['CO2'] = (1-c)*E_tech['CO2']+c*alpha_CO2*E_tech['CO2']
    E_tech['BC'] = (1 - c) * E_tech['BC'] + c * alpha_BC * E_tech['BC']
    E_tech['NOx'] = (1 - c) * E_tech['NOx'] + c * alpha_NOx * E_tech['NOx']
    E_tech['SO2'] = (1 - c) * E_tech['SO2'] + c * alpha_SO2 * E_tech['SO2']
    E_tech['H2O'] = (1 - c) * E_tech['H2O'] + c * alpha_H2O * E_tech['H2O']
    E_tech['Contrail'] = (1 - c) * E_tech['Contrail'] + c * alpha_contrails * E_tech['Contrail']
    E_tech['Fuel'] = (1 - c) * E_tech['Fuel'] + c * alpha_fuel * E_tech['Fuel']

    return E_techs(E_tech, c)


def calculate_ERF_techs(df, e_factors):
    """
    Function to calculate ERF from sensitivity to emissions reported in Lee et al. 2021
    :param df: dataframe with emissions
    :param e_factors: sensitivity to emissions reported in Lee et al. 2021
    :return: ERF of each species in each year
    """
    #f_CO2 = ufloat(0.035, 0.00057) # sensitivity to emissions for CO2
    df_uncertain = df.uncertainty
    df_best = df.best
    index = df.best.index
    columns = e_factors.index
    # sensitivity to emissions for other species + uncertainties (as in Lee et al. 2021)
    erf_data = np.array([ufloat(34.44, 9.90), ufloat(-18.60,6.90), ufloat(-9.35,3.40), ufloat(-2.80,1.00), ufloat(5.46,8.10),
                     ufloat(100.67, 165.50), ufloat(-19.91,16.00), ufloat(0.0052, 0.0026), ufloat(9.36*10**(-10),6.57*10**(-10))])
    erf_factors = pd.DataFrame(index = columns, columns = ['ERF factors'],
                                 data = erf_data)
    ERF_df = pd.DataFrame(index=index, columns=columns)
    ERF_df = ERF_df.fillna(0.)
    ERF_df['CO2'] = calc_ERF_CO2(df_best, start_year=df_best.index[0].year)
    ERF_df['O3 short'] = df_uncertain['NOx']*erf_factors.loc['O3 short',:].values
    ERF_df['CH4'] = df_uncertain['NOx']*erf_factors.loc['CH4',:].values
    ERF_df['O3 long'] = df_uncertain['NOx']*erf_factors.loc['O3 long',:].values
    ERF_df['SWV'] = df_uncertain['NOx']*erf_factors.loc['SWV',:].values
    ERF_df['netNOx'] = df_uncertain['NOx']*erf_factors.loc['netNOx',:].values# here actually 6,4
    ERF_df['BC'] = df_uncertain['BC']*erf_factors.loc['BC',:].values
    ERF_df['SO4'] = df_uncertain['SO2']*erf_factors.loc['SO4',:].values
    ERF_df['H2O'] = df_uncertain['H2O']*erf_factors.loc['H2O',:].values
    ERF_df['Contrails and C-C'] = df_uncertain['Contrail']*erf_factors.loc['Contrails and C-C',:].values
    ERF_df['non-CO2'] = ERF_df.loc[:,['netNOx', 'BC', 'SO4', 'H2O', 'Contrails and C-C']].sum(axis=1)
    ERF_df['Tot'] = ERF_df.loc[:,['netNOx', 'BC', 'SO4', 'H2O', 'Contrails and C-C', 'CO2']].sum(axis=1)
    return ERF_df

#============================== CALCULATE COSTS ======================================
def calculate_cost_perkm(aviation_df, CDR_df, cost_CDR = 150):
    """
    Calculates the cost of CDR per km flown
    :param aviation_df: emissions dataframe
    :param CDR_df: CDR dataframe
    :param cost_CDR: cost per tCO2 removed (in US$) - default: 150
    :return: amount of CDR deployed per km flown, cost of CDR per km flown
    """
    distance = aviation_df['Distance'] #in km
    CDR_tonne = CDR_df['Tot']*10**6 # in tonne CO2 removed
    CDR_per_km = CDR_tonne/distance # in tonne CO2 removed per km
    cost_per_km = pd.DataFrame(data = CDR_per_km*cost_CDR, columns = ['Tot']) # in US$/km
    cost_per_km[cost_per_km < 0] = ufloat(0.0000000001, 0.0)
    return CDR_per_km, cost_per_km

def calculate_cost_passenger(summary_df, km, passengers):
    """
    Calculate cost of CDR per passenger per flight
    :param summary_df: summary dataframe containing the cost per km flown
    :param km: kilometers flown in the flight
    :param passengers: number of passenger per flight
    :return: cost of CDR per passenger per flight
    """
    new_summary = summary_df.copy()
    new_summary['summary_Tot_costkm'] = summary_df[
                                           'summary_Tot_costkm'] * km / passengers  # tot extra cost per passenger
    new_summary['summary_Tot_costkm_std'] = summary_df[
                                               'summary_Tot_costkm_std'] * km / passengers  # tot extra cost per passenger
    return new_summary


#============================== ANALYSE WITH OTHER METRICS =====================================

def make_fair_scenarios(E1, E2, E3, ERF1=None, ERF2=None, ERF3=None,
                        E_CO2eq_1 = None, E_CO2eq_2 = None, E_CO2eq_3 = None, start_year = 1990, what = None):
    """
    runs fair with the different scenarios
    :param E1: emissions of scenario 1
    :param E2: emissions of scenario 2
    :param E3: emissions of scenario 3
    :param ERF1: ERF of scenario 1
    :param ERF2: ERF of scenario 2
    :param ERF3: ERF of scenario 3
    :param E_CO2eq_1: emissions in CO2eq of scenario 1
    :param E_CO2eq_2: emissions in CO2eq of scenario 2
    :param E_CO2eq_3: emissions in CO2eq of scenario 3
    :param start_year: start of analysis
    :param what: type of metric used (only used for GWP*)
    :return:
    """
    fair1 = run_Fair_metrics(E1, ERF1, E_CO2eq_1, start_year, what)
    fair2 = run_Fair_metrics(E2, ERF2, E_CO2eq_2, start_year, what)
    fair3 = run_Fair_metrics(E3, ERF3, E_CO2eq_3, start_year, what)
    return fair1, fair2, fair3


def run_Fair_metrics(E, ERF=None, E_CO2eq = None, start_year = 1990, what = None):
    """
    runs Fair with emissions calculated with different metrics
    :param E: emissions
    :param ERF: effective radiative forcing
    :param E_CO2eq: CO2-equivalent emissions
    :param start_year: start of analysis
    :param what: type of metric (only for GWP* needed to be specified)
    :return: Fair_outputs type array with forcing and temperatures
    """
    class Fair_outputs:
        """
        Class to store Fair output nicely
        """

        def __init__(self, C_aviation, F_aviation, T_aviation,
                    C_aviation_upper, F_aviation_upper, T_aviation_upper,
                    C_aviation_lower, F_aviation_lower, T_aviation_lower,
                     ):
            self.C_aviation = C_aviation
            self.F_aviation = F_aviation
            self.T_aviation = T_aviation
            self.C_aviation_upper = C_aviation_upper
            self.F_aviation_upper = F_aviation_upper
            self.T_aviation_upper = T_aviation_upper
            self.C_aviation_lower = C_aviation_lower
            self.F_aviation_lower = F_aviation_lower
            self.T_aviation_lower = T_aviation_lower
    index_start = start_year - 1940
    E_CO2_hist = make_CO2aviation_hist()[0]
    E_CO2_aviation = np.concatenate((E_CO2_hist[:index_start], pd.to_numeric(E['CO2']).values/(3.677 * 10 ** 3)), axis=0)
    E_CO2_aviation_upper = E_CO2_aviation.copy()
    E_CO2_aviation_lower = E_CO2_aviation.copy()
    ERF_nonCO2 = np.zeros(len(E_CO2_aviation))
    ERF_nonCO2_upper = np.zeros(len(E_CO2_aviation))
    ERF_nonCO2_lower = np.zeros(len(E_CO2_aviation))
    if ERF is not None:
        ERF_nonCO2[index_start:] = unumpy.nominal_values(ERF.iloc[:,10] / 10 ** 3)
        ERF_nonCO2_upper[index_start:] = unumpy.nominal_values(ERF.iloc[:, 10] / 10 ** 3) + \
                                         unumpy.std_devs(ERF.iloc[:, 10] / 10 ** 3)
        ERF_nonCO2_lower[index_start:] = unumpy.nominal_values(ERF.iloc[:, 10] / 10 ** 3) - \
                                         unumpy.std_devs(ERF.iloc[:, 10] / 10 ** 3)
    if E_CO2eq is not None:
        if what == 'GWP*':
            E_CO2_aviation[index_start:] += unumpy.nominal_values(E_CO2eq['Tot'] - E_CO2eq['CO2']) / (3.677 * 10 ** 3)
            E_CO2_aviation_upper[index_start:] += unumpy.nominal_values(E_CO2eq['Tot'] - E_CO2eq['CO2']) / (3.677 * 10 ** 3) + \
                                                  unumpy.std_devs(E_CO2eq['Tot'] - E_CO2eq['CO2']) / (3.677 * 10 ** 3)
            E_CO2_aviation_upper[index_start:] += (unumpy.nominal_values(E_CO2eq['Tot'] - E_CO2eq['CO2']) / (3.677 * 10 ** 3)  - \
                                                   unumpy.std_devs(E_CO2eq['Tot'] - E_CO2eq['CO2']) / (3.677 * 10 ** 3))
        else:
            E_CO2_aviation[index_start:] += (E_CO2eq['Tot'] - E_CO2eq['CO2']).values/(3.677*10**3)

    C_aviation, F_aviation, T_aviation = fair_scm(
        emissions=E_CO2_aviation,
        useMultigas=False,
        other_rf= ERF_nonCO2
    )
    C_aviation_upper, F_aviation_upper, T_aviation_upper = fair_scm(
        emissions=E_CO2_aviation_upper,
        useMultigas=False,
        other_rf= ERF_nonCO2_upper
    )
    C_aviation_lower, F_aviation_lower, T_aviation_lower = fair_scm(
        emissions=E_CO2_aviation_lower,
        useMultigas=False,
        other_rf= ERF_nonCO2_lower
    )

    return Fair_outputs(C_aviation, F_aviation, T_aviation,
                     C_aviation_upper, F_aviation_upper, T_aviation_upper,
                     C_aviation_lower, F_aviation_lower, T_aviation_lower)




#================================== MAKE SUMMARY FOR INTER-SCENARIOS COMPARATIVE PLOTS ====================
def make_bar_CDR_alltechs_summary(CDR_1_gold, CDR_2_gold, CDR_3_gold,
                         CDR_1_silver, CDR_2_silver, CDR_3_silver,
                         CDR_1_bronze, CDR_2_bronze, CDR_3_bronze,
                         CDR_1_tech1_gold, CDR_2_tech1_gold, CDR_3_tech1_gold,
                         CDR_1_tech1_silver, CDR_2_tech1_silver, CDR_3_tech1_silver,
                         CDR_1_tech1_bronze, CDR_2_tech1_bronze, CDR_3_tech1_bronze,
                         CDR_1_tech2_gold, CDR_2_tech2_gold, CDR_3_tech2_gold,
                         CDR_1_tech2_silver, CDR_2_tech2_silver, CDR_3_tech2_silver,
                         CDR_1_tech2_bronze, CDR_2_tech2_bronze, CDR_3_tech2_bronze,
                         scenario1 = 'BAU',scenario2 = 'Air Pollution',scenario3 = 'Mitigation',
                                  sign = 'neutral',
                         tech1 = TECH_1, tech2 = TECH_2, what = 'mean', date1 = '2030', date2= '2050', date3= '2100'):
    """
    make summary dataframe with either mean rates, rates in given years, cumulative, or cost of CDR
    :param sign: just when calculating cumulative CDR, specify in case you want to calculate positive and negative contribution separately
    :param tech1: default: Zero-CO$_2$ fuels
    :param tech2: default: E-airplanes
    :param what: type of summary value needed
    :param date1: in case of "rates", first date in which we output the CDR rate
    :param date2: second date in which we output the CDR rate
    :param date3: third date in which we output the CDR rate
    :return: dataframe with summary values for CDR
    """
    if what == 'cumulative' or what == 'cumulative_E':
        if sign == 'neutral':
            summaryCDR_1_gold = np.sum(unumpy.nominal_values(CDR_1_gold), axis=0)
            summaryCDR_2_gold = np.sum(unumpy.nominal_values(CDR_2_gold), axis=0)
            summaryCDR_3_gold = np.sum(unumpy.nominal_values(CDR_3_gold), axis=0)
            summaryCDR_1_silver = np.sum(unumpy.nominal_values(CDR_1_silver), axis=0)
            summaryCDR_2_silver = np.sum(unumpy.nominal_values(CDR_2_silver), axis=0)
            summaryCDR_3_silver = np.sum(unumpy.nominal_values(CDR_3_silver), axis=0)
            summaryCDR_1_bronze = np.sum(unumpy.nominal_values(CDR_1_bronze), axis=0)
            summaryCDR_2_bronze = np.sum(unumpy.nominal_values(CDR_2_bronze), axis=0)
            summaryCDR_3_bronze = np.sum(unumpy.nominal_values(CDR_3_bronze), axis=0)
            summaryCDR_std_1_gold = np.sum(unumpy.std_devs(CDR_1_gold), axis=0)
            summaryCDR_std_2_gold = np.sum(unumpy.std_devs(CDR_2_gold), axis=0)
            summaryCDR_std_3_gold = np.sum(unumpy.std_devs(CDR_3_gold), axis=0)
            summaryCDR_std_1_silver = np.sum(unumpy.std_devs(CDR_1_silver), axis=0)
            summaryCDR_std_2_silver = np.sum(unumpy.std_devs(CDR_2_silver), axis=0)
            summaryCDR_std_3_silver = np.sum(unumpy.std_devs(CDR_3_silver), axis=0)
            summaryCDR_std_1_bronze = np.sum(unumpy.std_devs(CDR_1_bronze), axis=0)
            summaryCDR_std_2_bronze = np.sum(unumpy.std_devs(CDR_2_bronze), axis=0)
            summaryCDR_std_3_bronze = np.sum(unumpy.std_devs(CDR_3_bronze), axis=0)
            summaryCDR_1_tech1_gold = np.sum(unumpy.nominal_values(CDR_1_tech1_gold), axis=0)
            summaryCDR_2_tech1_gold = np.sum(unumpy.nominal_values(CDR_2_tech1_gold), axis=0)
            summaryCDR_3_tech1_gold = np.sum(unumpy.nominal_values(CDR_3_tech1_gold), axis=0)
            summaryCDR_1_tech1_silver = np.sum(unumpy.nominal_values(CDR_1_tech1_silver), axis=0)
            summaryCDR_2_tech1_silver = np.sum(unumpy.nominal_values(CDR_2_tech1_silver), axis=0)
            summaryCDR_3_tech1_silver = np.sum(unumpy.nominal_values(CDR_3_tech1_silver), axis=0)
            summaryCDR_1_tech1_bronze = np.sum(unumpy.nominal_values(CDR_1_tech1_bronze), axis=0)
            summaryCDR_2_tech1_bronze = np.sum(unumpy.nominal_values(CDR_2_tech1_bronze), axis=0)
            summaryCDR_3_tech1_bronze = np.sum(unumpy.nominal_values(CDR_3_tech1_bronze), axis=0)
            summaryCDR_std_1_tech1_gold = np.sum(unumpy.std_devs(CDR_1_tech1_gold), axis=0)
            summaryCDR_std_2_tech1_gold = np.sum(unumpy.std_devs(CDR_2_tech1_gold), axis=0)
            summaryCDR_std_3_tech1_gold = np.sum(unumpy.std_devs(CDR_3_tech1_gold), axis=0)
            summaryCDR_std_1_tech1_silver = np.sum(unumpy.std_devs(CDR_1_tech1_silver), axis=0)
            summaryCDR_std_2_tech1_silver = np.sum(unumpy.std_devs(CDR_2_tech1_silver), axis=0)
            summaryCDR_std_3_tech1_silver = np.sum(unumpy.std_devs(CDR_3_tech1_silver), axis=0)
            summaryCDR_std_1_tech1_bronze = np.sum(unumpy.std_devs(CDR_1_tech1_bronze), axis=0)
            summaryCDR_std_2_tech1_bronze = np.sum(unumpy.std_devs(CDR_2_tech1_bronze), axis=0)
            summaryCDR_std_3_tech1_bronze = np.sum(unumpy.std_devs(CDR_3_tech1_bronze), axis=0)
            summaryCDR_1_tech2_gold = np.sum(unumpy.nominal_values(CDR_1_tech2_gold), axis=0)
            summaryCDR_2_tech2_gold = np.sum(unumpy.nominal_values(CDR_2_tech2_gold), axis=0)
            summaryCDR_3_tech2_gold = np.sum(unumpy.nominal_values(CDR_3_tech2_gold), axis=0)
            summaryCDR_1_tech2_silver = np.sum(unumpy.nominal_values(CDR_1_tech2_silver), axis=0)
            summaryCDR_2_tech2_silver = np.sum(unumpy.nominal_values(CDR_2_tech2_silver), axis=0)
            summaryCDR_3_tech2_silver = np.sum(unumpy.nominal_values(CDR_3_tech2_silver), axis=0)
            summaryCDR_1_tech2_bronze = np.sum(unumpy.nominal_values(CDR_1_tech2_bronze), axis=0)
            summaryCDR_2_tech2_bronze = np.sum(unumpy.nominal_values(CDR_2_tech2_bronze), axis=0)
            summaryCDR_3_tech2_bronze = np.sum(unumpy.nominal_values(CDR_3_tech2_bronze), axis=0)
            summaryCDR_std_1_tech2_gold = np.sum(unumpy.std_devs(CDR_1_tech2_gold), axis=0)
            summaryCDR_std_2_tech2_gold = np.sum(unumpy.std_devs(CDR_2_tech2_gold), axis=0)
            summaryCDR_std_3_tech2_gold = np.sum(unumpy.std_devs(CDR_3_tech2_gold), axis=0)
            summaryCDR_std_1_tech2_silver = np.sum(unumpy.std_devs(CDR_1_tech2_silver), axis=0)
            summaryCDR_std_2_tech2_silver = np.sum(unumpy.std_devs(CDR_2_tech2_silver), axis=0)
            summaryCDR_std_3_tech2_silver = np.sum(unumpy.std_devs(CDR_3_tech2_silver), axis=0)
            summaryCDR_std_1_tech2_bronze = np.sum(unumpy.std_devs(CDR_1_tech2_bronze), axis=0)
            summaryCDR_std_2_tech2_bronze = np.sum(unumpy.std_devs(CDR_2_tech2_bronze), axis=0)
            summaryCDR_std_3_tech2_bronze = np.sum(unumpy.std_devs(CDR_3_tech2_bronze), axis=0)
        elif sign == 'positive':
            summaryCDR_1_gold = np.sum(unumpy.nominal_values(CDR_1_gold[CDR_1_gold['Tot'] > 0]), axis=0)
            summaryCDR_2_gold = np.sum(unumpy.nominal_values(CDR_2_gold[CDR_2_gold['Tot'] > 0]), axis=0)
            summaryCDR_3_gold = np.sum(unumpy.nominal_values(CDR_3_gold[CDR_3_gold['Tot'] > 0]), axis=0)
            summaryCDR_std_1_gold = np.sum(unumpy.std_devs(CDR_1_gold[CDR_1_gold['Tot'] > 0]), axis=0)
            summaryCDR_std_2_gold = np.sum(unumpy.std_devs(CDR_2_gold[CDR_2_gold['Tot'] > 0]), axis=0)
            summaryCDR_std_3_gold = np.sum(unumpy.std_devs(CDR_3_gold[CDR_3_gold['Tot'] > 0]), axis=0)
            summaryCDR_1_tech1_gold = np.sum(unumpy.nominal_values(CDR_1_tech1_gold[CDR_1_tech1_gold['Tot'] > 0]),
                                              axis=0)
            summaryCDR_2_tech1_gold = np.sum(unumpy.nominal_values(CDR_2_tech1_gold[CDR_2_tech1_gold['Tot'] > 0]),
                                              axis=0)
            summaryCDR_3_tech1_gold = np.sum(unumpy.nominal_values(CDR_3_tech1_gold[CDR_3_tech1_gold['Tot'] > 0]),
                                              axis=0)
            summaryCDR_std_1_tech1_gold = np.sum(unumpy.std_devs(CDR_1_tech1_gold[CDR_1_tech1_gold['Tot'] > 0]),
                                                  axis=0)
            summaryCDR_std_2_tech1_gold = np.sum(unumpy.std_devs(CDR_2_tech1_gold[CDR_2_tech1_gold['Tot'] > 0]),
                                                  axis=0)
            summaryCDR_std_3_tech1_gold = np.sum(unumpy.std_devs(CDR_3_tech1_gold[CDR_3_tech1_gold['Tot'] > 0]),
                                                  axis=0)
            summaryCDR_1_tech2_gold = np.sum(unumpy.nominal_values(CDR_1_tech2_gold[CDR_1_tech2_gold['Tot'] > 0]),
                                              axis=0)
            summaryCDR_2_tech2_gold = np.sum(unumpy.nominal_values(CDR_2_tech2_gold[CDR_2_tech2_gold['Tot'] > 0]),
                                              axis=0)
            summaryCDR_3_tech2_gold = np.sum(unumpy.nominal_values(CDR_3_tech2_gold[CDR_3_tech2_gold['Tot'] > 0]),
                                              axis=0)
            summaryCDR_std_1_tech2_gold = np.sum(unumpy.std_devs(CDR_1_tech2_gold[CDR_1_tech2_gold['Tot'] > 0]),
                                                  axis=0)
            summaryCDR_std_2_tech2_gold = np.sum(unumpy.std_devs(CDR_2_tech2_gold[CDR_2_tech2_gold['Tot'] > 0]),
                                                  axis=0)
            summaryCDR_std_3_tech2_gold = np.sum(unumpy.std_devs(CDR_3_tech2_gold[CDR_3_tech2_gold['Tot'] > 0]),
                                                  axis=0)
            summaryCDR_1_silver = np.sum(unumpy.nominal_values(CDR_1_silver[CDR_1_silver['Tot'] > 0]), axis=0)
            summaryCDR_2_silver = np.sum(unumpy.nominal_values(CDR_2_silver[CDR_2_silver['Tot'] > 0]), axis=0)
            summaryCDR_3_silver = np.sum(unumpy.nominal_values(CDR_3_silver[CDR_3_silver['Tot'] > 0]), axis=0)
            summaryCDR_std_1_silver = np.sum(unumpy.std_devs(CDR_1_silver[CDR_1_silver['Tot'] > 0]), axis=0)
            summaryCDR_std_2_silver = np.sum(unumpy.std_devs(CDR_2_silver[CDR_2_silver['Tot'] > 0]), axis=0)
            summaryCDR_std_3_silver = np.sum(unumpy.std_devs(CDR_3_silver[CDR_3_silver['Tot'] > 0]), axis=0)
            summaryCDR_1_tech1_silver = np.sum(
                unumpy.nominal_values(CDR_1_tech1_silver[CDR_1_tech1_silver['Tot'] > 0]), axis=0)
            summaryCDR_2_tech1_silver = np.sum(
                unumpy.nominal_values(CDR_2_tech1_silver[CDR_2_tech1_silver['Tot'] > 0]), axis=0)
            summaryCDR_3_tech1_silver = np.sum(
                unumpy.nominal_values(CDR_3_tech1_silver[CDR_3_tech1_silver['Tot'] > 0]), axis=0)
            summaryCDR_std_1_tech1_silver = np.sum(unumpy.std_devs(CDR_1_tech1_silver[CDR_1_tech1_silver['Tot'] > 0]),
                                                    axis=0)
            summaryCDR_std_2_tech1_silver = np.sum(unumpy.std_devs(CDR_2_tech1_silver[CDR_2_tech1_silver['Tot'] > 0]),
                                                    axis=0)
            summaryCDR_std_3_tech1_silver = np.sum(unumpy.std_devs(CDR_3_tech1_silver[CDR_3_tech1_silver['Tot'] > 0]),
                                                    axis=0)
            summaryCDR_1_tech2_silver = np.sum(
                unumpy.nominal_values(CDR_1_tech2_silver[CDR_1_tech2_silver['Tot'] > 0]), axis=0)
            summaryCDR_2_tech2_silver = np.sum(
                unumpy.nominal_values(CDR_2_tech2_silver[CDR_2_tech2_silver['Tot'] > 0]), axis=0)
            summaryCDR_3_tech2_silver = np.sum(
                unumpy.nominal_values(CDR_3_tech2_silver[CDR_3_tech2_silver['Tot'] > 0]), axis=0)
            summaryCDR_std_1_tech2_silver = np.sum(unumpy.std_devs(CDR_1_tech2_silver[CDR_1_tech2_silver['Tot'] > 0]),
                                                    axis=0)
            summaryCDR_std_2_tech2_silver = np.sum(unumpy.std_devs(CDR_2_tech2_silver[CDR_2_tech2_silver['Tot'] > 0]),
                                                    axis=0)
            summaryCDR_std_3_tech2_silver = np.sum(unumpy.std_devs(CDR_3_tech2_silver[CDR_3_tech2_silver['Tot'] > 0]),
                                                    axis=0)
            summaryCDR_1_bronze = np.sum(unumpy.nominal_values(CDR_1_bronze[CDR_1_bronze['Tot'] > 0]), axis=0)
            summaryCDR_2_bronze = np.sum(unumpy.nominal_values(CDR_2_bronze[CDR_2_bronze['Tot'] > 0]), axis=0)
            summaryCDR_3_bronze = np.sum(unumpy.nominal_values(CDR_3_bronze[CDR_3_bronze['Tot'] > 0]), axis=0)
            summaryCDR_std_1_bronze = np.sum(unumpy.std_devs(CDR_1_bronze[CDR_1_bronze['Tot'] > 0]), axis=0)
            summaryCDR_std_2_bronze = np.sum(unumpy.std_devs(CDR_2_bronze[CDR_2_bronze['Tot'] > 0]), axis=0)
            summaryCDR_std_3_bronze = np.sum(unumpy.std_devs(CDR_3_bronze[CDR_3_bronze['Tot'] > 0]), axis=0)
            summaryCDR_1_tech1_bronze = np.sum(
                unumpy.nominal_values(CDR_1_tech1_bronze[CDR_1_tech1_bronze['Tot'] > 0]), axis=0)
            summaryCDR_2_tech1_bronze = np.sum(
                unumpy.nominal_values(CDR_2_tech1_bronze[CDR_2_tech1_bronze['Tot'] > 0]), axis=0)
            summaryCDR_3_tech1_bronze = np.sum(
                unumpy.nominal_values(CDR_3_tech1_bronze[CDR_3_tech1_bronze['Tot'] > 0]), axis=0)
            summaryCDR_std_1_tech1_bronze = np.sum(unumpy.std_devs(CDR_1_tech1_bronze[CDR_1_tech1_bronze['Tot'] > 0]),
                                                    axis=0)
            summaryCDR_std_2_tech1_bronze = np.sum(unumpy.std_devs(CDR_2_tech1_bronze[CDR_2_tech1_bronze['Tot'] > 0]),
                                                    axis=0)
            summaryCDR_std_3_tech1_bronze = np.sum(unumpy.std_devs(CDR_3_tech1_bronze[CDR_3_tech1_bronze['Tot'] > 0]),
                                                    axis=0)
            summaryCDR_1_tech2_bronze = np.sum(
                unumpy.nominal_values(CDR_1_tech2_bronze[CDR_1_tech2_bronze['Tot'] > 0]), axis=0)
            summaryCDR_2_tech2_bronze = np.sum(
                unumpy.nominal_values(CDR_2_tech2_bronze[CDR_2_tech2_bronze['Tot'] > 0]), axis=0)
            summaryCDR_3_tech2_bronze = np.sum(
                unumpy.nominal_values(CDR_3_tech2_bronze[CDR_3_tech2_bronze['Tot'] > 0]), axis=0)
            summaryCDR_std_1_tech2_bronze = np.sum(unumpy.std_devs(CDR_1_tech2_bronze[CDR_1_tech2_bronze['Tot'] > 0]),
                                                    axis=0)
            summaryCDR_std_2_tech2_bronze = np.sum(unumpy.std_devs(CDR_2_tech2_bronze[CDR_2_tech2_bronze['Tot'] > 0]),
                                                    axis=0)
            summaryCDR_std_3_tech2_bronze = np.sum(unumpy.std_devs(CDR_3_tech2_bronze[CDR_3_tech2_bronze['Tot'] > 0]),
                                                    axis=0)
        elif sign == 'negative':
            summaryCDR_1_gold = np.sum(unumpy.nominal_values(CDR_1_gold[CDR_1_gold['Tot'] < 0]), axis=0)
            summaryCDR_2_gold = np.sum(unumpy.nominal_values(CDR_2_gold[CDR_2_gold['Tot'] < 0]), axis=0)
            summaryCDR_3_gold = np.sum(unumpy.nominal_values(CDR_3_gold[CDR_3_gold['Tot'] < 0]), axis=0)
            summaryCDR_std_1_gold = np.sum(unumpy.std_devs(CDR_1_gold[CDR_1_gold['Tot'] < 0]), axis=0)
            summaryCDR_std_2_gold = np.sum(unumpy.std_devs(CDR_2_gold[CDR_2_gold['Tot'] < 0]), axis=0)
            summaryCDR_std_3_gold = np.sum(unumpy.std_devs(CDR_3_gold[CDR_3_gold['Tot'] < 0]), axis=0)
            summaryCDR_1_tech1_gold = np.sum(unumpy.nominal_values(CDR_1_tech1_gold[CDR_1_tech1_gold['Tot'] < 0]),
                                              axis=0)
            summaryCDR_2_tech1_gold = np.sum(unumpy.nominal_values(CDR_2_tech1_gold[CDR_2_tech1_gold['Tot'] < 0]),
                                              axis=0)
            summaryCDR_3_tech1_gold = np.sum(unumpy.nominal_values(CDR_3_tech1_gold[CDR_3_tech1_gold['Tot'] < 0]),
                                              axis=0)
            summaryCDR_std_1_tech1_gold = np.sum(unumpy.std_devs(CDR_1_tech1_gold[CDR_1_tech1_gold['Tot'] < 0]),
                                                  axis=0)
            summaryCDR_std_2_tech1_gold = np.sum(unumpy.std_devs(CDR_2_tech1_gold[CDR_2_tech1_gold['Tot'] < 0]),
                                                  axis=0)
            summaryCDR_std_3_tech1_gold = np.sum(unumpy.std_devs(CDR_3_tech1_gold[CDR_3_tech1_gold['Tot'] < 0]),
                                                  axis=0)
            summaryCDR_1_tech2_gold = np.sum(unumpy.nominal_values(CDR_1_tech2_gold[CDR_1_tech2_gold['Tot'] < 0]),
                                              axis=0)
            summaryCDR_2_tech2_gold = np.sum(unumpy.nominal_values(CDR_2_tech2_gold[CDR_2_tech2_gold['Tot'] < 0]),
                                              axis=0)
            summaryCDR_3_tech2_gold = np.sum(unumpy.nominal_values(CDR_3_tech2_gold[CDR_3_tech2_gold['Tot'] < 0]),
                                              axis=0)
            summaryCDR_std_1_tech2_gold = np.sum(unumpy.std_devs(CDR_1_tech2_gold[CDR_1_tech2_gold['Tot'] < 0]),
                                                  axis=0)
            summaryCDR_std_2_tech2_gold = np.sum(unumpy.std_devs(CDR_2_tech2_gold[CDR_2_tech2_gold['Tot'] < 0]),
                                                  axis=0)
            summaryCDR_std_3_tech2_gold = np.sum(unumpy.std_devs(CDR_3_tech2_gold[CDR_3_tech2_gold['Tot'] < 0]),
                                                  axis=0)
            summaryCDR_1_silver = np.sum(unumpy.nominal_values(CDR_1_silver[CDR_1_silver['Tot'] < 0]), axis=0)
            summaryCDR_2_silver = np.sum(unumpy.nominal_values(CDR_2_silver[CDR_2_silver['Tot'] < 0]), axis=0)
            summaryCDR_3_silver = np.sum(unumpy.nominal_values(CDR_3_silver[CDR_3_silver['Tot'] < 0]), axis=0)
            summaryCDR_std_1_silver = np.sum(unumpy.std_devs(CDR_1_silver[CDR_1_silver['Tot'] < 0]), axis=0)
            summaryCDR_std_2_silver = np.sum(unumpy.std_devs(CDR_2_silver[CDR_2_silver['Tot'] < 0]), axis=0)
            summaryCDR_std_3_silver = np.sum(unumpy.std_devs(CDR_3_silver[CDR_3_silver['Tot'] < 0]), axis=0)
            summaryCDR_1_tech1_silver = np.sum(
                unumpy.nominal_values(CDR_1_tech1_silver[CDR_1_tech1_silver['Tot'] < 0]), axis=0)
            summaryCDR_2_tech1_silver = np.sum(
                unumpy.nominal_values(CDR_2_tech1_silver[CDR_2_tech1_silver['Tot'] < 0]), axis=0)
            summaryCDR_3_tech1_silver = np.sum(
                unumpy.nominal_values(CDR_3_tech1_silver[CDR_3_tech1_silver['Tot'] < 0]), axis=0)
            summaryCDR_std_1_tech1_silver = np.sum(unumpy.std_devs(CDR_1_tech1_silver[CDR_1_tech1_silver['Tot'] < 0]),
                                                    axis=0)
            summaryCDR_std_2_tech1_silver = np.sum(unumpy.std_devs(CDR_2_tech1_silver[CDR_2_tech1_silver['Tot'] < 0]),
                                                    axis=0)
            summaryCDR_std_3_tech1_silver = np.sum(unumpy.std_devs(CDR_3_tech1_silver[CDR_3_tech1_silver['Tot'] < 0]),
                                                    axis=0)
            summaryCDR_1_tech2_silver = np.sum(
                unumpy.nominal_values(CDR_1_tech2_silver[CDR_1_tech2_silver['Tot'] < 0]), axis=0)
            summaryCDR_2_tech2_silver = np.sum(
                unumpy.nominal_values(CDR_2_tech2_silver[CDR_2_tech2_silver['Tot'] < 0]), axis=0)
            summaryCDR_3_tech2_silver = np.sum(
                unumpy.nominal_values(CDR_3_tech2_silver[CDR_3_tech2_silver['Tot'] < 0]), axis=0)
            summaryCDR_std_1_tech2_silver = np.sum(unumpy.std_devs(CDR_1_tech2_silver[CDR_1_tech2_silver['Tot'] < 0]),
                                                    axis=0)
            summaryCDR_std_2_tech2_silver = np.sum(unumpy.std_devs(CDR_2_tech2_silver[CDR_2_tech2_silver['Tot'] < 0]),
                                                    axis=0)
            summaryCDR_std_3_tech2_silver = np.sum(unumpy.std_devs(CDR_3_tech2_silver[CDR_3_tech2_silver['Tot'] < 0]),
                                                    axis=0)
            summaryCDR_1_bronze = np.sum(unumpy.nominal_values(CDR_1_bronze[CDR_1_bronze['Tot'] < 0]), axis=0)
            summaryCDR_2_bronze = np.sum(unumpy.nominal_values(CDR_2_bronze[CDR_2_bronze['Tot'] < 0]), axis=0)
            summaryCDR_3_bronze = np.sum(unumpy.nominal_values(CDR_3_bronze[CDR_3_bronze['Tot'] < 0]), axis=0)
            summaryCDR_std_1_bronze = np.sum(unumpy.std_devs(CDR_1_bronze[CDR_1_bronze['Tot'] < 0]), axis=0)
            summaryCDR_std_2_bronze = np.sum(unumpy.std_devs(CDR_2_bronze[CDR_2_bronze['Tot'] < 0]), axis=0)
            summaryCDR_std_3_bronze = np.sum(unumpy.std_devs(CDR_3_bronze[CDR_3_bronze['Tot'] < 0]), axis=0)
            summaryCDR_1_tech1_bronze = np.sum(
                unumpy.nominal_values(CDR_1_tech1_bronze[CDR_1_tech1_bronze['Tot'] < 0]), axis=0)
            summaryCDR_2_tech1_bronze = np.sum(
                unumpy.nominal_values(CDR_2_tech1_bronze[CDR_2_tech1_bronze['Tot'] < 0]), axis=0)
            summaryCDR_3_tech1_bronze = np.sum(
                unumpy.nominal_values(CDR_3_tech1_bronze[CDR_3_tech1_bronze['Tot'] < 0]), axis=0)
            summaryCDR_std_1_tech1_bronze = np.sum(unumpy.std_devs(CDR_1_tech1_bronze[CDR_1_tech1_bronze['Tot'] < 0]),
                                                    axis=0)
            summaryCDR_std_2_tech1_bronze = np.sum(unumpy.std_devs(CDR_2_tech1_bronze[CDR_2_tech1_bronze['Tot'] < 0]),
                                                    axis=0)
            summaryCDR_std_3_tech1_bronze = np.sum(unumpy.std_devs(CDR_3_tech1_bronze[CDR_3_tech1_bronze['Tot'] < 0]),
                                                    axis=0)
            summaryCDR_1_tech2_bronze = np.sum(
                unumpy.nominal_values(CDR_1_tech2_bronze[CDR_1_tech2_bronze['Tot'] < 0]), axis=0)
            summaryCDR_2_tech2_bronze = np.sum(
                unumpy.nominal_values(CDR_2_tech2_bronze[CDR_2_tech2_bronze['Tot'] < 0]), axis=0)
            summaryCDR_3_tech2_bronze = np.sum(
                unumpy.nominal_values(CDR_3_tech2_bronze[CDR_3_tech2_bronze['Tot'] < 0]), axis=0)
            summaryCDR_std_1_tech2_bronze = np.sum(unumpy.std_devs(CDR_1_tech2_bronze[CDR_1_tech2_bronze['Tot'] < 0]),
                                                    axis=0)
            summaryCDR_std_2_tech2_bronze = np.sum(unumpy.std_devs(CDR_2_tech2_bronze[CDR_2_tech2_bronze['Tot'] < 0]),
                                                    axis=0)
            summaryCDR_std_3_tech2_bronze = np.sum(unumpy.std_devs(CDR_3_tech2_bronze[CDR_3_tech2_bronze['Tot'] < 0]),
                                                    axis=0)
    elif what == 'mean':
        summaryCDR_1_gold = np.mean(unumpy.nominal_values(CDR_1_gold[CDR_1_gold.index >= '2020']), axis = 0)
        summaryCDR_2_gold = np.mean(unumpy.nominal_values(CDR_2_gold[CDR_2_gold.index >= '2020']), axis = 0)
        summaryCDR_3_gold = np.mean(unumpy.nominal_values(CDR_3_gold[CDR_3_gold.index >= '2020']), axis = 0)
        summaryCDR_std_1_gold = np.mean(unumpy.std_devs(CDR_1_gold[CDR_1_gold.index >= '2020']), axis = 0)
        summaryCDR_std_2_gold = np.mean(unumpy.std_devs(CDR_2_gold[CDR_2_gold.index >= '2020']), axis = 0)
        summaryCDR_std_3_gold = np.mean(unumpy.std_devs(CDR_3_gold[CDR_3_gold.index >= '2020']), axis = 0)
        summaryCDR_1_tech1_gold = np.mean(unumpy.nominal_values(CDR_1_tech1_gold[CDR_1_tech1_gold.index >= '2020']), axis = 0)
        summaryCDR_2_tech1_gold = np.mean(unumpy.nominal_values(CDR_2_tech1_gold[CDR_2_tech1_gold.index >= '2020']), axis = 0)
        summaryCDR_3_tech1_gold = np.mean(unumpy.nominal_values(CDR_3_tech1_gold[CDR_3_tech1_gold.index >= '2020']), axis = 0)
        summaryCDR_std_1_tech1_gold = np.mean(unumpy.std_devs(CDR_1_tech1_gold[CDR_1_tech1_gold.index >= '2020']), axis = 0)
        summaryCDR_std_2_tech1_gold = np.mean(unumpy.std_devs(CDR_2_tech1_gold[CDR_2_tech1_gold.index >= '2020']), axis = 0)
        summaryCDR_std_3_tech1_gold = np.mean(unumpy.std_devs(CDR_3_tech1_gold[CDR_3_tech1_gold.index >= '2020']), axis = 0)
        summaryCDR_1_tech2_gold = np.mean(unumpy.nominal_values(CDR_1_tech2_gold[CDR_1_tech2_gold.index >= '2020']), axis = 0)
        summaryCDR_2_tech2_gold = np.mean(unumpy.nominal_values(CDR_2_tech2_gold[CDR_2_tech2_gold.index >= '2020']), axis = 0)
        summaryCDR_3_tech2_gold = np.mean(unumpy.nominal_values(CDR_3_tech2_gold[CDR_3_tech2_gold.index >= '2020']), axis = 0)
        summaryCDR_std_1_tech2_gold = np.mean(unumpy.std_devs(CDR_1_tech2_gold[CDR_1_tech2_gold.index >= '2020']), axis = 0)
        summaryCDR_std_2_tech2_gold = np.mean(unumpy.std_devs(CDR_2_tech2_gold[CDR_2_tech2_gold.index >= '2020']), axis = 0)
        summaryCDR_std_3_tech2_gold = np.mean(unumpy.std_devs(CDR_3_tech2_gold[CDR_3_tech2_gold.index >= '2020']), axis = 0)
        summaryCDR_1_silver = np.mean(unumpy.nominal_values(CDR_1_silver[CDR_1_silver.index >= '2020']), axis = 0)
        summaryCDR_2_silver = np.mean(unumpy.nominal_values(CDR_2_silver[CDR_2_silver.index >= '2020']), axis = 0)
        summaryCDR_3_silver = np.mean(unumpy.nominal_values(CDR_3_silver[CDR_3_silver.index >= '2020']), axis = 0)
        summaryCDR_std_1_silver = np.mean(unumpy.std_devs(CDR_1_silver[CDR_1_silver.index >= '2020']), axis = 0)
        summaryCDR_std_2_silver = np.mean(unumpy.std_devs(CDR_2_silver[CDR_2_silver.index >= '2020']), axis = 0)
        summaryCDR_std_3_silver = np.mean(unumpy.std_devs(CDR_3_silver[CDR_3_silver.index >= '2020']), axis = 0)
        summaryCDR_1_tech1_silver = np.mean(unumpy.nominal_values(CDR_1_tech1_silver[CDR_1_tech1_silver.index >= '2020']), axis = 0)
        summaryCDR_2_tech1_silver = np.mean(unumpy.nominal_values(CDR_2_tech1_silver[CDR_2_tech1_silver.index >= '2020']), axis = 0)
        summaryCDR_3_tech1_silver = np.mean(unumpy.nominal_values(CDR_3_tech1_silver[CDR_3_tech1_silver.index >= '2020']), axis = 0)
        summaryCDR_std_1_tech1_silver = np.mean(unumpy.std_devs(CDR_1_tech1_silver[CDR_1_tech1_silver.index >= '2020']), axis = 0)
        summaryCDR_std_2_tech1_silver = np.mean(unumpy.std_devs(CDR_2_tech1_silver[CDR_2_tech1_silver.index >= '2020']), axis = 0)
        summaryCDR_std_3_tech1_silver = np.mean(unumpy.std_devs(CDR_3_tech1_silver[CDR_3_tech1_silver.index >= '2020']), axis = 0)
        summaryCDR_1_tech2_silver = np.mean(unumpy.nominal_values(CDR_1_tech2_silver[CDR_1_tech2_silver.index >= '2020']), axis = 0)
        summaryCDR_2_tech2_silver = np.mean(unumpy.nominal_values(CDR_2_tech2_silver[CDR_2_tech2_silver.index >= '2020']), axis = 0)
        summaryCDR_3_tech2_silver = np.mean(unumpy.nominal_values(CDR_3_tech2_silver[CDR_3_tech2_silver.index >= '2020']), axis = 0)
        summaryCDR_std_1_tech2_silver = np.mean(unumpy.std_devs(CDR_1_tech2_silver[CDR_1_tech2_silver.index >= '2020']), axis = 0)
        summaryCDR_std_2_tech2_silver = np.mean(unumpy.std_devs(CDR_2_tech2_silver[CDR_2_tech2_silver.index >= '2020']), axis = 0)
        summaryCDR_std_3_tech2_silver = np.mean(unumpy.std_devs(CDR_3_tech2_silver[CDR_3_tech2_silver.index >= '2020']), axis = 0)
        summaryCDR_1_bronze = np.mean(unumpy.nominal_values(CDR_1_bronze[CDR_1_bronze.index >= '2020']), axis = 0)
        summaryCDR_2_bronze = np.mean(unumpy.nominal_values(CDR_2_bronze[CDR_2_bronze.index >= '2020']), axis = 0)
        summaryCDR_3_bronze = np.mean(unumpy.nominal_values(CDR_3_bronze[CDR_3_bronze.index >= '2020']), axis = 0)
        summaryCDR_std_1_bronze = np.mean(unumpy.std_devs(CDR_1_bronze[CDR_1_bronze.index >= '2020']), axis = 0)
        summaryCDR_std_2_bronze = np.mean(unumpy.std_devs(CDR_2_bronze[CDR_2_bronze.index >= '2020']), axis = 0)
        summaryCDR_std_3_bronze = np.mean(unumpy.std_devs(CDR_3_bronze[CDR_3_bronze.index >= '2020']), axis = 0)
        summaryCDR_1_tech1_bronze = np.mean(unumpy.nominal_values(CDR_1_tech1_bronze[CDR_1_tech1_bronze.index >= '2020']), axis = 0)
        summaryCDR_2_tech1_bronze = np.mean(unumpy.nominal_values(CDR_2_tech1_bronze[CDR_2_tech1_bronze.index >= '2020']), axis = 0)
        summaryCDR_3_tech1_bronze = np.mean(unumpy.nominal_values(CDR_3_tech1_bronze[CDR_3_tech1_bronze.index >= '2020']), axis = 0)
        summaryCDR_std_1_tech1_bronze = np.mean(unumpy.std_devs(CDR_1_tech1_bronze[CDR_1_tech1_bronze.index >= '2020']), axis = 0)
        summaryCDR_std_2_tech1_bronze = np.mean(unumpy.std_devs(CDR_2_tech1_bronze[CDR_2_tech1_bronze.index >= '2020']), axis = 0)
        summaryCDR_std_3_tech1_bronze = np.mean(unumpy.std_devs(CDR_3_tech1_bronze[CDR_3_tech1_bronze.index >= '2020']), axis = 0)
        summaryCDR_1_tech2_bronze = np.mean(unumpy.nominal_values(CDR_1_tech2_bronze[CDR_1_tech2_bronze.index >= '2020']), axis = 0)
        summaryCDR_2_tech2_bronze = np.mean(unumpy.nominal_values(CDR_2_tech2_bronze[CDR_2_tech2_bronze.index >= '2020']), axis = 0)
        summaryCDR_3_tech2_bronze = np.mean(unumpy.nominal_values(CDR_3_tech2_bronze[CDR_3_tech2_bronze.index >= '2020']), axis = 0)
        summaryCDR_std_1_tech2_bronze = np.mean(unumpy.std_devs(CDR_1_tech2_bronze[CDR_1_tech2_bronze.index >= '2020']), axis = 0)
        summaryCDR_std_2_tech2_bronze = np.mean(unumpy.std_devs(CDR_2_tech2_bronze[CDR_2_tech2_bronze.index >= '2020']), axis = 0)
        summaryCDR_std_3_tech2_bronze = np.mean(unumpy.std_devs(CDR_3_tech2_bronze[CDR_3_tech2_bronze.index >= '2020']), axis = 0)

    elif what == 'rates':
        summaryCDR_date1_1_gold = unumpy.nominal_values(CDR_1_gold.loc[str(date1)])
        summaryCDR_date1_2_gold = unumpy.nominal_values(CDR_2_gold.loc[str(date1)])
        summaryCDR_date1_3_gold = unumpy.nominal_values(CDR_3_gold.loc[str(date1)])
        summaryCDR_date1_1_silver = unumpy.nominal_values(CDR_1_silver.loc[str(date1)])
        summaryCDR_date1_2_silver = unumpy.nominal_values(CDR_2_silver.loc[str(date1)])
        summaryCDR_date1_3_silver = unumpy.nominal_values(CDR_3_silver.loc[str(date1)])
        summaryCDR_date1_1_bronze = unumpy.nominal_values(CDR_1_bronze.loc[str(date1)])
        summaryCDR_date1_2_bronze = unumpy.nominal_values(CDR_2_bronze.loc[str(date1)])
        summaryCDR_date1_3_bronze = unumpy.nominal_values(CDR_3_bronze.loc[str(date1)])
        summaryCDR_date1_std_1_gold = unumpy.std_devs(CDR_1_gold.loc[str(date1)])
        summaryCDR_date1_std_2_gold = unumpy.std_devs(CDR_2_gold.loc[str(date1)])
        summaryCDR_date1_std_3_gold = unumpy.std_devs(CDR_3_gold.loc[str(date1)])
        summaryCDR_date1_std_1_silver = unumpy.std_devs(CDR_1_silver.loc[str(date1)])
        summaryCDR_date1_std_2_silver = unumpy.std_devs(CDR_2_silver.loc[str(date1)])
        summaryCDR_date1_std_3_silver = unumpy.std_devs(CDR_3_silver.loc[str(date1)])
        summaryCDR_date1_std_1_bronze = unumpy.std_devs(CDR_1_bronze.loc[str(date1)])
        summaryCDR_date1_std_2_bronze = unumpy.std_devs(CDR_2_bronze.loc[str(date1)])
        summaryCDR_date1_std_3_bronze = unumpy.std_devs(CDR_3_bronze.loc[str(date1)])
        summaryCDR_date1_1_tech1_gold = unumpy.nominal_values(CDR_1_tech1_gold.loc[str(date1)])
        summaryCDR_date1_2_tech1_gold = unumpy.nominal_values(CDR_2_tech1_gold.loc[str(date1)])
        summaryCDR_date1_3_tech1_gold = unumpy.nominal_values(CDR_3_tech1_gold.loc[str(date1)])
        summaryCDR_date1_1_tech1_silver = unumpy.nominal_values(CDR_1_tech1_silver.loc[str(date1)])
        summaryCDR_date1_2_tech1_silver = unumpy.nominal_values(CDR_2_tech1_silver.loc[str(date1)])
        summaryCDR_date1_3_tech1_silver = unumpy.nominal_values(CDR_3_tech1_silver.loc[str(date1)])
        summaryCDR_date1_1_tech1_bronze = unumpy.nominal_values(CDR_1_tech1_bronze.loc[str(date1)])
        summaryCDR_date1_2_tech1_bronze = unumpy.nominal_values(CDR_2_tech1_bronze.loc[str(date1)])
        summaryCDR_date1_3_tech1_bronze = unumpy.nominal_values(CDR_3_tech1_bronze.loc[str(date1)])
        summaryCDR_date1_std_1_tech1_gold = unumpy.std_devs(CDR_1_tech1_gold.loc[str(date1)])
        summaryCDR_date1_std_2_tech1_gold = unumpy.std_devs(CDR_2_tech1_gold.loc[str(date1)])
        summaryCDR_date1_std_3_tech1_gold = unumpy.std_devs(CDR_3_tech1_gold.loc[str(date1)])
        summaryCDR_date1_std_1_tech1_silver = unumpy.std_devs(CDR_1_tech1_silver.loc[str(date1)])
        summaryCDR_date1_std_2_tech1_silver = unumpy.std_devs(CDR_2_tech1_silver.loc[str(date1)])
        summaryCDR_date1_std_3_tech1_silver = unumpy.std_devs(CDR_3_tech1_silver.loc[str(date1)])
        summaryCDR_date1_std_1_tech1_bronze = unumpy.std_devs(CDR_1_tech1_bronze.loc[str(date1)])
        summaryCDR_date1_std_2_tech1_bronze = unumpy.std_devs(CDR_2_tech1_bronze.loc[str(date1)])
        summaryCDR_date1_std_3_tech1_bronze = unumpy.std_devs(CDR_3_tech1_bronze.loc[str(date1)])
        summaryCDR_date1_1_tech2_gold = unumpy.nominal_values(CDR_1_tech2_gold.loc[str(date1)])
        summaryCDR_date1_2_tech2_gold = unumpy.nominal_values(CDR_2_tech2_gold.loc[str(date1)])
        summaryCDR_date1_3_tech2_gold = unumpy.nominal_values(CDR_3_tech2_gold.loc[str(date1)])
        summaryCDR_date1_1_tech2_silver = unumpy.nominal_values(CDR_1_tech2_silver.loc[str(date1)])
        summaryCDR_date1_2_tech2_silver = unumpy.nominal_values(CDR_2_tech2_silver.loc[str(date1)])
        summaryCDR_date1_3_tech2_silver = unumpy.nominal_values(CDR_3_tech2_silver.loc[str(date1)])
        summaryCDR_date1_1_tech2_bronze = unumpy.nominal_values(CDR_1_tech2_bronze.loc[str(date1)])
        summaryCDR_date1_2_tech2_bronze = unumpy.nominal_values(CDR_2_tech2_bronze.loc[str(date1)])
        summaryCDR_date1_3_tech2_bronze = unumpy.nominal_values(CDR_3_tech2_bronze.loc[str(date1)])
        summaryCDR_date1_std_1_tech2_gold = unumpy.std_devs(CDR_1_tech2_gold.loc[str(date1)])
        summaryCDR_date1_std_2_tech2_gold = unumpy.std_devs(CDR_2_tech2_gold.loc[str(date1)])
        summaryCDR_date1_std_3_tech2_gold = unumpy.std_devs(CDR_3_tech2_gold.loc[str(date1)])
        summaryCDR_date1_std_1_tech2_silver = unumpy.std_devs(CDR_1_tech2_silver.loc[str(date1)])
        summaryCDR_date1_std_2_tech2_silver = unumpy.std_devs(CDR_2_tech2_silver.loc[str(date1)])
        summaryCDR_date1_std_3_tech2_silver = unumpy.std_devs(CDR_3_tech2_silver.loc[str(date1)])
        summaryCDR_date1_std_1_tech2_bronze = unumpy.std_devs(CDR_1_tech2_bronze.loc[str(date1)])
        summaryCDR_date1_std_2_tech2_bronze = unumpy.std_devs(CDR_2_tech2_bronze.loc[str(date1)])
        summaryCDR_date1_std_3_tech2_bronze = unumpy.std_devs(CDR_3_tech2_bronze.loc[str(date1)])


        summaryCDR_date3_1_gold = unumpy.nominal_values(CDR_1_gold.loc[str(date3)])
        summaryCDR_date3_2_gold = unumpy.nominal_values(CDR_2_gold.loc[str(date3)])
        summaryCDR_date3_3_gold = unumpy.nominal_values(CDR_3_gold.loc[str(date3)])
        summaryCDR_date3_1_silver = unumpy.nominal_values(CDR_1_silver.loc[str(date3)])
        summaryCDR_date3_2_silver = unumpy.nominal_values(CDR_2_silver.loc[str(date3)])
        summaryCDR_date3_3_silver = unumpy.nominal_values(CDR_3_silver.loc[str(date3)])
        summaryCDR_date3_1_bronze = unumpy.nominal_values(CDR_1_bronze.loc[str(date3)])
        summaryCDR_date3_2_bronze = unumpy.nominal_values(CDR_2_bronze.loc[str(date3)])
        summaryCDR_date3_3_bronze = unumpy.nominal_values(CDR_3_bronze.loc[str(date3)])
        summaryCDR_date3_std_1_gold = unumpy.std_devs(CDR_1_gold.loc[str(date3)])
        summaryCDR_date3_std_2_gold = unumpy.std_devs(CDR_2_gold.loc[str(date3)])
        summaryCDR_date3_std_3_gold = unumpy.std_devs(CDR_3_gold.loc[str(date3)])
        summaryCDR_date3_std_1_silver = unumpy.std_devs(CDR_1_silver.loc[str(date3)])
        summaryCDR_date3_std_2_silver = unumpy.std_devs(CDR_2_silver.loc[str(date3)])
        summaryCDR_date3_std_3_silver = unumpy.std_devs(CDR_3_silver.loc[str(date3)])
        summaryCDR_date3_std_1_bronze = unumpy.std_devs(CDR_1_bronze.loc[str(date3)])
        summaryCDR_date3_std_2_bronze = unumpy.std_devs(CDR_2_bronze.loc[str(date3)])
        summaryCDR_date3_std_3_bronze = unumpy.std_devs(CDR_3_bronze.loc[str(date3)])
        summaryCDR_date3_1_tech1_gold = unumpy.nominal_values(CDR_1_tech1_gold.loc[str(date3)])
        summaryCDR_date3_2_tech1_gold = unumpy.nominal_values(CDR_2_tech1_gold.loc[str(date3)])
        summaryCDR_date3_3_tech1_gold = unumpy.nominal_values(CDR_3_tech1_gold.loc[str(date3)])
        summaryCDR_date3_1_tech1_silver = unumpy.nominal_values(CDR_1_tech1_silver.loc[str(date3)])
        summaryCDR_date3_2_tech1_silver = unumpy.nominal_values(CDR_2_tech1_silver.loc[str(date3)])
        summaryCDR_date3_3_tech1_silver = unumpy.nominal_values(CDR_3_tech1_silver.loc[str(date3)])
        summaryCDR_date3_1_tech1_bronze = unumpy.nominal_values(CDR_1_tech1_bronze.loc[str(date3)])
        summaryCDR_date3_2_tech1_bronze = unumpy.nominal_values(CDR_2_tech1_bronze.loc[str(date3)])
        summaryCDR_date3_3_tech1_bronze = unumpy.nominal_values(CDR_3_tech1_bronze.loc[str(date3)])
        summaryCDR_date3_std_1_tech1_gold = unumpy.std_devs(CDR_1_tech1_gold.loc[str(date3)])
        summaryCDR_date3_std_2_tech1_gold = unumpy.std_devs(CDR_2_tech1_gold.loc[str(date3)])
        summaryCDR_date3_std_3_tech1_gold = unumpy.std_devs(CDR_3_tech1_gold.loc[str(date3)])
        summaryCDR_date3_std_1_tech1_silver = unumpy.std_devs(CDR_1_tech1_silver.loc[str(date3)])
        summaryCDR_date3_std_2_tech1_silver = unumpy.std_devs(CDR_2_tech1_silver.loc[str(date3)])
        summaryCDR_date3_std_3_tech1_silver = unumpy.std_devs(CDR_3_tech1_silver.loc[str(date3)])
        summaryCDR_date3_std_1_tech1_bronze = unumpy.std_devs(CDR_1_tech1_bronze.loc[str(date3)])
        summaryCDR_date3_std_2_tech1_bronze = unumpy.std_devs(CDR_2_tech1_bronze.loc[str(date3)])
        summaryCDR_date3_std_3_tech1_bronze = unumpy.std_devs(CDR_3_tech1_bronze.loc[str(date3)])
        summaryCDR_date3_1_tech2_gold = unumpy.nominal_values(CDR_1_tech2_gold.loc[str(date3)])
        summaryCDR_date3_2_tech2_gold = unumpy.nominal_values(CDR_2_tech2_gold.loc[str(date3)])
        summaryCDR_date3_3_tech2_gold = unumpy.nominal_values(CDR_3_tech2_gold.loc[str(date3)])
        summaryCDR_date3_1_tech2_silver = unumpy.nominal_values(CDR_1_tech2_silver.loc[str(date3)])
        summaryCDR_date3_2_tech2_silver = unumpy.nominal_values(CDR_2_tech2_silver.loc[str(date3)])
        summaryCDR_date3_3_tech2_silver = unumpy.nominal_values(CDR_3_tech2_silver.loc[str(date3)])
        summaryCDR_date3_1_tech2_bronze = unumpy.nominal_values(CDR_1_tech2_bronze.loc[str(date3)])
        summaryCDR_date3_2_tech2_bronze = unumpy.nominal_values(CDR_2_tech2_bronze.loc[str(date3)])
        summaryCDR_date3_3_tech2_bronze = unumpy.nominal_values(CDR_3_tech2_bronze.loc[str(date3)])
        summaryCDR_date3_std_1_tech2_gold = unumpy.std_devs(CDR_1_tech2_gold.loc[str(date3)])
        summaryCDR_date3_std_2_tech2_gold = unumpy.std_devs(CDR_2_tech2_gold.loc[str(date3)])
        summaryCDR_date3_std_3_tech2_gold = unumpy.std_devs(CDR_3_tech2_gold.loc[str(date3)])
        summaryCDR_date3_std_1_tech2_silver = unumpy.std_devs(CDR_1_tech2_silver.loc[str(date3)])
        summaryCDR_date3_std_2_tech2_silver = unumpy.std_devs(CDR_2_tech2_silver.loc[str(date3)])
        summaryCDR_date3_std_3_tech2_silver = unumpy.std_devs(CDR_3_tech2_silver.loc[str(date3)])
        summaryCDR_date3_std_1_tech2_bronze = unumpy.std_devs(CDR_1_tech2_bronze.loc[str(date3)])
        summaryCDR_date3_std_2_tech2_bronze = unumpy.std_devs(CDR_2_tech2_bronze.loc[str(date3)])
        summaryCDR_date3_std_3_tech2_bronze = unumpy.std_devs(CDR_3_tech2_bronze.loc[str(date3)])
        summaryCDR_date2_1_gold = unumpy.nominal_values(CDR_1_gold.loc[str(date2)])
        summaryCDR_date2_2_gold = unumpy.nominal_values(CDR_2_gold.loc[str(date2)])
        summaryCDR_date2_3_gold = unumpy.nominal_values(CDR_3_gold.loc[str(date2)])
        summaryCDR_date2_1_silver = unumpy.nominal_values(CDR_1_silver.loc[str(date2)])
        summaryCDR_date2_2_silver = unumpy.nominal_values(CDR_2_silver.loc[str(date2)])
        summaryCDR_date2_3_silver = unumpy.nominal_values(CDR_3_silver.loc[str(date2)])
        summaryCDR_date2_1_bronze = unumpy.nominal_values(CDR_1_bronze.loc[str(date2)])
        summaryCDR_date2_2_bronze = unumpy.nominal_values(CDR_2_bronze.loc[str(date2)])
        summaryCDR_date2_3_bronze = unumpy.nominal_values(CDR_3_bronze.loc[str(date2)])
        summaryCDR_date2_std_1_gold = unumpy.std_devs(CDR_1_gold.loc[str(date2)])
        summaryCDR_date2_std_2_gold = unumpy.std_devs(CDR_2_gold.loc[str(date2)])
        summaryCDR_date2_std_3_gold = unumpy.std_devs(CDR_3_gold.loc[str(date2)])
        summaryCDR_date2_std_1_silver = unumpy.std_devs(CDR_1_silver.loc[str(date2)])
        summaryCDR_date2_std_2_silver = unumpy.std_devs(CDR_2_silver.loc[str(date2)])
        summaryCDR_date2_std_3_silver = unumpy.std_devs(CDR_3_silver.loc[str(date2)])
        summaryCDR_date2_std_1_bronze = unumpy.std_devs(CDR_1_bronze.loc[str(date2)])
        summaryCDR_date2_std_2_bronze = unumpy.std_devs(CDR_2_bronze.loc[str(date2)])
        summaryCDR_date2_std_3_bronze = unumpy.std_devs(CDR_3_bronze.loc[str(date2)])
        summaryCDR_date2_1_tech1_gold = unumpy.nominal_values(CDR_1_tech1_gold.loc[str(date2)])
        summaryCDR_date2_2_tech1_gold = unumpy.nominal_values(CDR_2_tech1_gold.loc[str(date2)])
        summaryCDR_date2_3_tech1_gold = unumpy.nominal_values(CDR_3_tech1_gold.loc[str(date2)])
        summaryCDR_date2_1_tech1_silver = unumpy.nominal_values(CDR_1_tech1_silver.loc[str(date2)])
        summaryCDR_date2_2_tech1_silver = unumpy.nominal_values(CDR_2_tech1_silver.loc[str(date2)])
        summaryCDR_date2_3_tech1_silver = unumpy.nominal_values(CDR_3_tech1_silver.loc[str(date2)])
        summaryCDR_date2_1_tech1_bronze = unumpy.nominal_values(CDR_1_tech1_bronze.loc[str(date2)])
        summaryCDR_date2_2_tech1_bronze = unumpy.nominal_values(CDR_2_tech1_bronze.loc[str(date2)])
        summaryCDR_date2_3_tech1_bronze = unumpy.nominal_values(CDR_3_tech1_bronze.loc[str(date2)])
        summaryCDR_date2_std_1_tech1_gold = unumpy.std_devs(CDR_1_tech1_gold.loc[str(date2)])
        summaryCDR_date2_std_2_tech1_gold = unumpy.std_devs(CDR_2_tech1_gold.loc[str(date2)])
        summaryCDR_date2_std_3_tech1_gold = unumpy.std_devs(CDR_3_tech1_gold.loc[str(date2)])
        summaryCDR_date2_std_1_tech1_silver = unumpy.std_devs(CDR_1_tech1_silver.loc[str(date2)])
        summaryCDR_date2_std_2_tech1_silver = unumpy.std_devs(CDR_2_tech1_silver.loc[str(date2)])
        summaryCDR_date2_std_3_tech1_silver = unumpy.std_devs(CDR_3_tech1_silver.loc[str(date2)])
        summaryCDR_date2_std_1_tech1_bronze = unumpy.std_devs(CDR_1_tech1_bronze.loc[str(date2)])
        summaryCDR_date2_std_2_tech1_bronze = unumpy.std_devs(CDR_2_tech1_bronze.loc[str(date2)])
        summaryCDR_date2_std_3_tech1_bronze = unumpy.std_devs(CDR_3_tech1_bronze.loc[str(date2)])
        summaryCDR_date2_1_tech2_gold = unumpy.nominal_values(CDR_1_tech2_gold.loc[str(date2)])
        summaryCDR_date2_2_tech2_gold = unumpy.nominal_values(CDR_2_tech2_gold.loc[str(date2)])
        summaryCDR_date2_3_tech2_gold = unumpy.nominal_values(CDR_3_tech2_gold.loc[str(date2)])
        summaryCDR_date2_1_tech2_silver = unumpy.nominal_values(CDR_1_tech2_silver.loc[str(date2)])
        summaryCDR_date2_2_tech2_silver = unumpy.nominal_values(CDR_2_tech2_silver.loc[str(date2)])
        summaryCDR_date2_3_tech2_silver = unumpy.nominal_values(CDR_3_tech2_silver.loc[str(date2)])
        summaryCDR_date2_1_tech2_bronze = unumpy.nominal_values(CDR_1_tech2_bronze.loc[str(date2)])
        summaryCDR_date2_2_tech2_bronze = unumpy.nominal_values(CDR_2_tech2_bronze.loc[str(date2)])
        summaryCDR_date2_3_tech2_bronze = unumpy.nominal_values(CDR_3_tech2_bronze.loc[str(date2)])
        summaryCDR_date2_std_1_tech2_gold = unumpy.std_devs(CDR_1_tech2_gold.loc[str(date2)])
        summaryCDR_date2_std_2_tech2_gold = unumpy.std_devs(CDR_2_tech2_gold.loc[str(date2)])
        summaryCDR_date2_std_3_tech2_gold = unumpy.std_devs(CDR_3_tech2_gold.loc[str(date2)])
        summaryCDR_date2_std_1_tech2_silver = unumpy.std_devs(CDR_1_tech2_silver.loc[str(date2)])
        summaryCDR_date2_std_2_tech2_silver = unumpy.std_devs(CDR_2_tech2_silver.loc[str(date2)])
        summaryCDR_date2_std_3_tech2_silver = unumpy.std_devs(CDR_3_tech2_silver.loc[str(date2)])
        summaryCDR_date2_std_1_tech2_bronze = unumpy.std_devs(CDR_1_tech2_bronze.loc[str(date2)])
        summaryCDR_date2_std_2_tech2_bronze = unumpy.std_devs(CDR_2_tech2_bronze.loc[str(date2)])
        summaryCDR_date2_std_3_tech2_bronze = unumpy.std_devs(CDR_3_tech2_bronze.loc[str(date2)])

    if what == 'rates':
        CDR_summary = pd.DataFrame({"summary_CO2_CDR": [float((summaryCDR_date1_1_gold[:,9])),
                                                        float((summaryCDR_date1_2_gold[:,9])),
                                                        float((summaryCDR_date1_3_gold[:,9])),
                                                        float((summaryCDR_date1_1_silver[:,9])),
                                                        float((summaryCDR_date1_2_silver[:,9])),
                                                        float((summaryCDR_date1_3_silver[:,9])),
                                                        float((summaryCDR_date1_1_bronze[:,9])),
                                                        float((summaryCDR_date1_2_bronze[:,9])),
                                                        float((summaryCDR_date1_3_bronze[:,9])),
                                                        float((summaryCDR_date1_1_tech1_gold[:,9])),
                                                        float((summaryCDR_date1_2_tech1_gold[:,9])),
                                                        float((summaryCDR_date1_3_tech1_gold[:,9])),
                                                        float((summaryCDR_date1_1_tech1_silver[:,9])),
                                                        float((summaryCDR_date1_2_tech1_silver[:,9])),
                                                        float((summaryCDR_date1_3_tech1_silver[:,9])),
                                                        float((summaryCDR_date1_1_tech1_bronze[:,9])),
                                                        float((summaryCDR_date1_2_tech1_bronze[:,9])),
                                                        float((summaryCDR_date1_3_tech1_bronze[:,9])),
                                                        float((summaryCDR_date1_1_tech2_gold[:,9])),
                                                        float((summaryCDR_date1_2_tech2_gold[:,9])),
                                                        float((summaryCDR_date1_3_tech2_gold[:,9])),
                                                        float((summaryCDR_date1_1_tech2_silver[:,9])),
                                                        float((summaryCDR_date1_2_tech2_silver[:,9])),
                                                        float((summaryCDR_date1_3_tech2_silver[:,9])),
                                                        float((summaryCDR_date1_1_tech2_bronze[:,9])),
                                                        float((summaryCDR_date1_2_tech2_bronze[:,9])),
                                                        float((summaryCDR_date1_3_tech2_bronze[:,9])),
                                                        float((summaryCDR_date2_1_gold[:,9])),
                                                        float((summaryCDR_date2_2_gold[:,9])),
                                                        float((summaryCDR_date2_3_gold[:,9])),
                                                        float((summaryCDR_date2_1_silver[:,9])),
                                                        float((summaryCDR_date2_2_silver[:,9])),
                                                        float((summaryCDR_date2_3_silver[:,9])),
                                                        float((summaryCDR_date2_1_bronze[:,9])),
                                                        float((summaryCDR_date2_2_bronze[:,9])),
                                                        float((summaryCDR_date2_3_bronze[:,9])),
                                                        float((summaryCDR_date2_1_tech1_gold[:,9])),
                                                        float((summaryCDR_date2_2_tech1_gold[:,9])),
                                                        float((summaryCDR_date2_3_tech1_gold[:,9])),
                                                        float((summaryCDR_date2_1_tech1_silver[:,9])),
                                                        float((summaryCDR_date2_2_tech1_silver[:,9])),
                                                        float((summaryCDR_date2_3_tech1_silver[:,9])),
                                                        float((summaryCDR_date2_1_tech1_bronze[:,9])),
                                                        float((summaryCDR_date2_2_tech1_bronze[:,9])),
                                                        float((summaryCDR_date2_3_tech1_bronze[:,9])),
                                                        float((summaryCDR_date2_1_tech2_gold[:,9])),
                                                        float((summaryCDR_date2_2_tech2_gold[:,9])),
                                                        float((summaryCDR_date2_3_tech2_gold[:,9])),
                                                        float((summaryCDR_date2_1_tech2_silver[:,9])),
                                                        float((summaryCDR_date2_2_tech2_silver[:,9])),
                                                        float((summaryCDR_date2_3_tech2_silver[:,9])),
                                                        float((summaryCDR_date2_1_tech2_bronze[:,9])),
                                                        float((summaryCDR_date2_2_tech2_bronze[:,9])),
                                                        float((summaryCDR_date2_3_tech2_bronze[:,9])),
                                                        float((summaryCDR_date3_1_gold[:,9])),
                                                        float((summaryCDR_date3_2_gold[:,9])),
                                                        float((summaryCDR_date3_3_gold[:,9])),
                                                        float((summaryCDR_date3_1_silver[:,9])),
                                                        float((summaryCDR_date3_2_silver[:,9])),
                                                        float((summaryCDR_date3_3_silver[:,9])),
                                                        float((summaryCDR_date3_1_bronze[:,9])),
                                                        float((summaryCDR_date3_2_bronze[:,9])),
                                                        float((summaryCDR_date3_3_bronze[:,9])),
                                                        float((summaryCDR_date3_1_tech1_gold[:,9])),
                                                        float((summaryCDR_date3_2_tech1_gold[:,9])),
                                                        float((summaryCDR_date3_3_tech1_gold[:,9])),
                                                        float((summaryCDR_date3_1_tech1_silver[:,9])),
                                                        float((summaryCDR_date3_2_tech1_silver[:,9])),
                                                        float((summaryCDR_date3_3_tech1_silver[:,9])),
                                                        float((summaryCDR_date3_1_tech1_bronze[:,9])),
                                                        float((summaryCDR_date3_2_tech1_bronze[:,9])),
                                                        float((summaryCDR_date3_3_tech1_bronze[:,9])),
                                                        float((summaryCDR_date3_1_tech2_gold[:,9])),
                                                        float((summaryCDR_date3_2_tech2_gold[:,9])),
                                                        float((summaryCDR_date3_3_tech2_gold[:,9])),
                                                        float((summaryCDR_date3_1_tech2_silver[:,9])),
                                                        float((summaryCDR_date3_2_tech2_silver[:,9])),
                                                        float((summaryCDR_date3_3_tech2_silver[:,9])),
                                                        float((summaryCDR_date3_1_tech2_bronze[:,9])),
                                                        float((summaryCDR_date3_2_tech2_bronze[:,9])),
                                                        float((summaryCDR_date3_3_tech2_bronze[:,9])),

                                                        ],
                                    "summary_CO2_CDR_std": [float((summaryCDR_date1_std_1_gold[:,9])),
                                                            float((summaryCDR_date1_std_2_gold[:,9])),
                                                            float((summaryCDR_date1_std_3_gold[:,9])),
                                                            float((summaryCDR_date1_std_1_silver[:,9])),
                                                            float((summaryCDR_date1_std_2_silver[:,9])),
                                                            float((summaryCDR_date1_std_3_silver[:,9])),
                                                            float((summaryCDR_date1_std_1_bronze[:,9])),
                                                            float((summaryCDR_date1_std_2_bronze[:,9])),
                                                            float((summaryCDR_date1_std_3_bronze[:,9])),
                                                            float((summaryCDR_date1_std_1_tech1_gold[:,9])),
                                                            float((summaryCDR_date1_std_2_tech1_gold[:,9])),
                                                            float((summaryCDR_date1_std_3_tech1_gold[:,9])),
                                                            float((summaryCDR_date1_std_1_tech1_silver[:,9])),
                                                            float((summaryCDR_date1_std_2_tech1_silver[:,9])),
                                                            float((summaryCDR_date1_std_3_tech1_silver[:,9])),
                                                            float((summaryCDR_date1_std_1_tech1_bronze[:,9])),
                                                            float((summaryCDR_date1_std_2_tech1_bronze[:,9])),
                                                            float((summaryCDR_date1_std_3_tech1_bronze[:,9])),
                                                            float((summaryCDR_date1_std_1_tech2_gold[:,9])),
                                                            float((summaryCDR_date1_std_2_tech2_gold[:,9])),
                                                            float((summaryCDR_date1_std_3_tech2_gold[:,9])),
                                                            float((summaryCDR_date1_std_1_tech2_silver[:,9])),
                                                            float((summaryCDR_date1_std_2_tech2_silver[:,9])),
                                                            float((summaryCDR_date1_std_3_tech2_silver[:,9])),
                                                            float((summaryCDR_date1_std_1_tech2_bronze[:,9])),
                                                            float((summaryCDR_date1_std_2_tech2_bronze[:,9])),
                                                            float((summaryCDR_date1_std_3_tech2_bronze[:,9])),
                                                            float((summaryCDR_date2_std_1_gold[:,9])),
                                                            float((summaryCDR_date2_std_2_gold[:,9])),
                                                            float((summaryCDR_date2_std_3_gold[:,9])),
                                                            float((summaryCDR_date2_std_1_silver[:,9])),
                                                            float((summaryCDR_date2_std_2_silver[:,9])),
                                                            float((summaryCDR_date2_std_3_silver[:,9])),
                                                            float((summaryCDR_date2_std_1_bronze[:,9])),
                                                            float((summaryCDR_date2_std_2_bronze[:,9])),
                                                            float((summaryCDR_date2_std_3_bronze[:,9])),
                                                            float((summaryCDR_date2_std_1_tech1_gold[:,9])),
                                                            float((summaryCDR_date2_std_2_tech1_gold[:,9])),
                                                            float((summaryCDR_date2_std_3_tech1_gold[:,9])),
                                                            float((summaryCDR_date2_std_1_tech1_silver[:,9])),
                                                            float((summaryCDR_date2_std_2_tech1_silver[:,9])),
                                                            float((summaryCDR_date2_std_3_tech1_silver[:,9])),
                                                            float((summaryCDR_date2_std_1_tech1_bronze[:,9])),
                                                            float((summaryCDR_date2_std_2_tech1_bronze[:,9])),
                                                            float((summaryCDR_date2_std_3_tech1_bronze[:,9])),
                                                            float((summaryCDR_date2_std_1_tech2_gold[:,9])),
                                                            float((summaryCDR_date2_std_2_tech2_gold[:,9])),
                                                            float((summaryCDR_date2_std_3_tech2_gold[:,9])),
                                                            float((summaryCDR_date2_std_1_tech2_silver[:,9])),
                                                            float((summaryCDR_date2_std_2_tech2_silver[:,9])),
                                                            float((summaryCDR_date2_std_3_tech2_silver[:,9])),
                                                            float((summaryCDR_date2_std_1_tech2_bronze[:,9])),
                                                            float((summaryCDR_date2_std_2_tech2_bronze[:,9])),
                                                            float((summaryCDR_date2_std_3_tech2_bronze[:,9])),
                                                            float((summaryCDR_date3_std_1_gold[:,9])),
                                                            float((summaryCDR_date3_std_2_gold[:,9])),
                                                            float((summaryCDR_date3_std_3_gold[:,9])),
                                                            float((summaryCDR_date3_std_1_silver[:,9])),
                                                            float((summaryCDR_date3_std_2_silver[:,9])),
                                                            float((summaryCDR_date3_std_3_silver[:,9])),
                                                            float((summaryCDR_date3_std_1_bronze[:,9])),
                                                            float((summaryCDR_date3_std_2_bronze[:,9])),
                                                            float((summaryCDR_date3_std_3_bronze[:,9])),
                                                            float((summaryCDR_date3_std_1_tech1_gold[:,9])),
                                                            float((summaryCDR_date3_std_2_tech1_gold[:,9])),
                                                            float((summaryCDR_date3_std_3_tech1_gold[:,9])),
                                                            float((summaryCDR_date3_std_1_tech1_silver[:,9])),
                                                            float((summaryCDR_date3_std_2_tech1_silver[:,9])),
                                                            float((summaryCDR_date3_std_3_tech1_silver[:,9])),
                                                            float((summaryCDR_date3_std_1_tech1_bronze[:,9])),
                                                            float((summaryCDR_date3_std_2_tech1_bronze[:,9])),
                                                            float((summaryCDR_date3_std_3_tech1_bronze[:,9])),
                                                            float((summaryCDR_date3_std_1_tech2_gold[:,9])),
                                                            float((summaryCDR_date3_std_2_tech2_gold[:,9])),
                                                            float((summaryCDR_date3_std_3_tech2_gold[:,9])),
                                                            float((summaryCDR_date3_std_1_tech2_silver[:,9])),
                                                            float((summaryCDR_date3_std_2_tech2_silver[:,9])),
                                                            float((summaryCDR_date3_std_3_tech2_silver[:,9])),
                                                            float((summaryCDR_date3_std_1_tech2_bronze[:,9])),
                                                            float((summaryCDR_date3_std_2_tech2_bronze[:,9])),
                                                            float((summaryCDR_date3_std_3_tech2_bronze[:,9]))
                                                            ],
                                    "summary_Tot_CDR": [float((summaryCDR_date1_1_gold[:,11])),
                                                        float((summaryCDR_date1_2_gold[:,11])),
                                                        float((summaryCDR_date1_3_gold[:,11])),
                                                        float((summaryCDR_date1_1_silver[:,11])),
                                                        float((summaryCDR_date1_2_silver[:,11])),
                                                        float((summaryCDR_date1_3_silver[:,11])),
                                                        float((summaryCDR_date1_1_bronze[:,11])),
                                                        float((summaryCDR_date1_2_bronze[:,11])),
                                                        float((summaryCDR_date1_3_bronze[:,11])),
                                                        float((summaryCDR_date1_1_tech1_gold[:,11])),
                                                        float((summaryCDR_date1_2_tech1_gold[:,11])),
                                                        float((summaryCDR_date1_3_tech1_gold[:,11])),
                                                        float((summaryCDR_date1_1_tech1_silver[:,11])),
                                                        float((summaryCDR_date1_2_tech1_silver[:,11])),
                                                        float((summaryCDR_date1_3_tech1_silver[:,11])),
                                                        float((summaryCDR_date1_1_tech1_bronze[:,11])),
                                                        float((summaryCDR_date1_2_tech1_bronze[:,11])),
                                                        float((summaryCDR_date1_3_tech1_bronze[:,11])),
                                                        float((summaryCDR_date1_1_tech2_gold[:,11])),
                                                        float((summaryCDR_date1_2_tech2_gold[:,11])),
                                                        float((summaryCDR_date1_3_tech2_gold[:,11])),
                                                        float((summaryCDR_date1_1_tech2_silver[:,11])),
                                                        float((summaryCDR_date1_2_tech2_silver[:,11])),
                                                        float((summaryCDR_date1_3_tech2_silver[:,11])),
                                                        float((summaryCDR_date1_1_tech2_bronze[:,11])),
                                                        float((summaryCDR_date1_2_tech2_bronze[:,11])),
                                                        float((summaryCDR_date1_3_tech2_bronze[:,11])),
                                                        float((summaryCDR_date2_1_gold[:,11])),
                                                        float((summaryCDR_date2_2_gold[:,11])),
                                                        float((summaryCDR_date2_3_gold[:,11])),
                                                        float((summaryCDR_date2_1_silver[:,11])),
                                                        float((summaryCDR_date2_2_silver[:,11])),
                                                        float((summaryCDR_date2_3_silver[:,11])),
                                                        float((summaryCDR_date2_1_bronze[:,11])),
                                                        float((summaryCDR_date2_2_bronze[:,11])),
                                                        float((summaryCDR_date2_3_bronze[:,11])),
                                                        float((summaryCDR_date2_1_tech1_gold[:,11])),
                                                        float((summaryCDR_date2_2_tech1_gold[:,11])),
                                                        float((summaryCDR_date2_3_tech1_gold[:,11])),
                                                        float((summaryCDR_date2_1_tech1_silver[:,11])),
                                                        float((summaryCDR_date2_2_tech1_silver[:,11])),
                                                        float((summaryCDR_date2_3_tech1_silver[:,11])),
                                                        float((summaryCDR_date2_1_tech1_bronze[:,11])),
                                                        float((summaryCDR_date2_2_tech1_bronze[:,11])),
                                                        float((summaryCDR_date2_3_tech1_bronze[:,11])),
                                                        float((summaryCDR_date2_1_tech2_gold[:,11])),
                                                        float((summaryCDR_date2_2_tech2_gold[:,11])),
                                                        float((summaryCDR_date2_3_tech2_gold[:,11])),
                                                        float((summaryCDR_date2_1_tech2_silver[:,11])),
                                                        float((summaryCDR_date2_2_tech2_silver[:,11])),
                                                        float((summaryCDR_date2_3_tech2_silver[:,11])),
                                                        float((summaryCDR_date2_1_tech2_bronze[:,11])),
                                                        float((summaryCDR_date2_2_tech2_bronze[:,11])),
                                                        float((summaryCDR_date2_3_tech2_bronze[:,11])),
                                                        float((summaryCDR_date3_1_gold[:,11])),
                                                        float((summaryCDR_date3_2_gold[:,11])),
                                                        float((summaryCDR_date3_3_gold[:,11])),
                                                        float((summaryCDR_date3_1_silver[:,11])),
                                                        float((summaryCDR_date3_2_silver[:,11])),
                                                        float((summaryCDR_date3_3_silver[:,11])),
                                                        float((summaryCDR_date3_1_bronze[:,11])),
                                                        float((summaryCDR_date3_2_bronze[:,11])),
                                                        float((summaryCDR_date3_3_bronze[:,11])),
                                                        float((summaryCDR_date3_1_tech1_gold[:,11])),
                                                        float((summaryCDR_date3_2_tech1_gold[:,11])),
                                                        float((summaryCDR_date3_3_tech1_gold[:,11])),
                                                        float((summaryCDR_date3_1_tech1_silver[:,11])),
                                                        float((summaryCDR_date3_2_tech1_silver[:,11])),
                                                        float((summaryCDR_date3_3_tech1_silver[:,11])),
                                                        float((summaryCDR_date3_1_tech1_bronze[:,11])),
                                                        float((summaryCDR_date3_2_tech1_bronze[:,11])),
                                                        float((summaryCDR_date3_3_tech1_bronze[:,11])),
                                                        float((summaryCDR_date3_1_tech2_gold[:,11])),
                                                        float((summaryCDR_date3_2_tech2_gold[:,11])),
                                                        float((summaryCDR_date3_3_tech2_gold[:,11])),
                                                        float((summaryCDR_date3_1_tech2_silver[:,11])),
                                                        float((summaryCDR_date3_2_tech2_silver[:,11])),
                                                        float((summaryCDR_date3_3_tech2_silver[:,11])),
                                                        float((summaryCDR_date3_1_tech2_bronze[:,11])),
                                                        float((summaryCDR_date3_2_tech2_bronze[:,11])),
                                                        float((summaryCDR_date3_3_tech2_bronze[:,11])),
                                                        ],
                                    "summary_Tot_CDR_std": [float((summaryCDR_date1_std_1_gold[:,11])),
                                                            float((summaryCDR_date1_std_2_gold[:,11])),
                                                            float((summaryCDR_date1_std_3_gold[:,11])),
                                                            float((summaryCDR_date1_std_1_silver[:,11])),
                                                            float((summaryCDR_date1_std_2_silver[:,11])),
                                                            float((summaryCDR_date1_std_3_silver[:,11])),
                                                            float((summaryCDR_date1_std_1_bronze[:,11])),
                                                            float((summaryCDR_date1_std_2_bronze[:,11])),
                                                            float((summaryCDR_date1_std_3_bronze[:,11])),
                                                            float((summaryCDR_date1_std_1_tech1_gold[:,11])),
                                                            float((summaryCDR_date1_std_2_tech1_gold[:,11])),
                                                            float((summaryCDR_date1_std_3_tech1_gold[:,11])),
                                                            float((summaryCDR_date1_std_1_tech1_silver[:,11])),
                                                            float((summaryCDR_date1_std_2_tech1_silver[:,11])),
                                                            float((summaryCDR_date1_std_3_tech1_silver[:,11])),
                                                            float((summaryCDR_date1_std_1_tech1_bronze[:,11])),
                                                            float((summaryCDR_date1_std_2_tech1_bronze[:,11])),
                                                            float((summaryCDR_date1_std_3_tech1_bronze[:,11])),
                                                            float((summaryCDR_date1_std_1_tech2_gold[:,11])),
                                                            float((summaryCDR_date1_std_2_tech2_gold[:,11])),
                                                            float((summaryCDR_date1_std_3_tech2_gold[:,11])),
                                                            float((summaryCDR_date1_std_1_tech2_silver[:,11])),
                                                            float((summaryCDR_date1_std_2_tech2_silver[:,11])),
                                                            float((summaryCDR_date1_std_3_tech2_silver[:,11])),
                                                            float((summaryCDR_date1_std_1_tech2_bronze[:,11])),
                                                            float((summaryCDR_date1_std_2_tech2_bronze[:,11])),
                                                            float((summaryCDR_date1_std_3_tech2_bronze[:,11])),
                                                            float((summaryCDR_date2_std_1_gold[:,11])),
                                                            float((summaryCDR_date2_std_2_gold[:,11])),
                                                            float((summaryCDR_date2_std_3_gold[:,11])),
                                                            float((summaryCDR_date2_std_1_silver[:,11])),
                                                            float((summaryCDR_date2_std_2_silver[:,11])),
                                                            float((summaryCDR_date2_std_3_silver[:,11])),
                                                            float((summaryCDR_date2_std_1_bronze[:,11])),
                                                            float((summaryCDR_date2_std_2_bronze[:,11])),
                                                            float((summaryCDR_date2_std_3_bronze[:,11])),
                                                            float((summaryCDR_date2_std_1_tech1_gold[:,11])),
                                                            float((summaryCDR_date2_std_2_tech1_gold[:,11])),
                                                            float((summaryCDR_date2_std_3_tech1_gold[:,11])),
                                                            float((summaryCDR_date2_std_1_tech1_silver[:,11])),
                                                            float((summaryCDR_date2_std_2_tech1_silver[:,11])),
                                                            float((summaryCDR_date2_std_3_tech1_silver[:,11])),
                                                            float((summaryCDR_date2_std_1_tech1_bronze[:,11])),
                                                            float((summaryCDR_date2_std_2_tech1_bronze[:,11])),
                                                            float((summaryCDR_date2_std_3_tech1_bronze[:,11])),
                                                            float((summaryCDR_date2_std_1_tech2_gold[:,11])),
                                                            float((summaryCDR_date2_std_2_tech2_gold[:,11])),
                                                            float((summaryCDR_date2_std_3_tech2_gold[:,11])),
                                                            float((summaryCDR_date2_std_1_tech2_silver[:,11])),
                                                            float((summaryCDR_date2_std_2_tech2_silver[:,11])),
                                                            float((summaryCDR_date2_std_3_tech2_silver[:,11])),
                                                            float((summaryCDR_date2_std_1_tech2_bronze[:,11])),
                                                            float((summaryCDR_date2_std_2_tech2_bronze[:,11])),
                                                            float((summaryCDR_date2_std_3_tech2_bronze[:,11])),
                                                            float((summaryCDR_date3_std_1_gold[:,11])),
                                                            float((summaryCDR_date3_std_2_gold[:,11])),
                                                            float((summaryCDR_date3_std_3_gold[:,11])),
                                                            float((summaryCDR_date3_std_1_silver[:,11])),
                                                            float((summaryCDR_date3_std_2_silver[:,11])),
                                                            float((summaryCDR_date3_std_3_silver[:,11])),
                                                            float((summaryCDR_date3_std_1_bronze[:,11])),
                                                            float((summaryCDR_date3_std_2_bronze[:,11])),
                                                            float((summaryCDR_date3_std_3_bronze[:,11])),
                                                            float((summaryCDR_date3_std_1_tech1_gold[:,11])),
                                                            float((summaryCDR_date3_std_2_tech1_gold[:,11])),
                                                            float((summaryCDR_date3_std_3_tech1_gold[:,11])),
                                                            float((summaryCDR_date3_std_1_tech1_silver[:,11])),
                                                            float((summaryCDR_date3_std_2_tech1_silver[:,11])),
                                                            float((summaryCDR_date3_std_3_tech1_silver[:,11])),
                                                            float((summaryCDR_date3_std_1_tech1_bronze[:,11])),
                                                            float((summaryCDR_date3_std_2_tech1_bronze[:,11])),
                                                            float((summaryCDR_date3_std_3_tech1_bronze[:,11])),
                                                            float((summaryCDR_date3_std_1_tech2_gold[:,11])),
                                                            float((summaryCDR_date3_std_2_tech2_gold[:,11])),
                                                            float((summaryCDR_date3_std_3_tech2_gold[:,11])),
                                                            float((summaryCDR_date3_std_1_tech2_silver[:,11])),
                                                            float((summaryCDR_date3_std_2_tech2_silver[:,11])),
                                                            float((summaryCDR_date3_std_3_tech2_silver[:,11])),
                                                            float((summaryCDR_date3_std_1_tech2_bronze[:,11])),
                                                            float((summaryCDR_date3_std_2_tech2_bronze[:,11])),
                                                            float((summaryCDR_date3_std_3_tech2_bronze[:,11]))],
                                    "Year": [date1, date1, date1, date1, date1, date1, date1, date1, date1,
                                             date1, date1, date1, date1, date1, date1, date1, date1, date1,
                                             date1, date1, date1, date1, date1, date1, date1, date1, date1,
                                             date2, date2, date2, date2, date2, date2, date2, date2, date2,
                                             date2, date2, date2, date2, date2, date2, date2, date2, date2,
                                             date2, date2, date2, date2, date2, date2, date2, date2, date2,
                                             date3, date3, date3, date3, date3, date3, date3, date3, date3,
                                             date3, date3, date3, date3, date3, date3, date3, date3, date3,
                                             date3, date3, date3, date3, date3, date3, date3, date3, date3,],
                                    "Scenario": [scenario1, scenario2, scenario3, scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3, scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3, scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3, scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3, scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3, scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3, scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3, scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3, scenario1, scenario2, scenario3,
                                                 scenario1, scenario2, scenario3
                                                 ],
                                    "Technology": ["Jet A1", "Jet A1", "Jet A1", "Jet A1", "Jet A1", "Jet A1", "Jet A1",
                                                   "Jet A1", "Jet A1",
                                                   tech1, tech1, tech1, tech1, tech1, tech1, tech1, tech1, tech1,
                                                   tech2, tech2, tech2, tech2, tech2, tech2, tech2, tech2, tech2,
                                                   "Jet A1", "Jet A1", "Jet A1", "Jet A1", "Jet A1", "Jet A1", "Jet A1",
                                                   "Jet A1", "Jet A1",
                                                   tech1, tech1, tech1, tech1, tech1, tech1, tech1, tech1, tech1,
                                                   tech2, tech2, tech2, tech2, tech2, tech2, tech2, tech2, tech2,
                                                   "Jet A1", "Jet A1", "Jet A1", "Jet A1", "Jet A1", "Jet A1", "Jet A1",
                                                   "Jet A1", "Jet A1",
                                                   tech1, tech1, tech1, tech1, tech1, tech1, tech1, tech1, tech1,
                                                   tech2, tech2, tech2, tech2, tech2, tech2, tech2, tech2, tech2,
                                                   ],
                                    "Climate neutrality": [baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2,
                                                           baseline_3, baseline_3, baseline_3,
                                                           baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2,
                                                           baseline_3, baseline_3, baseline_3,
                                                           baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2,
                                                           baseline_3, baseline_3, baseline_3,
                                                           baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2,
                                                           baseline_3, baseline_3, baseline_3,
                                                           baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2,
                                                           baseline_3, baseline_3, baseline_3,
                                                           baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2,
                                                           baseline_3, baseline_3, baseline_3,
                                                           baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2,
                                                           baseline_3, baseline_3, baseline_3,
                                                           baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2,
                                                           baseline_3, baseline_3, baseline_3,
                                                           baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2,
                                                           baseline_3, baseline_3, baseline_3,
                                                           ]})

    elif what == 'cumulative_E':
        CDR_summary = pd.DataFrame({"summary_CO2": [float((summaryCDR_1_gold[0])),
                                                    float((summaryCDR_2_gold[0])),
                                                    float((summaryCDR_3_gold[0])),
                                                    float((summaryCDR_1_silver[0])),
                                                    float((summaryCDR_2_silver[0])),
                                                    float((summaryCDR_3_silver[0])),
                                                    float((summaryCDR_1_bronze[0])),
                                                    float((summaryCDR_2_bronze[0])),
                                                    float((summaryCDR_3_bronze[0])),
                                                    float((summaryCDR_1_tech1_gold[0])),
                                                    float((summaryCDR_2_tech1_gold[0])),
                                                    float((summaryCDR_3_tech1_gold[0])),
                                                    float((summaryCDR_1_tech1_silver[0])),
                                                    float((summaryCDR_2_tech1_silver[0])),
                                                    float((summaryCDR_3_tech1_silver[0])),
                                                    float((summaryCDR_1_tech1_bronze[0])),
                                                    float((summaryCDR_2_tech1_bronze[0])),
                                                    float((summaryCDR_3_tech1_bronze[0])),
                                                        float((summaryCDR_1_tech2_gold[0])),
                                                        float((summaryCDR_2_tech2_gold[0])),
                                                        float((summaryCDR_3_tech2_gold[0])),
                                                        float((summaryCDR_1_tech2_silver[0])),
                                                        float((summaryCDR_2_tech2_silver[0])),
                                                        float((summaryCDR_3_tech2_silver[0])),
                                                        float((summaryCDR_1_tech2_bronze[0])),
                                                        float((summaryCDR_2_tech2_bronze[0])),
                                                        float((summaryCDR_3_tech2_bronze[0])),
                                                        ],
                                    "summary_CO2_std": [float((summaryCDR_std_1_gold[0])),
                                                    float((summaryCDR_std_2_gold[0])),
                                                    float((summaryCDR_std_3_gold[0])),
                                                    float((summaryCDR_std_1_silver[0])),
                                                    float((summaryCDR_std_2_silver[0])),
                                                    float((summaryCDR_std_3_silver[0])),
                                                    float((summaryCDR_std_1_bronze[0])),
                                                    float((summaryCDR_std_2_bronze[0])),
                                                    float((summaryCDR_std_3_bronze[0])),
                                                    float((summaryCDR_std_1_tech1_gold[0])),
                                                    float((summaryCDR_std_2_tech1_gold[0])),
                                                    float((summaryCDR_std_3_tech1_gold[0])),
                                                    float((summaryCDR_std_1_tech1_silver[0])),
                                                    float((summaryCDR_std_2_tech1_silver[0])),
                                                    float((summaryCDR_std_3_tech1_silver[0])),
                                                    float((summaryCDR_std_1_tech1_bronze[0])),
                                                    float((summaryCDR_std_2_tech1_bronze[0])),
                                                    float((summaryCDR_std_3_tech1_bronze[0])),
                                                        float((summaryCDR_std_1_tech2_gold[0])),
                                                        float((summaryCDR_std_2_tech2_gold[0])),
                                                        float((summaryCDR_std_3_tech2_gold[0])),
                                                        float((summaryCDR_std_1_tech2_silver[0])),
                                                        float((summaryCDR_std_2_tech2_silver[0])),
                                                        float((summaryCDR_std_3_tech2_silver[0])),
                                                        float((summaryCDR_std_1_tech2_bronze[0])),
                                                        float((summaryCDR_std_2_tech2_bronze[0])),
                                                        float((summaryCDR_std_3_tech2_bronze[0]))],
                        "Scenario": [scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3,
                                     scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3,
                                     scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3],
                        "Technology": ["Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1",
                                       tech1,tech1,tech1,tech1,tech1,tech1,tech1,tech1,tech1,
                                       tech2,tech2,tech2,tech2,tech2,tech2,tech2,tech2,tech2],
                        "Climate neutrality": [baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2, baseline_3, baseline_3, baseline_3,
                                               baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2, baseline_3, baseline_3, baseline_3,
                                               baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2, baseline_3, baseline_3, baseline_3]})

    else:
        CDR_summary = pd.DataFrame({"summary_CO2_CDR": [float((summaryCDR_1_gold[9])),
                                                    float((summaryCDR_2_gold[9])),
                                                    float((summaryCDR_3_gold[9])),
                                                    float((summaryCDR_1_silver[9])),
                                                    float((summaryCDR_2_silver[9])),
                                                    float((summaryCDR_3_silver[9])),
                                                    float((summaryCDR_1_bronze[9])),
                                                    float((summaryCDR_2_bronze[9])),
                                                    float((summaryCDR_3_bronze[9])),
                                                    float((summaryCDR_1_tech1_gold[9])),
                                                    float((summaryCDR_2_tech1_gold[9])),
                                                    float((summaryCDR_3_tech1_gold[9])),
                                                    float((summaryCDR_1_tech1_silver[9])),
                                                    float((summaryCDR_2_tech1_silver[9])),
                                                    float((summaryCDR_3_tech1_silver[9])),
                                                    float((summaryCDR_1_tech1_bronze[9])),
                                                    float((summaryCDR_2_tech1_bronze[9])),
                                                    float((summaryCDR_3_tech1_bronze[9])),
                                                        float((summaryCDR_1_tech2_gold[9])),
                                                        float((summaryCDR_2_tech2_gold[9])),
                                                        float((summaryCDR_3_tech2_gold[9])),
                                                        float((summaryCDR_1_tech2_silver[9])),
                                                        float((summaryCDR_2_tech2_silver[9])),
                                                        float((summaryCDR_3_tech2_silver[9])),
                                                        float((summaryCDR_1_tech2_bronze[9])),
                                                        float((summaryCDR_2_tech2_bronze[9])),
                                                        float((summaryCDR_3_tech2_bronze[9])),
                                                        ],
                                    "summary_CO2_CDR_std": [float((summaryCDR_std_1_gold[9])),
                                                    float((summaryCDR_std_2_gold[9])),
                                                    float((summaryCDR_std_3_gold[9])),
                                                    float((summaryCDR_std_1_silver[9])),
                                                    float((summaryCDR_std_2_silver[9])),
                                                    float((summaryCDR_std_3_silver[9])),
                                                    float((summaryCDR_std_1_bronze[9])),
                                                    float((summaryCDR_std_2_bronze[9])),
                                                    float((summaryCDR_std_3_bronze[9])),
                                                    float((summaryCDR_std_1_tech1_gold[9])),
                                                    float((summaryCDR_std_2_tech1_gold[9])),
                                                    float((summaryCDR_std_3_tech1_gold[9])),
                                                    float((summaryCDR_std_1_tech1_silver[9])),
                                                    float((summaryCDR_std_2_tech1_silver[9])),
                                                    float((summaryCDR_std_3_tech1_silver[9])),
                                                    float((summaryCDR_std_1_tech1_bronze[9])),
                                                    float((summaryCDR_std_2_tech1_bronze[9])),
                                                    float((summaryCDR_std_3_tech1_bronze[9])),
                                                        float((summaryCDR_std_1_tech2_gold[9])),
                                                        float((summaryCDR_std_2_tech2_gold[9])),
                                                        float((summaryCDR_std_3_tech2_gold[9])),
                                                        float((summaryCDR_std_1_tech2_silver[9])),
                                                        float((summaryCDR_std_2_tech2_silver[9])),
                                                        float((summaryCDR_std_3_tech2_silver[9])),
                                                        float((summaryCDR_std_1_tech2_bronze[9])),
                                                        float((summaryCDR_std_2_tech2_bronze[9])),
                                                        float((summaryCDR_std_3_tech2_bronze[9]))],
                                    "summary_Tot_CDR": [float((summaryCDR_1_gold[11])),
                                                    float((summaryCDR_2_gold[11])),
                                                    float((summaryCDR_3_gold[11])),
                                                    float((summaryCDR_1_silver[11])),
                                                    float((summaryCDR_2_silver[11])),
                                                    float((summaryCDR_3_silver[11])),
                                                    float((summaryCDR_1_bronze[11])),
                                                    float((summaryCDR_2_bronze[11])),
                                                    float((summaryCDR_3_bronze[11])),
                                                    float((summaryCDR_1_tech1_gold[11])),
                                                    float((summaryCDR_2_tech1_gold[11])),
                                                    float((summaryCDR_3_tech1_gold[11])),
                                                    float((summaryCDR_1_tech1_silver[11])),
                                                    float((summaryCDR_2_tech1_silver[11])),
                                                    float((summaryCDR_3_tech1_silver[11])),
                                                    float((summaryCDR_1_tech1_bronze[11])),
                                                    float((summaryCDR_2_tech1_bronze[11])),
                                                    float((summaryCDR_3_tech1_bronze[11])),
                                                        float((summaryCDR_1_tech2_gold[11])),
                                                        float((summaryCDR_2_tech2_gold[11])),
                                                        float((summaryCDR_3_tech2_gold[11])),
                                                        float((summaryCDR_1_tech2_silver[11])),
                                                        float((summaryCDR_2_tech2_silver[11])),
                                                        float((summaryCDR_3_tech2_silver[11])),
                                                        float((summaryCDR_1_tech2_bronze[11])),
                                                        float((summaryCDR_2_tech2_bronze[11])),
                                                        float((summaryCDR_3_tech2_bronze[11])),],
                                    "summary_Tot_CDR_std": [float((summaryCDR_std_1_gold[11])),
                                                    float((summaryCDR_std_2_gold[11])),
                                                    float((summaryCDR_std_3_gold[11])),
                                                    float((summaryCDR_std_1_silver[11])),
                                                    float((summaryCDR_std_2_silver[11])),
                                                    float((summaryCDR_std_3_silver[11])),
                                                    float((summaryCDR_std_1_bronze[11])),
                                                    float((summaryCDR_std_2_bronze[11])),
                                                    float((summaryCDR_std_3_bronze[11])),
                                                    float((summaryCDR_std_1_tech1_gold[11])),
                                                    float((summaryCDR_std_2_tech1_gold[11])),
                                                    float((summaryCDR_std_3_tech1_gold[11])),
                                                    float((summaryCDR_std_1_tech1_silver[11])),
                                                    float((summaryCDR_std_2_tech1_silver[11])),
                                                    float((summaryCDR_std_3_tech1_silver[11])),
                                                    float((summaryCDR_std_1_tech1_bronze[11])),
                                                    float((summaryCDR_std_2_tech1_bronze[11])),
                                                    float((summaryCDR_std_3_tech1_bronze[11])),
                                                        float((summaryCDR_std_1_tech2_gold[11])),
                                                        float((summaryCDR_std_2_tech2_gold[11])),
                                                        float((summaryCDR_std_3_tech2_gold[11])),
                                                        float((summaryCDR_std_1_tech2_silver[11])),
                                                        float((summaryCDR_std_2_tech2_silver[11])),
                                                        float((summaryCDR_std_3_tech2_silver[11])),
                                                        float((summaryCDR_std_1_tech2_bronze[11])),
                                                        float((summaryCDR_std_2_tech2_bronze[11])),
                                                        float((summaryCDR_std_3_tech2_bronze[11]))],
                        "Scenario": [scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3,
                                     scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3,
                                     scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3],
                        "Technology": ["Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1",
                                       tech1,tech1,tech1,tech1,tech1,tech1,tech1,tech1,tech1,
                                       tech2,tech2,tech2,tech2,tech2,tech2,tech2,tech2,tech2],
                        "Climate neutrality": [baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2, baseline_3, baseline_3, baseline_3,
                                               baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2, baseline_3, baseline_3, baseline_3,
                                               baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2, baseline_3, baseline_3, baseline_3]})
    return CDR_summary

def make_bar_EWF_alltechs_summary(ERF_1, ERF_2, ERF_3,
                         ERF_1_tech1, ERF_2_tech1, ERF_3_tech1,
                         ERF_1_tech2, ERF_2_tech2, ERF_3_tech2,
                                   E_1, E_2, E_3,
                                   E_1_tech1, E_2_tech1, E_3_tech1,
                                   E_1_tech2, E_2_tech2, E_3_tech2,
                         scenario1 = 'BAU',scenario2 = 'Air Pollution',scenario3 = 'Mitigation',
                         tech1 = TECH_1, tech2 = TECH_2):
    """
    make summary dataframe to produce EWF analysis (outputting tot CO2-eq emissions / CO2 emissions=
    :param tech1: technology 1, default : Zero-CO$_2$ fuels
    :param tech2: technology 2, default : E-airplanes
    :return: summary dataframe
    """
    E_CO2_1 = pd.to_numeric(E_1['CO2']).values
    E_CO2_2 = pd.to_numeric(E_2['CO2']).values
    E_CO2_3 = pd.to_numeric(E_3['CO2']).values
    E_CO2_tech1_1 = pd.to_numeric(E_1_tech1.best['CO2']).values
    E_CO2_tech1_2 =  pd.to_numeric(E_2_tech1.best['CO2']).values
    E_CO2_tech1_3 =  pd.to_numeric(E_3_tech1.best['CO2']).values
    E_CO2_tech2_1 = pd.to_numeric(E_1_tech2.best['CO2']).values
    E_CO2_tech2_2 = pd.to_numeric(E_2_tech2.best['CO2']).values
    E_CO2_tech2_3 = pd.to_numeric(E_3_tech2.best['CO2']).values
    E_GWPstar_1 = calculate_GWPstar_CDR(ERF_1['non-CO2'], 20, start_date = 2010, init_date = 1990)
    E_GWPstar_2 = calculate_GWPstar_CDR(ERF_2['non-CO2'], 20, start_date = 2010, init_date = 1990)
    E_GWPstar_3 = calculate_GWPstar_CDR(ERF_3['non-CO2'], 20, start_date=2010, init_date=1990)
    E_GWPstar_tech1_1 = calculate_GWPstar_CDR(ERF_1_tech1['non-CO2'], 20, start_date = 2010, init_date = 1990)
    E_GWPstar_tech1_2 = calculate_GWPstar_CDR(ERF_2_tech1['non-CO2'], 20, start_date = 2010, init_date = 1990)
    E_GWPstar_tech1_3 = calculate_GWPstar_CDR(ERF_3_tech1['non-CO2'], 20, start_date=2010, init_date=1990)
    E_GWPstar_tech2_1 = calculate_GWPstar_CDR(ERF_1_tech2['non-CO2'], 20, start_date = 2010, init_date = 1990)
    E_GWPstar_tech2_2 = calculate_GWPstar_CDR(ERF_2_tech2['non-CO2'], 20, start_date = 2010, init_date = 1990)
    E_GWPstar_tech2_3 = calculate_GWPstar_CDR(ERF_3_tech2['non-CO2'], 20, start_date=2010, init_date=1990)

    EWF_1 = (np.sum(E_GWPstar_1)+np.sum(E_CO2_1))/np.sum(E_CO2_1)
    EWF_2 = (np.sum(E_GWPstar_2) +np.sum(E_CO2_2)) / np.sum(E_CO2_2)
    EWF_3 = (np.sum(E_GWPstar_3) + np.sum(E_CO2_3)) / np.sum(E_CO2_3)
    EWF_tech1_1 = (np.sum(E_GWPstar_tech1_1) + np.sum(E_CO2_tech1_1))/np.sum(E_CO2_tech1_1)
    EWF_tech1_2 = (np.sum(E_GWPstar_tech1_2) + np.sum(E_CO2_tech1_2))/ np.sum(E_CO2_tech1_2)
    EWF_tech1_3 = (np.sum(E_GWPstar_tech1_3) + np.sum(E_CO2_tech1_3)) / np.sum(E_CO2_tech1_3)
    EWF_tech2_1 = (np.sum(E_GWPstar_tech2_1) + np.sum(E_CO2_tech2_1))/np.sum(E_CO2_tech2_1)
    EWF_tech2_2 = (np.sum(E_GWPstar_tech2_2) + np.sum(E_CO2_tech2_2)) / np.sum(E_CO2_tech2_2)
    EWF_tech2_3 = (np.sum(E_GWPstar_tech2_3) + np.sum(E_CO2_tech2_3))/ np.sum(E_CO2_tech2_3)


    summary = pd.DataFrame({"EWF": [float(unumpy.nominal_values(EWF_1)),
                                                float(unumpy.nominal_values(EWF_2)),
                                                float(unumpy.nominal_values(EWF_3)),
                                                float(unumpy.nominal_values(EWF_tech1_1)),
                                                float(unumpy.nominal_values(EWF_tech1_2)),
                                                float(unumpy.nominal_values(EWF_tech1_3)),
                                    float(unumpy.nominal_values(EWF_tech2_1)),
                                    float(unumpy.nominal_values(EWF_tech2_2)),
                                    float(unumpy.nominal_values(EWF_tech2_3)),
                                                    ],
                                "EWF_std": [float(unumpy.std_devs(EWF_1)),
                                                float(unumpy.std_devs(EWF_2)),
                                                float(unumpy.std_devs(EWF_3)),
                                                float(unumpy.std_devs(EWF_tech1_1)),
                                                float(unumpy.std_devs(EWF_tech1_2)),
                                                float(unumpy.std_devs(EWF_tech1_3)),
                                    float(unumpy.std_devs(EWF_tech2_1)),
                                    float(unumpy.std_devs(EWF_tech2_2)),
                                    float(unumpy.std_devs(EWF_tech2_3)),
                                                    ],
                                 "Scenario": [scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3],
                    "Technology": ["Jet A1","Jet A1","Jet A1",tech1,tech1,tech1, tech2,tech2,tech2,],})
    #fair_summary = pd.DataFrame(index = pd.to_datetime([date1, date2, date3], format='%Y'), data= np.stack((fair_date1, fair_date2, fair_date3)), columns = fair.columns)
    return summary


def make_bar_fair_alltechs_summary(fair_1_gold, fair_2_gold, fair_3_gold,
                         fair_1_silver, fair_2_silver, fair_3_silver,
                         fair_1_bronze, fair_2_bronze, fair_3_bronze,
                         fair_1_tech1_gold, fair_2_tech1_gold, fair_3_tech1_gold,
                         fair_1_tech1_silver, fair_2_tech1_silver, fair_3_tech1_silver,
                         fair_1_tech1_bronze, fair_2_tech1_bronze, fair_3_tech1_bronze,
                         fair_1_tech2_gold, fair_2_tech2_gold, fair_3_tech2_gold,
                         fair_1_tech2_silver, fair_2_tech2_silver, fair_3_tech2_silver,
                         fair_1_tech2_bronze, fair_2_tech2_bronze, fair_3_tech2_bronze,
                         scenario1 = 'BAU',scenario2 = 'Air Pollution',scenario3 = 'Mitigation',
                         tech1 = TECH_1, tech2 = TECH_2, what = 'ERF'):
    """
    Makes summary dataframe for climatic outputs (T or ERF)
    :param what: what to output, T (temperatures) or ERF
    :return: summary dataframe
    """
    if what == 'ERF':
        aviation_1 = fair_1_gold.F_aviation[-1]
        aviation_2 = fair_2_gold.F_aviation[-1]
        aviation_3 = fair_3_gold.F_aviation[-1]
        aviation_tech1_1 = fair_1_tech1_gold.F_aviation[-1]
        aviation_tech1_2 = fair_2_tech1_gold.F_aviation[-1]
        aviation_tech1_3 = fair_3_tech1_gold.F_aviation[-1]
        aviation_tech2_1 = fair_1_tech2_gold.F_aviation[-1]
        aviation_tech2_2 = fair_2_tech2_gold.F_aviation[-1]
        aviation_tech2_3 = fair_3_tech2_gold.F_aviation[-1]
        aviation_std_1 = 0.5*(fair_1_gold.F_aviation_upper[-1] - fair_1_gold.F_aviation_lower[-1])
        aviation_std_2 = 0.5*(fair_2_gold.F_aviation_upper[-1] - fair_2_gold.F_aviation_lower[-1])
        aviation_std_3 = 0.5*(fair_3_gold.F_aviation_upper[-1] - fair_3_gold.F_aviation_lower[-1])
        aviation_std_tech1_1 = 0.5*(fair_1_tech1_gold.F_aviation_upper[-1] - fair_1_tech1_gold.F_aviation_lower[
            -1])
        aviation_std_tech1_2 = 0.5*(fair_2_tech1_gold.F_aviation_upper[-1] - fair_2_tech1_gold.F_aviation_lower[
            -1])
        aviation_std_tech1_3 = 0.5*(fair_3_tech1_gold.F_aviation_upper[-1] - fair_3_tech1_gold.F_aviation_lower[
            -1])
        aviation_std_tech2_1 = 0.5*(fair_1_tech2_gold.F_aviation_upper[-1] - fair_1_tech2_gold.F_aviation_lower[
            -1])
        aviation_std_tech2_2 = 0.5*(fair_2_tech2_gold.F_aviation_upper[-1] - fair_2_tech2_gold.F_aviation_lower[
            -1])
        aviation_std_tech2_3 = 0.5*(fair_3_tech2_gold.F_aviation_upper[-1] - fair_3_tech2_gold.F_aviation_lower[
            -1])

        summary_1_gold = fair_1_gold.F_avCDR[-1]
        summary_2_gold = fair_2_gold.F_avCDR[-1]
        summary_3_gold = fair_3_gold.F_avCDR[-1]
        summary_1_silver = fair_1_silver.F_avCDR[-1]
        summary_2_silver = fair_2_silver.F_avCDR[-1]
        summary_3_silver = fair_3_silver.F_avCDR[-1]
        summary_1_bronze = fair_1_bronze.F_avCDR[-1]
        summary_2_bronze = fair_2_bronze.F_avCDR[-1]
        summary_3_bronze = fair_3_bronze.F_avCDR[-1]
        summary_std_1_gold = 0.5*(fair_1_gold.F_avCDR_upper[-1] - fair_1_gold.F_avCDR_lower[-1])
        summary_std_2_gold = 0.5*(fair_2_gold.F_avCDR_upper[-1] - fair_2_gold.F_avCDR_lower[-1])
        summary_std_3_gold = 0.5*(fair_3_gold.F_avCDR_upper[-1] - fair_3_gold.F_avCDR_lower[-1])
        summary_std_1_silver = 0.5*(fair_1_silver.F_avCDR_upper[-1] - fair_1_silver.F_avCDR_lower[-1])
        summary_std_2_silver = 0.5*(fair_2_silver.F_avCDR_upper[-1] - fair_2_silver.F_avCDR_lower[-1])
        summary_std_3_silver = 0.5*(fair_3_silver.F_avCDR_upper[-1] - fair_3_silver.F_avCDR_lower[-1])
        summary_std_1_bronze = 0.5*(fair_1_bronze.F_avCDR_upper[-1] - fair_1_bronze.F_avCDR_lower[-1])
        summary_std_2_bronze = 0.5*(fair_2_bronze.F_avCDR_upper[-1] - fair_2_bronze.F_avCDR_lower[-1])
        summary_std_3_bronze = 0.5*(fair_3_bronze.F_avCDR_upper[-1] - fair_3_bronze.F_avCDR_lower[-1])
        summary_1_tech1_gold = fair_1_tech1_gold.F_avCDR[-1]
        summary_2_tech1_gold = fair_2_tech1_gold.F_avCDR[-1]
        summary_3_tech1_gold = fair_3_tech1_gold.F_avCDR[-1]
        summary_1_tech1_silver = fair_1_tech1_silver.F_avCDR[-1]
        summary_2_tech1_silver = fair_2_tech1_silver.F_avCDR[-1]
        summary_3_tech1_silver = fair_3_tech1_silver.F_avCDR[-1]
        summary_1_tech1_bronze = fair_1_tech1_bronze.F_avCDR[-1]
        summary_2_tech1_bronze = fair_2_tech1_bronze.F_avCDR[-1]
        summary_3_tech1_bronze = fair_3_tech1_bronze.F_avCDR[-1]
        summary_std_1_tech1_gold =  0.5*(fair_1_tech1_gold.F_avCDR_upper[-1] -  fair_1_tech1_gold.F_avCDR_lower[-1])
        summary_std_2_tech1_gold = 0.5*(fair_2_tech1_gold.F_avCDR_upper[-1] -  fair_2_tech1_gold.F_avCDR_lower[-1])
        summary_std_3_tech1_gold = 0.5*(fair_3_tech1_gold.F_avCDR_upper[-1] -  fair_3_tech1_gold.F_avCDR_lower[-1])
        summary_std_1_tech1_silver = 0.5*(fair_1_tech1_silver.F_avCDR_upper[-1] -  fair_1_tech1_silver.F_avCDR_lower[-1])
        summary_std_2_tech1_silver = 0.5*(fair_2_tech1_silver.F_avCDR_upper[-1] -  fair_2_tech1_silver.F_avCDR_lower[-1])
        summary_std_3_tech1_silver = 0.5*(fair_3_tech1_silver.F_avCDR_upper[-1] -  fair_3_tech1_silver.F_avCDR_lower[-1])
        summary_std_1_tech1_bronze = 0.5*(fair_1_tech1_bronze.F_avCDR_upper[-1] -  fair_1_tech1_bronze.F_avCDR_lower[-1])
        summary_std_2_tech1_bronze = 0.5*(fair_2_tech1_bronze.F_avCDR_upper[-1] -  fair_2_tech1_bronze.F_avCDR_lower[-1])
        summary_std_3_tech1_bronze = 0.5*(fair_3_tech1_bronze.F_avCDR_upper[-1] -  fair_3_tech1_bronze.F_avCDR_lower[-1])
        summary_1_tech2_gold = fair_1_tech2_gold.F_avCDR[-1]
        summary_2_tech2_gold = fair_2_tech2_gold.F_avCDR[-1]
        summary_3_tech2_gold = fair_3_tech2_gold.F_avCDR[-1]
        summary_1_tech2_silver = fair_1_tech2_silver.F_avCDR[-1]
        summary_2_tech2_silver = fair_2_tech2_silver.F_avCDR[-1]
        summary_3_tech2_silver = fair_3_tech2_silver.F_avCDR[-1]
        summary_1_tech2_bronze = fair_1_tech2_bronze.F_avCDR[-1]
        summary_2_tech2_bronze = fair_2_tech2_bronze.F_avCDR[-1]
        summary_3_tech2_bronze = fair_3_tech2_bronze.F_avCDR[-1]
        summary_std_1_tech2_gold =  0.5*(fair_1_tech2_gold.F_avCDR_upper[-1] -  fair_1_tech2_gold.F_avCDR_lower[-1])
        summary_std_2_tech2_gold = 0.5*(fair_2_tech2_gold.F_avCDR_upper[-1] -  fair_2_tech2_gold.F_avCDR_lower[-1])
        summary_std_3_tech2_gold = 0.5*(fair_3_tech2_gold.F_avCDR_upper[-1] -  fair_3_tech2_gold.F_avCDR_lower[-1])
        summary_std_1_tech2_silver = 0.5*(fair_1_tech2_silver.F_avCDR_upper[-1] -  fair_1_tech2_silver.F_avCDR_lower[-1])
        summary_std_2_tech2_silver = 0.5*(fair_2_tech2_silver.F_avCDR_upper[-1] -  fair_2_tech2_silver.F_avCDR_lower[-1])
        summary_std_3_tech2_silver = 0.5*(fair_3_tech2_silver.F_avCDR_upper[-1] -  fair_3_tech2_silver.F_avCDR_lower[-1])
        summary_std_1_tech2_bronze = 0.5*(fair_1_tech2_bronze.F_avCDR_upper[-1] -  fair_1_tech2_bronze.F_avCDR_lower[-1])
        summary_std_2_tech2_bronze = 0.5*(fair_2_tech2_bronze.F_avCDR_upper[-1] -  fair_2_tech2_bronze.F_avCDR_lower[-1])
        summary_std_3_tech2_bronze = 0.5*(fair_3_tech2_bronze.F_avCDR_upper[-1] -  fair_3_tech2_bronze.F_avCDR_lower[-1])

    elif what == 'T':
        aviation_1 = fair_1_gold.T_aviation[-1]
        aviation_2 = fair_2_gold.T_aviation[-1]
        aviation_3 = fair_3_gold.T_aviation[-1]
        aviation_tech1_1 = fair_1_tech1_gold.T_aviation[-1]
        aviation_tech1_2 = fair_2_tech1_gold.T_aviation[-1]
        aviation_tech1_3 = fair_3_tech1_gold.T_aviation[-1]
        aviation_tech2_1 = fair_1_tech2_gold.T_aviation[-1]
        aviation_tech2_2 = fair_2_tech2_gold.T_aviation[-1]
        aviation_tech2_3 = fair_3_tech2_gold.T_aviation[-1]
        aviation_std_1 = 0.5*(fair_1_gold.T_aviation_upper[-1] - fair_1_gold.T_aviation_lower[-1])
        aviation_std_2 = 0.5*(fair_2_gold.T_aviation_upper[-1] - fair_2_gold.T_aviation_lower[-1])
        aviation_std_3 = 0.5*(fair_3_gold.T_aviation_upper[-1] - fair_3_gold.T_aviation_lower[-1])
        aviation_std_tech1_1 = 0.5*(fair_1_tech1_gold.T_aviation_upper[-1] - fair_1_tech1_gold.T_aviation_lower[-1])
        aviation_std_tech1_2 = 0.5*(fair_2_tech1_gold.T_aviation_upper[-1] - fair_2_tech1_gold.T_aviation_lower[-1])
        aviation_std_tech1_3 = 0.5*(fair_3_tech1_gold.T_aviation_upper[-1] - fair_3_tech1_gold.T_aviation_lower[-1])
        aviation_std_tech2_1 = 0.5*(fair_1_tech2_gold.T_aviation_upper[-1] - fair_1_tech2_gold.T_aviation_lower[-1])
        aviation_std_tech2_2 = 0.5*(fair_2_tech2_gold.T_aviation_upper[-1] - fair_2_tech2_gold.T_aviation_lower[-1])
        aviation_std_tech2_3 = 0.5*(fair_3_tech2_gold.T_aviation_upper[-1] - fair_3_tech2_gold.T_aviation_lower[-1])
        summary_1_gold = fair_1_gold.T_avCDR[-1]
        summary_2_gold = fair_2_gold.T_avCDR[-1]
        summary_3_gold = fair_3_gold.T_avCDR[-1]
        summary_1_silver = fair_1_silver.T_avCDR[-1]
        summary_2_silver = fair_2_silver.T_avCDR[-1]
        summary_3_silver = fair_3_silver.T_avCDR[-1]
        summary_1_bronze = fair_1_bronze.T_avCDR[-1]
        summary_2_bronze = fair_2_bronze.T_avCDR[-1]
        summary_3_bronze = fair_3_bronze.T_avCDR[-1]
        summary_std_1_gold = 0.5*(fair_1_gold.T_avCDR_upper[-1] - fair_1_gold.T_avCDR_lower[-1])
        summary_std_2_gold = 0.5*(fair_2_gold.T_avCDR_upper[-1] - fair_2_gold.T_avCDR_lower[-1])
        summary_std_3_gold = 0.5*(fair_3_gold.T_avCDR_upper[-1] - fair_3_gold.T_avCDR_lower[-1])
        summary_std_1_silver = 0.5*(fair_1_silver.T_avCDR_upper[-1] - fair_1_silver.T_avCDR_lower[-1])
        summary_std_2_silver = 0.5*(fair_2_silver.T_avCDR_upper[-1] - fair_2_silver.T_avCDR_lower[-1])
        summary_std_3_silver = 0.5*(fair_3_silver.T_avCDR_upper[-1] - fair_3_silver.T_avCDR_lower[-1])
        summary_std_1_bronze = 0.5*(fair_1_bronze.T_avCDR_upper[-1] - fair_1_bronze.T_avCDR_lower[-1])
        summary_std_2_bronze = 0.5*(fair_2_bronze.T_avCDR_upper[-1] - fair_2_bronze.T_avCDR_lower[-1])
        summary_std_3_bronze = 0.5*(fair_3_bronze.T_avCDR_upper[-1] - fair_3_bronze.T_avCDR_lower[-1])
        summary_1_tech1_gold = fair_1_tech1_gold.T_avCDR[-1]
        summary_2_tech1_gold = fair_2_tech1_gold.T_avCDR[-1]
        summary_3_tech1_gold = fair_3_tech1_gold.T_avCDR[-1]
        summary_1_tech1_silver = fair_1_tech1_silver.T_avCDR[-1]
        summary_2_tech1_silver = fair_2_tech1_silver.T_avCDR[-1]
        summary_3_tech1_silver = fair_3_tech1_silver.T_avCDR[-1]
        summary_1_tech1_bronze = fair_1_tech1_bronze.T_avCDR[-1]
        summary_2_tech1_bronze = fair_2_tech1_bronze.T_avCDR[-1]
        summary_3_tech1_bronze = fair_3_tech1_bronze.T_avCDR[-1]
        summary_std_1_tech1_gold =  0.5*(fair_1_tech1_gold.T_avCDR_upper[-1] -  fair_1_tech1_gold.T_avCDR_lower[-1])
        summary_std_2_tech1_gold = 0.5*(fair_2_tech1_gold.T_avCDR_upper[-1] -  fair_2_tech1_gold.T_avCDR_lower[-1])
        summary_std_3_tech1_gold = 0.5*(fair_3_tech1_gold.T_avCDR_upper[-1] -  fair_3_tech1_gold.T_avCDR_lower[-1])
        summary_std_1_tech1_silver = 0.5*(fair_1_tech1_silver.T_avCDR_upper[-1] -  fair_1_tech1_silver.T_avCDR_lower[-1])
        summary_std_2_tech1_silver = 0.5*(fair_2_tech1_silver.T_avCDR_upper[-1] -  fair_2_tech1_silver.T_avCDR_lower[-1])
        summary_std_3_tech1_silver = 0.5*(fair_3_tech1_silver.T_avCDR_upper[-1] -  fair_3_tech1_silver.T_avCDR_lower[-1])
        summary_std_1_tech1_bronze = 0.5*(fair_1_tech1_bronze.T_avCDR_upper[-1] -  fair_1_tech1_bronze.T_avCDR_lower[-1])
        summary_std_2_tech1_bronze = 0.5*(fair_2_tech1_bronze.T_avCDR_upper[-1] -  fair_2_tech1_bronze.T_avCDR_lower[-1])
        summary_std_3_tech1_bronze = 0.5*(fair_3_tech1_bronze.T_avCDR_upper[-1] -  fair_3_tech1_bronze.T_avCDR_lower[-1])
        summary_1_tech2_gold = fair_1_tech2_gold.T_avCDR[-1]
        summary_2_tech2_gold = fair_2_tech2_gold.T_avCDR[-1]
        summary_3_tech2_gold = fair_3_tech2_gold.T_avCDR[-1]
        summary_1_tech2_silver = fair_1_tech2_silver.T_avCDR[-1]
        summary_2_tech2_silver = fair_2_tech2_silver.T_avCDR[-1]
        summary_3_tech2_silver = fair_3_tech2_silver.T_avCDR[-1]
        summary_1_tech2_bronze = fair_1_tech2_bronze.T_avCDR[-1]
        summary_2_tech2_bronze = fair_2_tech2_bronze.T_avCDR[-1]
        summary_3_tech2_bronze = fair_3_tech2_bronze.T_avCDR[-1]
        summary_std_1_tech2_gold =  0.5*(fair_1_tech2_gold.T_avCDR_upper[-1] -  fair_1_tech2_gold.T_avCDR_lower[-1])
        summary_std_2_tech2_gold = 0.5*(fair_2_tech2_gold.T_avCDR_upper[-1] -  fair_2_tech2_gold.T_avCDR_lower[-1])
        summary_std_3_tech2_gold = 0.5*(fair_3_tech2_gold.T_avCDR_upper[-1] -  fair_3_tech2_gold.T_avCDR_lower[-1])
        summary_std_1_tech2_silver = 0.5*(fair_1_tech2_silver.T_avCDR_upper[-1] -  fair_1_tech2_silver.T_avCDR_lower[-1])
        summary_std_2_tech2_silver = 0.5*(fair_2_tech2_silver.T_avCDR_upper[-1] -  fair_2_tech2_silver.T_avCDR_lower[-1])
        summary_std_3_tech2_silver = 0.5*(fair_3_tech2_silver.T_avCDR_upper[-1] -  fair_3_tech2_silver.T_avCDR_lower[-1])
        summary_std_1_tech2_bronze = 0.5*(fair_1_tech2_bronze.T_avCDR_upper[-1] -  fair_1_tech2_bronze.T_avCDR_lower[-1])
        summary_std_2_tech2_bronze = 0.5*(fair_2_tech2_bronze.T_avCDR_upper[-1] -  fair_2_tech2_bronze.T_avCDR_lower[-1])
        summary_std_3_tech2_bronze = 0.5*(fair_3_tech2_bronze.T_avCDR_upper[-1] -  fair_3_tech2_bronze.T_avCDR_lower[-1])

    fair_summary = pd.DataFrame({"summary_fair": [float((summary_1_gold)),
                                                float((summary_2_gold)),
                                                float((summary_3_gold)),
                                                float((summary_1_silver)),
                                                float((summary_2_silver)),
                                                float((summary_3_silver)),
                                                float((summary_1_bronze)),
                                                float((summary_2_bronze)),
                                                float((summary_3_bronze)),
                                                float((summary_1_tech1_gold)),
                                                float((summary_2_tech1_gold)),
                                                float((summary_3_tech1_gold)),
                                                float((summary_1_tech1_silver)),
                                                float((summary_2_tech1_silver)),
                                                float((summary_3_tech1_silver)),
                                                float((summary_1_tech1_bronze)),
                                                float((summary_2_tech1_bronze)),
                                                float((summary_3_tech1_bronze)),
                                                    float((summary_1_tech2_gold)),
                                                    float((summary_2_tech2_gold)),
                                                    float((summary_3_tech2_gold)),
                                                    float((summary_1_tech2_silver)),
                                                    float((summary_2_tech2_silver)),
                                                    float((summary_3_tech2_silver)),
                                                    float((summary_1_tech2_bronze)),
                                                    float((summary_2_tech2_bronze)),
                                                    float((summary_3_tech2_bronze)),
                                                    ],
                                "summary_fair_std": [float((summary_std_1_gold)),
                                                float((summary_std_2_gold)),
                                                float((summary_std_3_gold)),
                                                float((summary_std_1_silver)),
                                                float((summary_std_2_silver)),
                                                float((summary_std_3_silver)),
                                                float((summary_std_1_bronze)),
                                                float((summary_std_2_bronze)),
                                                float((summary_std_3_bronze)),
                                                float((summary_std_1_tech1_gold)),
                                                float((summary_std_2_tech1_gold)),
                                                float((summary_std_3_tech1_gold)),
                                                float((summary_std_1_tech1_silver)),
                                                float((summary_std_2_tech1_silver)),
                                                float((summary_std_3_tech1_silver)),
                                                float((summary_std_1_tech1_bronze)),
                                                float((summary_std_2_tech1_bronze)),
                                                float((summary_std_3_tech1_bronze)),
                                                    float((summary_std_1_tech2_gold)),
                                                    float((summary_std_2_tech2_gold)),
                                                    float((summary_std_3_tech2_gold)),
                                                    float((summary_std_1_tech2_silver)),
                                                    float((summary_std_2_tech2_silver)),
                                                    float((summary_std_3_tech2_silver)),
                                                    float((summary_std_1_tech2_bronze)),
                                                    float((summary_std_2_tech2_bronze)),
                                                    float((summary_std_3_tech2_bronze))],
                                "noCDR_fair": [float((aviation_1)),
                                                float((aviation_2)),
                                                float((aviation_3)),
                                                  float((aviation_1)),
                                                  float((aviation_2)),
                                                  float((aviation_3)),
                                                  float((aviation_1)),
                                                  float((aviation_2)),
                                                  float((aviation_3)),
                                                float((aviation_tech1_1)),
                                                float((aviation_tech1_2)),
                                                float((aviation_tech1_3)),
                                                  float((aviation_tech1_1)),
                                                  float((aviation_tech1_2)),
                                                  float((aviation_tech1_3)),
                                                  float((aviation_tech1_1)),
                                                  float((aviation_tech1_2)),
                                                  float((aviation_tech1_3)),
                                                  float((aviation_tech2_1)),
                                                  float((aviation_tech2_2)),
                                                  float((aviation_tech2_3)),
                                                  float((aviation_tech2_1)),
                                                  float((aviation_tech2_2)),
                                                  float((aviation_tech2_3)),
                                                  float((aviation_tech2_1)),
                                                  float((aviation_tech2_2)),
                                                  float((aviation_tech2_3)),
                                                    ],

                                 "noCDR_fair_std": [float((aviation_std_1)),
                                                float((aviation_std_2)),
                                                float((aviation_std_3)),
                                                float((aviation_std_1)),
                                                float((aviation_std_2)),
                                                float((aviation_std_3)),
                                                float((aviation_std_1)),
                                                float((aviation_std_2)),
                                                float((aviation_std_3)),
                                                float((aviation_std_tech1_1)),
                                                float((aviation_std_tech1_2)),
                                                float((aviation_std_tech1_3)),
                                                float((aviation_std_tech1_1)),
                                                float((aviation_std_tech1_2)),
                                                float((aviation_std_tech1_3)),
                                                float((aviation_std_tech1_1)),
                                                float((aviation_std_tech1_2)),
                                                float((aviation_std_tech1_3)),
                                                float((aviation_std_tech2_1)),
                                                float((aviation_std_tech2_2)),
                                                float((aviation_std_tech2_3)),
                                                float((aviation_std_tech2_1)),
                                                float((aviation_std_tech2_2)),
                                                float((aviation_std_tech2_3)),
                                                float((aviation_std_tech2_1)),
                                                float((aviation_std_tech2_2)),
                                                float((aviation_std_tech2_3)),
                                                ],
                                 "Scenario": [scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3,
                                 scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3,
                                 scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3],
                    "Technology": ["Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1",
                                   tech1,tech1,tech1,tech1,tech1,tech1,tech1,tech1,tech1,
                                   tech2,tech2,tech2,tech2,tech2,tech2,tech2,tech2,tech2],
                    "Climate neutrality": [baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2, baseline_3, baseline_3, baseline_3,
                                           baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2, baseline_3, baseline_3, baseline_3,
                                           baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2, baseline_3, baseline_3, baseline_3]})
    #fair_summary = pd.DataFrame(index = pd.to_datetime([date1, date2, date3], format='%Y'), data= np.stack((fair_date1, fair_date2, fair_date3)), columns = fair.columns)
    return fair_summary


def make_summary_T_CO2only(fair_gold_CO2only_A_1, fair_gold_CO2only_A_2, fair_gold_CO2only_A_3,
                           fair_silver_CO2only_A_1, fair_silver_CO2only_A_2, fair_silver_CO2only_A_3,
                           fair_bronze_CO2only_A_1, fair_bronze_CO2only_A_2, fair_bronze_CO2only_A_3,
                           scenario1, scenario2, scenario3, baseline_1 = 'Gold', baseline_2 = 'Silver', baseline_3 = 'Bronze'):
    """
    Make summary dataframe for temperatures due to CO2 emissions only
    :return: summary dataframe
    """
    summary_T_CO2only_df = pd.DataFrame(
        data=[[fair_gold_CO2only_A_1.T_avCDR[-1], fair_silver_CO2only_A_1.T_avCDR[-1], fair_bronze_CO2only_A_1.T_avCDR[-1]],
              [fair_gold_CO2only_A_2.T_avCDR[-1], fair_silver_CO2only_A_2.T_avCDR[-1], fair_bronze_CO2only_A_2.T_avCDR[-1]],
              [fair_gold_CO2only_A_3.T_avCDR[-1], fair_silver_CO2only_A_3.T_avCDR[-1], fair_bronze_CO2only_A_3.T_avCDR[-1]]],
        columns = [baseline_1, baseline_2, baseline_3],
        index=[scenario1, scenario2, scenario3])
    summary_T_CO2only_err_df = pd.DataFrame(
        data=[[(fair_gold_CO2only_A_1.T_avCDR_upper[-1]-fair_gold_CO2only_A_1.T_avCDR_lower[-1])*0.5,
               (fair_silver_CO2only_A_1.T_avCDR_upper[-1] - fair_silver_CO2only_A_1.T_avCDR_lower[-1]) * 0.5,
               (fair_bronze_CO2only_A_1.T_avCDR_upper[-1] - fair_bronze_CO2only_A_1.T_avCDR_lower[-1]) * 0.5],
              [(fair_gold_CO2only_A_2.T_avCDR_upper[-1] - fair_gold_CO2only_A_2.T_avCDR_lower[-1]) * 0.5,
               (fair_silver_CO2only_A_2.T_avCDR_upper[-1] - fair_silver_CO2only_A_2.T_avCDR_lower[-1]) * 0.5,
               (fair_bronze_CO2only_A_2.T_avCDR_upper[-1] - fair_bronze_CO2only_A_2.T_avCDR_lower[-1]) * 0.5],
              [(fair_gold_CO2only_A_3.T_avCDR_upper[-1] - fair_gold_CO2only_A_3.T_avCDR_lower[-1]) * 0.5,
               (fair_silver_CO2only_A_3.T_avCDR_upper[-1] - fair_silver_CO2only_A_3.T_avCDR_lower[-1]) * 0.5,
               (fair_bronze_CO2only_A_3.T_avCDR_upper[-1] - fair_bronze_CO2only_A_3.T_avCDR_lower[-1]) * 0.5]
              ],
        columns = [baseline_1, baseline_2, baseline_3],
        index=[scenario1, scenario2, scenario3])
    return summary_T_CO2only_df, summary_T_CO2only_err_df


def make_bar_costkm_alltechs_summary(costkm_1_gold, costkm_2_gold, costkm_3_gold,
                         costkm_1_silver, costkm_2_silver, costkm_3_silver,
                         costkm_1_bronze, costkm_2_bronze, costkm_3_bronze,
                         costkm_1_tech1_gold, costkm_2_tech1_gold, costkm_3_tech1_gold,
                         costkm_1_tech1_silver, costkm_2_tech1_silver, costkm_3_tech1_silver,
                         costkm_1_tech1_bronze, costkm_2_tech1_bronze, costkm_3_tech1_bronze,
                         costkm_1_tech2_gold, costkm_2_tech2_gold, costkm_3_tech2_gold,
                         costkm_1_tech2_silver, costkm_2_tech2_silver, costkm_3_tech2_silver,
                         costkm_1_tech2_bronze, costkm_2_tech2_bronze, costkm_3_tech2_bronze,
                         scenario1 = 'BAU',scenario2 = 'Air Pollution',scenario3 = 'Mitigation',
                         tech1 = TECH_1, tech2 = TECH_2, what = 'mean'):
    """
    Make summary of cost of CDR per km flown.
    :param what: what to calculate the cost based on, whether cumulative of mean CDR and km flown
    :return: summary data frame
    """
    if what == 'cumulative':
        summarycostkm_1_gold = np.sum(unumpy.nominal_values(costkm_1_gold), axis = 0)
        summarycostkm_2_gold = np.sum(unumpy.nominal_values(costkm_2_gold), axis = 0)
        summarycostkm_3_gold = np.sum(unumpy.nominal_values(costkm_3_gold), axis = 0)
        summarycostkm_1_silver = np.sum(unumpy.nominal_values(costkm_1_silver), axis = 0)
        summarycostkm_2_silver = np.sum(unumpy.nominal_values(costkm_2_silver), axis = 0)
        summarycostkm_3_silver = np.sum(unumpy.nominal_values(costkm_3_silver), axis = 0)
        summarycostkm_1_bronze = np.sum(unumpy.nominal_values(costkm_1_bronze), axis = 0)
        summarycostkm_2_bronze = np.sum(unumpy.nominal_values(costkm_2_bronze), axis = 0)
        summarycostkm_3_bronze = np.sum(unumpy.nominal_values(costkm_3_bronze), axis = 0)
        summarycostkm_std_1_gold = np.sum(unumpy.std_devs(costkm_1_gold), axis = 0)
        summarycostkm_std_2_gold = np.sum(unumpy.std_devs(costkm_2_gold), axis = 0)
        summarycostkm_std_3_gold = np.sum(unumpy.std_devs(costkm_3_gold), axis = 0)
        summarycostkm_std_1_silver = np.sum(unumpy.std_devs(costkm_1_silver), axis = 0)
        summarycostkm_std_2_silver = np.sum(unumpy.std_devs(costkm_2_silver), axis = 0)
        summarycostkm_std_3_silver = np.sum(unumpy.std_devs(costkm_3_silver), axis = 0)
        summarycostkm_std_1_bronze = np.sum(unumpy.std_devs(costkm_1_bronze), axis = 0)
        summarycostkm_std_2_bronze = np.sum(unumpy.std_devs(costkm_2_bronze), axis = 0)
        summarycostkm_std_3_bronze = np.sum(unumpy.std_devs(costkm_3_bronze), axis = 0)
        summarycostkm_1_tech1_gold = np.sum(unumpy.nominal_values(costkm_1_tech1_gold), axis = 0)
        summarycostkm_2_tech1_gold = np.sum(unumpy.nominal_values(costkm_2_tech1_gold), axis = 0)
        summarycostkm_3_tech1_gold = np.sum(unumpy.nominal_values(costkm_3_tech1_gold), axis = 0)
        summarycostkm_1_tech1_silver = np.sum(unumpy.nominal_values(costkm_1_tech1_silver), axis = 0)
        summarycostkm_2_tech1_silver = np.sum(unumpy.nominal_values(costkm_2_tech1_silver), axis = 0)
        summarycostkm_3_tech1_silver = np.sum(unumpy.nominal_values(costkm_3_tech1_silver), axis = 0)
        summarycostkm_1_tech1_bronze = np.sum(unumpy.nominal_values(costkm_1_tech1_bronze), axis = 0)
        summarycostkm_2_tech1_bronze = np.sum(unumpy.nominal_values(costkm_2_tech1_bronze), axis = 0)
        summarycostkm_3_tech1_bronze = np.sum(unumpy.nominal_values(costkm_3_tech1_bronze), axis = 0)
        summarycostkm_std_1_tech1_gold = np.sum(unumpy.std_devs(costkm_1_tech1_gold), axis = 0)
        summarycostkm_std_2_tech1_gold = np.sum(unumpy.std_devs(costkm_2_tech1_gold), axis = 0)
        summarycostkm_std_3_tech1_gold = np.sum(unumpy.std_devs(costkm_3_tech1_gold), axis = 0)
        summarycostkm_std_1_tech1_silver = np.sum(unumpy.std_devs(costkm_1_tech1_silver), axis = 0)
        summarycostkm_std_2_tech1_silver = np.sum(unumpy.std_devs(costkm_2_tech1_silver), axis = 0)
        summarycostkm_std_3_tech1_silver = np.sum(unumpy.std_devs(costkm_3_tech1_silver), axis = 0)
        summarycostkm_std_1_tech1_bronze = np.sum(unumpy.std_devs(costkm_1_tech1_bronze), axis = 0)
        summarycostkm_std_2_tech1_bronze = np.sum(unumpy.std_devs(costkm_2_tech1_bronze), axis = 0)
        summarycostkm_std_3_tech1_bronze = np.sum(unumpy.std_devs(costkm_3_tech1_bronze), axis = 0)
        summarycostkm_1_tech2_gold = np.sum(unumpy.nominal_values(costkm_1_tech2_gold), axis = 0)
        summarycostkm_2_tech2_gold = np.sum(unumpy.nominal_values(costkm_2_tech2_gold), axis = 0)
        summarycostkm_3_tech2_gold = np.sum(unumpy.nominal_values(costkm_3_tech2_gold), axis = 0)
        summarycostkm_1_tech2_silver = np.sum(unumpy.nominal_values(costkm_1_tech2_silver), axis = 0)
        summarycostkm_2_tech2_silver = np.sum(unumpy.nominal_values(costkm_2_tech2_silver), axis = 0)
        summarycostkm_3_tech2_silver = np.sum(unumpy.nominal_values(costkm_3_tech2_silver), axis = 0)
        summarycostkm_1_tech2_bronze = np.sum(unumpy.nominal_values(costkm_1_tech2_bronze), axis = 0)
        summarycostkm_2_tech2_bronze = np.sum(unumpy.nominal_values(costkm_2_tech2_bronze), axis = 0)
        summarycostkm_3_tech2_bronze = np.sum(unumpy.nominal_values(costkm_3_tech2_bronze), axis = 0)
        summarycostkm_std_1_tech2_gold = np.sum(unumpy.std_devs(costkm_1_tech2_gold), axis = 0)
        summarycostkm_std_2_tech2_gold = np.sum(unumpy.std_devs(costkm_2_tech2_gold), axis = 0)
        summarycostkm_std_3_tech2_gold = np.sum(unumpy.std_devs(costkm_3_tech2_gold), axis = 0)
        summarycostkm_std_1_tech2_silver = np.sum(unumpy.std_devs(costkm_1_tech2_silver), axis = 0)
        summarycostkm_std_2_tech2_silver = np.sum(unumpy.std_devs(costkm_2_tech2_silver), axis = 0)
        summarycostkm_std_3_tech2_silver = np.sum(unumpy.std_devs(costkm_3_tech2_silver), axis = 0)
        summarycostkm_std_1_tech2_bronze = np.sum(unumpy.std_devs(costkm_1_tech2_bronze), axis = 0)
        summarycostkm_std_2_tech2_bronze = np.sum(unumpy.std_devs(costkm_2_tech2_bronze), axis = 0)
        summarycostkm_std_3_tech2_bronze = np.sum(unumpy.std_devs(costkm_3_tech2_bronze), axis = 0)

    elif what == 'mean':
        summarycostkm_1_gold = np.mean(unumpy.nominal_values(costkm_1_gold[costkm_1_gold.index >= '2020']), axis = 0)
        summarycostkm_2_gold = np.mean(unumpy.nominal_values(costkm_2_gold[costkm_2_gold.index >= '2020']), axis = 0)
        summarycostkm_3_gold = np.mean(unumpy.nominal_values(costkm_3_gold[costkm_3_gold.index >= '2020']), axis = 0)
        summarycostkm_std_1_gold = np.mean(unumpy.std_devs(costkm_1_gold[costkm_1_gold.index >= '2020']), axis = 0)
        summarycostkm_std_2_gold = np.mean(unumpy.std_devs(costkm_2_gold[costkm_2_gold.index >= '2020']), axis = 0)
        summarycostkm_std_3_gold = np.mean(unumpy.std_devs(costkm_3_gold[costkm_3_gold.index >= '2020']), axis = 0)
        summarycostkm_1_tech1_gold = np.mean(unumpy.nominal_values(costkm_1_tech1_gold[costkm_1_tech1_gold.index >= '2020']), axis = 0)
        summarycostkm_2_tech1_gold = np.mean(unumpy.nominal_values(costkm_2_tech1_gold[costkm_2_tech1_gold.index >= '2020']), axis = 0)
        summarycostkm_3_tech1_gold = np.mean(unumpy.nominal_values(costkm_3_tech1_gold[costkm_3_tech1_gold.index >= '2020']), axis = 0)
        summarycostkm_std_1_tech1_gold = np.mean(unumpy.std_devs(costkm_1_tech1_gold[costkm_1_tech1_gold.index >= '2020']), axis = 0)
        summarycostkm_std_2_tech1_gold = np.mean(unumpy.std_devs(costkm_2_tech1_gold[costkm_2_tech1_gold.index >= '2020']), axis = 0)
        summarycostkm_std_3_tech1_gold = np.mean(unumpy.std_devs(costkm_3_tech1_gold[costkm_3_tech1_gold .index >= '2020']), axis = 0)
        summarycostkm_1_tech2_gold = np.mean(unumpy.nominal_values(costkm_1_tech2_gold[costkm_1_tech2_gold .index >= '2020']), axis = 0)
        summarycostkm_2_tech2_gold = np.mean(unumpy.nominal_values(costkm_2_tech2_gold[costkm_2_tech2_gold .index >= '2020']), axis = 0)
        summarycostkm_3_tech2_gold = np.mean(unumpy.nominal_values(costkm_3_tech2_gold[costkm_3_tech2_gold .index >= '2020']), axis = 0)
        summarycostkm_std_1_tech2_gold = np.mean(unumpy.std_devs(costkm_1_tech2_gold[costkm_1_tech2_gold .index >= '2020']), axis = 0)
        summarycostkm_std_2_tech2_gold = np.mean(unumpy.std_devs(costkm_2_tech2_gold[costkm_2_tech2_gold .index >= '2020']), axis = 0)
        summarycostkm_std_3_tech2_gold = np.mean(unumpy.std_devs(costkm_3_tech2_gold[costkm_3_tech2_gold .index >= '2020']), axis = 0)
        summarycostkm_1_silver = np.mean(unumpy.nominal_values(costkm_1_silver[costkm_1_silver .index >= '2020']), axis = 0)
        summarycostkm_2_silver = np.mean(unumpy.nominal_values(costkm_2_silver[costkm_2_silver .index >= '2020']), axis = 0)
        summarycostkm_3_silver = np.mean(unumpy.nominal_values(costkm_3_silver[costkm_3_silver .index >= '2020']), axis = 0)
        summarycostkm_std_1_silver = np.mean(unumpy.std_devs(costkm_1_silver[costkm_1_silver .index >= '2020']), axis = 0)
        summarycostkm_std_2_silver = np.mean(unumpy.std_devs(costkm_2_silver[costkm_2_silver .index >= '2020']), axis = 0)
        summarycostkm_std_3_silver = np.mean(unumpy.std_devs(costkm_3_silver[costkm_3_silver .index >= '2020']), axis = 0)
        summarycostkm_1_tech1_silver = np.mean(unumpy.nominal_values(costkm_1_tech1_silver[costkm_1_tech1_silver .index >= '2020']), axis = 0)
        summarycostkm_2_tech1_silver = np.mean(unumpy.nominal_values(costkm_2_tech1_silver[costkm_2_tech1_silver .index >= '2020']), axis = 0)
        summarycostkm_3_tech1_silver = np.mean(unumpy.nominal_values(costkm_3_tech1_silver[costkm_3_tech1_silver .index >= '2020']), axis = 0)
        summarycostkm_std_1_tech1_silver = np.mean(unumpy.std_devs(costkm_1_tech1_silver[costkm_1_tech1_silver .index >= '2020']), axis = 0)
        summarycostkm_std_2_tech1_silver = np.mean(unumpy.std_devs(costkm_2_tech1_silver[costkm_2_tech1_silver .index >= '2020']), axis = 0)
        summarycostkm_std_3_tech1_silver = np.mean(unumpy.std_devs(costkm_3_tech1_silver[costkm_3_tech1_silver .index >= '2020']), axis = 0)
        summarycostkm_1_tech2_silver = np.mean(unumpy.nominal_values(costkm_1_tech2_silver[costkm_1_tech2_silver .index >= '2020']), axis = 0)
        summarycostkm_2_tech2_silver = np.mean(unumpy.nominal_values(costkm_2_tech2_silver[costkm_2_tech2_silver .index >= '2020']), axis = 0)
        summarycostkm_3_tech2_silver = np.mean(unumpy.nominal_values(costkm_3_tech2_silver[costkm_3_tech2_silver .index >= '2020']), axis = 0)
        summarycostkm_std_1_tech2_silver = np.mean(unumpy.std_devs(costkm_1_tech2_silver[costkm_1_tech2_silver .index >= '2020']), axis = 0)
        summarycostkm_std_2_tech2_silver = np.mean(unumpy.std_devs(costkm_2_tech2_silver[costkm_2_tech2_silver .index >= '2020']), axis = 0)
        summarycostkm_std_3_tech2_silver = np.mean(unumpy.std_devs(costkm_3_tech2_silver[costkm_3_tech2_silver .index >= '2020']), axis = 0)
        summarycostkm_1_bronze = np.mean(unumpy.nominal_values(costkm_1_bronze[costkm_1_bronze .index >= '2020']), axis = 0)
        summarycostkm_2_bronze = np.mean(unumpy.nominal_values(costkm_2_bronze[costkm_2_bronze .index >= '2020']), axis = 0)
        summarycostkm_3_bronze = np.mean(unumpy.nominal_values(costkm_3_bronze[costkm_3_bronze .index >= '2020']), axis = 0)
        summarycostkm_std_1_bronze = np.mean(unumpy.std_devs(costkm_1_bronze[costkm_1_bronze .index >= '2020']), axis = 0)
        summarycostkm_std_2_bronze = np.mean(unumpy.std_devs(costkm_2_bronze[costkm_2_bronze .index >= '2020']), axis = 0)
        summarycostkm_std_3_bronze = np.mean(unumpy.std_devs(costkm_3_bronze[costkm_3_bronze .index >= '2020']), axis = 0)
        summarycostkm_1_tech1_bronze = np.mean(unumpy.nominal_values(costkm_1_tech1_bronze[costkm_1_tech1_bronze .index >= '2020']), axis = 0)
        summarycostkm_2_tech1_bronze = np.mean(unumpy.nominal_values(costkm_2_tech1_bronze[costkm_2_tech1_bronze .index >= '2020']), axis = 0)
        summarycostkm_3_tech1_bronze = np.mean(unumpy.nominal_values(costkm_3_tech1_bronze[costkm_3_tech1_bronze .index >= '2020']), axis = 0)
        summarycostkm_std_1_tech1_bronze = np.mean(unumpy.std_devs(costkm_1_tech1_bronze[costkm_1_tech1_bronze .index >= '2020']), axis = 0)
        summarycostkm_std_2_tech1_bronze = np.mean(unumpy.std_devs(costkm_2_tech1_bronze[costkm_2_tech1_bronze .index >= '2020']), axis = 0)
        summarycostkm_std_3_tech1_bronze = np.mean(unumpy.std_devs(costkm_3_tech1_bronze[costkm_3_tech1_bronze .index >= '2020']), axis = 0)
        summarycostkm_1_tech2_bronze = np.mean(unumpy.nominal_values(costkm_1_tech2_bronze[costkm_1_tech2_bronze .index >= '2020']), axis = 0)
        summarycostkm_2_tech2_bronze = np.mean(unumpy.nominal_values(costkm_2_tech2_bronze[costkm_2_tech2_bronze .index >= '2020']), axis = 0)
        summarycostkm_3_tech2_bronze = np.mean(unumpy.nominal_values(costkm_3_tech2_bronze[costkm_3_tech2_bronze .index >= '2020']), axis = 0)
        summarycostkm_std_1_tech2_bronze = np.mean(unumpy.std_devs(costkm_1_tech2_bronze[costkm_1_tech2_bronze .index >= '2020']), axis = 0)
        summarycostkm_std_2_tech2_bronze = np.mean(unumpy.std_devs(costkm_2_tech2_bronze[costkm_2_tech2_bronze .index >= '2020']), axis = 0)
        summarycostkm_std_3_tech2_bronze = np.mean(unumpy.std_devs(costkm_3_tech2_bronze[costkm_3_tech2_bronze .index >= '2020']), axis = 0)

    costkm_summary = pd.DataFrame({"summary_Tot_costkm": [float((summarycostkm_1_gold )),
                                                float((summarycostkm_2_gold )),
                                                float((summarycostkm_3_gold )),
                                                float((summarycostkm_1_silver )),
                                                float((summarycostkm_2_silver )),
                                                float((summarycostkm_3_silver )),
                                                float((summarycostkm_1_bronze )),
                                                float((summarycostkm_2_bronze )),
                                                float((summarycostkm_3_bronze )),
                                                float((summarycostkm_1_tech1_gold )),
                                                float((summarycostkm_2_tech1_gold )),
                                                float((summarycostkm_3_tech1_gold )),
                                                float((summarycostkm_1_tech1_silver )),
                                                float((summarycostkm_2_tech1_silver )),
                                                float((summarycostkm_3_tech1_silver )),
                                                float((summarycostkm_1_tech1_bronze )),
                                                float((summarycostkm_2_tech1_bronze )),
                                                float((summarycostkm_3_tech1_bronze )),
                                                    float((summarycostkm_1_tech2_gold )),
                                                    float((summarycostkm_2_tech2_gold )),
                                                    float((summarycostkm_3_tech2_gold )),
                                                    float((summarycostkm_1_tech2_silver )),
                                                    float((summarycostkm_2_tech2_silver )),
                                                    float((summarycostkm_3_tech2_silver )),
                                                    float((summarycostkm_1_tech2_bronze )),
                                                    float((summarycostkm_2_tech2_bronze )),
                                                    float((summarycostkm_3_tech2_bronze )),],
                                "summary_Tot_costkm_std": [float((summarycostkm_std_1_gold )),
                                                float((summarycostkm_std_2_gold )),
                                                float((summarycostkm_std_3_gold )),
                                                float((summarycostkm_std_1_silver )),
                                                float((summarycostkm_std_2_silver )),
                                                float((summarycostkm_std_3_silver )),
                                                float((summarycostkm_std_1_bronze )),
                                                float((summarycostkm_std_2_bronze )),
                                                float((summarycostkm_std_3_bronze )),
                                                float((summarycostkm_std_1_tech1_gold )),
                                                float((summarycostkm_std_2_tech1_gold )),
                                                float((summarycostkm_std_3_tech1_gold )),
                                                float((summarycostkm_std_1_tech1_silver )),
                                                float((summarycostkm_std_2_tech1_silver )),
                                                float((summarycostkm_std_3_tech1_silver )),
                                                float((summarycostkm_std_1_tech1_bronze )),
                                                float((summarycostkm_std_2_tech1_bronze )),
                                                float((summarycostkm_std_3_tech1_bronze )),
                                                    float((summarycostkm_std_1_tech2_gold )),
                                                    float((summarycostkm_std_2_tech2_gold )),
                                                    float((summarycostkm_std_3_tech2_gold )),
                                                    float((summarycostkm_std_1_tech2_silver )),
                                                    float((summarycostkm_std_2_tech2_silver )),
                                                    float((summarycostkm_std_3_tech2_silver )),
                                                    float((summarycostkm_std_1_tech2_bronze )),
                                                    float((summarycostkm_std_2_tech2_bronze )),
                                                    float((summarycostkm_std_3_tech2_bronze ))],
                    "Scenario": [scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3,
                                 scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3,
                                 scenario1,scenario2,scenario3,scenario1,scenario2,scenario3, scenario1,scenario2,scenario3],
                    "Technology": ["Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1","Jet A1",
                                   tech1,tech1,tech1,tech1,tech1,tech1,tech1,tech1,tech1,
                                   tech2,tech2,tech2,tech2,tech2,tech2,tech2,tech2,tech2],
                    "Climate neutrality": [baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2, baseline_3, baseline_3, baseline_3,
                                           baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2, baseline_3, baseline_3, baseline_3,
                                           baseline_1, baseline_1, baseline_1, baseline_2, baseline_2, baseline_2, baseline_3, baseline_3, baseline_3]})
    #costkm_summary = pd.DataFrame(index = pd.to_datetime([date1, date2, date3], format='%Y'), data= np.stack((costkm_date1, costkm_date2, costkm_date3)), columns = costkm.columns)
    return costkm_summary

#=============================== MAKE SUMMARY ALL METRICS ==================================
def make_bar_T_metrics_summary(fair_base_1, fair_base_2, fair_base_3 ,
                         fair_EWF_1, fair_EWF_2, fair_EWF_3,
                         fair_GWP100_1, fair_GWP100_2, fair_GWP100_3,
                                   fair_GWPstar_1, fair_GWPstar_2, fair_GWPstar_3,
                               base = 'Lee',
                         scenario1 = 'BAU',scenario2 = 'Air Pollution',scenario3 = 'Mitigation',
                         metric1 = 'EWF', metric2 = 'GWP100', metric3 = 'GWP*'):
    """
    make summary dataframe of temperatures under emissions calculated with different metrics
    :param base: name of "real" emissions (ERF calculated following Lee et al. 2021)
    :param metric1: name of first metric, default: EWF
    :param metric2: name of second metric, default : GWP100
    :param metric3: name of third metric, default: GWP*
    :return:
    """
    summary_base_1 = fair_base_1.T_aviation[-1]
    summary_base_2 = fair_base_2.T_aviation[-1]
    summary_base_3 = fair_base_3.T_aviation[-1]
    summary_base_std_1 = 0.5*(fair_base_1.T_aviation_upper[-1] - fair_base_1.T_aviation_lower[-1])
    summary_base_std_2 = 0.5*(fair_base_2.T_aviation_upper[-1] - fair_base_2.T_aviation_lower[-1])
    summary_base_std_3 = 0.5*(fair_base_3.T_aviation_upper[-1] - fair_base_3.T_aviation_lower[-1])
    summary_EWF_1 = fair_EWF_1.T_aviation[-1]
    summary_EWF_2 = fair_EWF_2.T_aviation[-1]
    summary_EWF_3 = fair_EWF_3.T_aviation[-1]
    summary_EWF_std_1 = 0.5*(fair_EWF_1.T_aviation_upper[-1] - fair_EWF_1.T_aviation_lower[-1])
    summary_EWF_std_2 = 0.5*(fair_EWF_2.T_aviation_upper[-1] - fair_EWF_2.T_aviation_lower[-1])
    summary_EWF_std_3 = 0.5*(fair_EWF_3.T_aviation_upper[-1] - fair_EWF_3.T_aviation_lower[-1])
    summary_GWP100_1 = fair_GWP100_1.T_aviation[-1]
    summary_GWP100_2 = fair_GWP100_2.T_aviation[-1]
    summary_GWP100_3 = fair_GWP100_3.T_aviation[-1]
    summary_GWP100_std_1 = 0.5*(fair_GWP100_1.T_aviation_upper[-1] - fair_GWP100_1.T_aviation_lower[-1])
    summary_GWP100_std_2 = 0.5*(fair_GWP100_2.T_aviation_upper[-1] - fair_GWP100_2.T_aviation_lower[-1])
    summary_GWP100_std_3 = 0.5*(fair_GWP100_3.T_aviation_upper[-1] - fair_GWP100_3.T_aviation_lower[-1])
    summary_GWPstar_1 = fair_GWPstar_1.T_aviation[-1]
    summary_GWPstar_2 = fair_GWPstar_2.T_aviation[-1]
    summary_GWPstar_3 = fair_GWPstar_3.T_aviation[-1]
    summary_GWPstar_std_1 = 0.5*(fair_GWPstar_1.T_aviation_upper[-1] - fair_GWPstar_1.T_aviation_lower[-1])
    summary_GWPstar_std_2 = 0.5*(fair_GWPstar_2.T_aviation_upper[-1] - fair_GWPstar_2.T_aviation_lower[-1])
    summary_GWPstar_std_3 = 0.5*(fair_GWPstar_3.T_aviation_upper[-1] - fair_GWPstar_3.T_aviation_lower[-1])

    summary = pd.DataFrame({"T_2100": [summary_base_1, summary_base_2, summary_base_3,
                                       summary_EWF_1, summary_EWF_2, summary_EWF_3,
                                       summary_GWP100_1, summary_GWP100_2, summary_GWP100_3,
                                       summary_GWPstar_1, summary_GWPstar_2, summary_GWPstar_3,],
                                "T_2100_std": [summary_base_std_1, summary_base_std_2, summary_base_std_3,
                                       summary_EWF_std_1, summary_EWF_std_2, summary_EWF_std_3,
                                       summary_GWP100_std_1, summary_GWP100_std_2, summary_GWP100_std_3,
                                       summary_GWPstar_std_1, summary_GWPstar_std_2, summary_GWPstar_std_3
                                                    ],
                                 "Scenario": [scenario1,scenario2,scenario3,scenario1,scenario2,scenario3,
                                              scenario1,scenario2,scenario3, scenario1,scenario2,scenario3],
                    "Metric": [base, base, base, metric1,metric1,metric1,
                               metric2,metric2,metric2,metric3,metric3,metric3,],})
    return summary


def sensitivity_transition_date(aviation_df1, aviation_df2, aviation_df3, aviation_df_ref, ERF_df_ref, start_date,
                                uptake_type, tech, erf_factors, trans_date1 = 2050, trans_date2 = 2070, trans_date3=2090):
    date_tech = [str(trans_date1), str(trans_date2), str(trans_date3)]
    container = pd.DataFrame(columns = date_tech, index = ['SSP1_2.6', 'SSP2_4.5','SSP5_8.5',
                                                           'SSP1_2.6 Bronze', 'SSP1_2.6 Silver','SSP1_2.6 Gold',
                                                           'SSP2_4.5 Bronze', 'SSP2_4.5 Silver', 'SSP2_4.5 Gold',
                                                           'SSP5_8.5 Bronze', 'SSP5_8.5 Silver', 'SSP5_8.5 Gold'])
    for i in [trans_date1, trans_date2, trans_date3]:
        aviation_tech_df_1 = make_emissions_new_tech(aviation_df1, start_date, uptake_type, tech, i)  # in this scenario: start of tech1 in 2020
        aviation_tech_df_2 = make_emissions_new_tech(aviation_df2, start_date, uptake_type, tech, i)
        aviation_tech_df_3 = make_emissions_new_tech(aviation_df3, start_date, uptake_type, tech, i)
        ERF_tech_df_1 = calculate_ERF_techs(aviation_tech_df_1, erf_factors)
        ERF_tech_df_2 = calculate_ERF_techs(aviation_tech_df_2, erf_factors)
        ERF_tech_df_3 = calculate_ERF_techs(aviation_tech_df_3, erf_factors)
        # make baseline BRONZE
        bronze_baseline_tech_df_1, bronze_baseline_tech_df_2, bronze_baseline_tech_df_3, bronze_ERF_tech_df_1, bronze_ERF_tech_df_2, bronze_ERF_tech_df_3 = \
            make_target_scenarios(aviation_tech_df_1.best, aviation_tech_df_2.best, aviation_tech_df_3.best,
                                  erf_factors, 2019, 'Bronze')
        # calculate CDR for BRONZE
        CDR_bronze_tech_df_1, CDR_bronze_tech_df_2, CDR_bronze_tech_df_3 = make_CDR_scenarios(
            ERF_tech_df_1 - bronze_ERF_tech_df_1, ERF_tech_df_2 - bronze_ERF_tech_df_2,
            ERF_tech_df_3 - bronze_ERF_tech_df_3,
            aviation_tech_df_1.best - bronze_baseline_tech_df_1,
            aviation_tech_df_2.best - bronze_baseline_tech_df_2,
            aviation_tech_df_3.best - bronze_baseline_tech_df_3, dt=20, start_date=2019)

        # calculate CDR for SILVER
        CDR_silver_tech_df_1, CDR_silver_tech_df_2, CDR_silver_tech_df_3 = make_CDR_scenarios(
            ERF_tech_df_1 - ERF_df_ref, ERF_tech_df_2 - ERF_df_ref, ERF_tech_df_3 - ERF_df_ref,
            aviation_tech_df_1.best - aviation_df_ref, aviation_tech_df_2.best - aviation_df_ref,
            aviation_tech_df_3.best - aviation_df_ref, dt=20, start_date=2019)

        # make baseline GOLD
        gold_baseline_tech_df_1, gold_baseline_tech_df_2, gold_baseline_tech_df_3, gold_ERF_tech_df_1, gold_ERF_tech_df_2, gold_ERF_tech_df_3 = \
            make_target_scenarios(aviation_tech_df_1.best, aviation_tech_df_2.best, aviation_tech_df_3.best,
                                  erf_factors, 2019, 'Gold', intermediate_goal=True)
        # calculate CDR for GOLD
        CDR_gold_tech_df_1, CDR_gold_tech_df_2, CDR_gold_tech_df_3 = make_CDR_scenarios(
            ERF_tech_df_1 - gold_ERF_tech_df_1, ERF_tech_df_2 - gold_ERF_tech_df_2,
            ERF_tech_df_3 - gold_ERF_tech_df_3,
            aviation_tech_df_1.best - gold_baseline_tech_df_1, aviation_tech_df_2.best - gold_baseline_tech_df_2,
            aviation_tech_df_3.best - gold_baseline_tech_df_3, dt=20, start_date=2019)
        # Bronze
        fair_bronze_tech_1 = test_CO2_Fair(aviation_tech_df_1.best, CDR_bronze_tech_df_1, ERF_tech_df_1,
                                            bronze_baseline_tech_df_1, bronze_ERF_tech_df_1,
                                            start_year=2019, baseline='Gold')
        fair_bronze_tech_2 = test_CO2_Fair(aviation_tech_df_2.best, CDR_bronze_tech_df_2, ERF_tech_df_2,
                                            bronze_baseline_tech_df_2, bronze_ERF_tech_df_2,
                                            start_year=2019, baseline='Gold')
        fair_bronze_tech_3 = test_CO2_Fair(aviation_tech_df_3.best, CDR_bronze_tech_df_3, ERF_tech_df_3,
                                            bronze_baseline_tech_df_3, bronze_ERF_tech_df_3,
                                            start_year=2019, baseline='Gold')

        # Silver
        fair_silver_tech_1 = test_CO2_Fair(aviation_tech_df_1.best, CDR_silver_tech_df_1, ERF_tech_df_1,
                                            aviation_df_ref, ERF_df_ref,
                                            start_year=2019, baseline='SSP1_19')
        fair_silver_tech_2 = test_CO2_Fair(aviation_tech_df_2.best, CDR_silver_tech_df_2, ERF_tech_df_2,
                                            aviation_df_ref, ERF_df_ref,
                                            start_year=2019, baseline='SSP1_19')
        fair_silver_tech_3 = test_CO2_Fair(aviation_tech_df_2.best, CDR_silver_tech_df_2, ERF_tech_df_2,
                                            aviation_df_ref, ERF_df_ref,
                                            start_year=2019, baseline='SSP1_19')

        # Gold
        fair_gold_tech_1 = test_CO2_Fair(aviation_tech_df_1.best, CDR_gold_tech_df_1, ERF_tech_df_1,
                                          gold_baseline_tech_df_1, gold_ERF_tech_df_1,
                                          start_year=2019, baseline='Gold')
        fair_gold_tech_2 = test_CO2_Fair(aviation_tech_df_2.best, CDR_gold_tech_df_2, ERF_tech_df_2,
                                          gold_baseline_tech_df_2, gold_ERF_tech_df_2,
                                          start_year=2019, baseline='Gold')
        fair_gold_tech_3 = test_CO2_Fair(aviation_tech_df_3.best, CDR_gold_tech_df_3, ERF_tech_df_3,
                                          gold_baseline_tech_df_3, gold_ERF_tech_df_3,
                                          start_year=2019, baseline='Gold')

        summary_tech_df, summary_tech_err_df = make_summary_T_CO2only(
            fair_bronze_tech_1, fair_bronze_tech_2, fair_bronze_tech_3,
            fair_silver_tech_1, fair_silver_tech_2, fair_silver_tech_3,
            fair_gold_tech_1, fair_gold_tech_2, fair_gold_tech_3,
            'SSP1_26', 'SSP2_45', 'SSP5_85', 'Bronze', 'Silver', 'Gold')

        container[str(i)] = np.concatenate((np.array([fair_bronze_tech_1.T_aviation[-1],fair_bronze_tech_2.T_aviation[-1],
                                                     fair_bronze_tech_3.T_aviation[-1]]),
                                           unumpy.uarray(summary_tech_df.to_numpy().flatten(),
                                          summary_tech_err_df.to_numpy().flatten())))
    container['percent change '+ str(trans_date1)] = 100*(unumpy.nominal_values(container[str(trans_date1)]) -
                                                 unumpy.nominal_values(container[str(trans_date2)]))/ \
                                                unumpy.nominal_values(container[str(trans_date2)])
    container['percent change '+ str(trans_date3)] = 100*(unumpy.nominal_values(container[str(trans_date3)]) -
                                                 unumpy.nominal_values(container[str(trans_date2)]))/ \
                                                unumpy.nominal_values(container[str(trans_date2)])

    return container


def sensitivity_covid(aviation_df1, aviation_df2, aviation_df3, aviation_df_ref, ERF_df_ref, start_date,
                                uptake_type, tech, erf_factors, trans_date1 = 2050, trans_date2 = 2070, trans_date3=2090):
    date_tech = [str(trans_date1), str(trans_date2), str(trans_date3)]
    container = pd.DataFrame(columns = date_tech, index = ['SSP1_2.6', 'SSP2_4.5','SSP5_8.5',
                                                           'SSP1_2.6 Bronze', 'SSP1_2.6 Silver','SSP1_2.6 Gold',
                                                           'SSP2_4.5 Bronze', 'SSP2_4.5 Silver', 'SSP2_4.5 Gold',
                                                           'SSP5_8.5 Bronze', 'SSP5_8.5 Silver', 'SSP5_8.5 Gold'])
    for i in [trans_date1, trans_date2, trans_date3]:
        aviation_tech_df_1 = make_emissions_new_tech(aviation_df1, start_date, uptake_type, tech, i)  # in this scenario: start of tech1 in 2020
        aviation_tech_df_2 = make_emissions_new_tech(aviation_df2, start_date, uptake_type, tech, i)
        aviation_tech_df_3 = make_emissions_new_tech(aviation_df3, start_date, uptake_type, tech, i)
        ERF_tech_df_1 = calculate_ERF_techs(aviation_tech_df_1, erf_factors)
        ERF_tech_df_2 = calculate_ERF_techs(aviation_tech_df_2, erf_factors)
        ERF_tech_df_3 = calculate_ERF_techs(aviation_tech_df_3, erf_factors)
        # make baseline BRONZE
        bronze_baseline_tech_df_1, bronze_baseline_tech_df_2, bronze_baseline_tech_df_3, bronze_ERF_tech_df_1, bronze_ERF_tech_df_2, bronze_ERF_tech_df_3 = \
            make_target_scenarios(aviation_tech_df_1.best, aviation_tech_df_2.best, aviation_tech_df_3.best,
                                  erf_factors, 2019, 'Bronze')
        # calculate CDR for BRONZE
        CDR_bronze_tech_df_1, CDR_bronze_tech_df_2, CDR_bronze_tech_df_3 = make_CDR_scenarios(
            ERF_tech_df_1 - bronze_ERF_tech_df_1, ERF_tech_df_2 - bronze_ERF_tech_df_2,
            ERF_tech_df_3 - bronze_ERF_tech_df_3,
            aviation_tech_df_1.best - bronze_baseline_tech_df_1,
            aviation_tech_df_2.best - bronze_baseline_tech_df_2,
            aviation_tech_df_3.best - bronze_baseline_tech_df_3, dt=20, start_date=2019)

        # calculate CDR for SILVER
        CDR_silver_tech_df_1, CDR_silver_tech_df_2, CDR_silver_tech_df_3 = make_CDR_scenarios(
            ERF_tech_df_1 - ERF_df_ref, ERF_tech_df_2 - ERF_df_ref, ERF_tech_df_3 - ERF_df_ref,
            aviation_tech_df_1.best - aviation_df_ref, aviation_tech_df_2.best - aviation_df_ref,
            aviation_tech_df_3.best - aviation_df_ref, dt=20, start_date=2019)

        # make baseline GOLD
        gold_baseline_tech_df_1, gold_baseline_tech_df_2, gold_baseline_tech_df_3, gold_ERF_tech_df_1, gold_ERF_tech_df_2, gold_ERF_tech_df_3 = \
            make_target_scenarios(aviation_tech_df_1.best, aviation_tech_df_2.best, aviation_tech_df_3.best,
                                  erf_factors, 2019, 'Gold', intermediate_goal=True)
        # calculate CDR for GOLD
        CDR_gold_tech_df_1, CDR_gold_tech_df_2, CDR_gold_tech_df_3 = make_CDR_scenarios(
            ERF_tech_df_1 - gold_ERF_tech_df_1, ERF_tech_df_2 - gold_ERF_tech_df_2,
            ERF_tech_df_3 - gold_ERF_tech_df_3,
            aviation_tech_df_1.best - gold_baseline_tech_df_1, aviation_tech_df_2.best - gold_baseline_tech_df_2,
            aviation_tech_df_3.best - gold_baseline_tech_df_3, dt=20, start_date=2019)
        # Bronze
        fair_bronze_tech_1 = test_CO2_Fair(aviation_tech_df_1.best, CDR_bronze_tech_df_1, ERF_tech_df_1,
                                            bronze_baseline_tech_df_1, bronze_ERF_tech_df_1,
                                            start_year=2019, baseline='Gold')
        fair_bronze_tech_2 = test_CO2_Fair(aviation_tech_df_2.best, CDR_bronze_tech_df_2, ERF_tech_df_2,
                                            bronze_baseline_tech_df_2, bronze_ERF_tech_df_2,
                                            start_year=2019, baseline='Gold')
        fair_bronze_tech_3 = test_CO2_Fair(aviation_tech_df_3.best, CDR_bronze_tech_df_3, ERF_tech_df_3,
                                            bronze_baseline_tech_df_3, bronze_ERF_tech_df_3,
                                            start_year=2019, baseline='Gold')

        # Silver
        fair_silver_tech_1 = test_CO2_Fair(aviation_tech_df_1.best, CDR_silver_tech_df_1, ERF_tech_df_1,
                                            aviation_df_ref, ERF_df_ref,
                                            start_year=2019, baseline='SSP1_19')
        fair_silver_tech_2 = test_CO2_Fair(aviation_tech_df_2.best, CDR_silver_tech_df_2, ERF_tech_df_2,
                                            aviation_df_ref, ERF_df_ref,
                                            start_year=2019, baseline='SSP1_19')
        fair_silver_tech_3 = test_CO2_Fair(aviation_tech_df_2.best, CDR_silver_tech_df_2, ERF_tech_df_2,
                                            aviation_df_ref, ERF_df_ref,
                                            start_year=2019, baseline='SSP1_19')

        # Gold
        fair_gold_tech_1 = test_CO2_Fair(aviation_tech_df_1.best, CDR_gold_tech_df_1, ERF_tech_df_1,
                                          gold_baseline_tech_df_1, gold_ERF_tech_df_1,
                                          start_year=2019, baseline='Gold')
        fair_gold_tech_2 = test_CO2_Fair(aviation_tech_df_2.best, CDR_gold_tech_df_2, ERF_tech_df_2,
                                          gold_baseline_tech_df_2, gold_ERF_tech_df_2,
                                          start_year=2019, baseline='Gold')
        fair_gold_tech_3 = test_CO2_Fair(aviation_tech_df_3.best, CDR_gold_tech_df_3, ERF_tech_df_3,
                                          gold_baseline_tech_df_3, gold_ERF_tech_df_3,
                                          start_year=2019, baseline='Gold')

        summary_tech_df, summary_tech_err_df = make_summary_T_CO2only(
            fair_bronze_tech_1, fair_bronze_tech_2, fair_bronze_tech_3,
            fair_silver_tech_1, fair_silver_tech_2, fair_silver_tech_3,
            fair_gold_tech_1, fair_gold_tech_2, fair_gold_tech_3,
            'SSP1_26', 'SSP2_45', 'SSP5_85', 'Bronze', 'Silver', 'Gold')

        container[str(i)] = np.concatenate((np.array([fair_bronze_tech_1.T_aviation[-1],fair_bronze_tech_2.T_aviation[-1],
                                                     fair_bronze_tech_3.T_aviation[-1]]),
                                           unumpy.uarray(summary_tech_df.to_numpy().flatten(),
                                          summary_tech_err_df.to_numpy().flatten())))
    container['percent change '+ str(trans_date1)] = 100*(unumpy.nominal_values(container[str(trans_date1)]) -
                                                 unumpy.nominal_values(container[str(trans_date2)]))/ \
                                                unumpy.nominal_values(container[str(trans_date2)])
    container['percent change '+ str(trans_date3)] = 100*(unumpy.nominal_values(container[str(trans_date3)]) -
                                                 unumpy.nominal_values(container[str(trans_date2)]))/ \
                                                unumpy.nominal_values(container[str(trans_date2)])

    return container




