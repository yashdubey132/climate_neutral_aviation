# imports
import numpy as np

import functions
import plots
from importlib import reload
reload(functions)
reload(plots)
from plots import *
from functions import *


# DEFINE SCENARIOS
SCENARIO1 = 'SSP1_26'
SCENARIO2 = 'SSP2_45'
SCENARIO3 = 'SSP5_85'
REF_SCENARIO = 'SSP1_19'

# DEFINE BASELINES
BASELINE1 = 'Gold'
BASELINE2 = 'Silver'
BASELINE3 = 'Bronze'
BASELINE4 = 'EWF'

TECH_1 = 'Zero-CO$_2$ fuels'
TECH_2 = 'No-emissions aircraft'

# set constants
START_DATE = 2019 #start of mitigation policy
START_DATE_REF = 2019 # start of mitigation policy for silver scenario
DATE_TECH1 = 2050 # date when the transition to tech1 is completed
DATE_TECH2 = 2080 # date when the transition to tech2 is completed
DELTA_T = 20 # delta t used in GWP* calculation
UPTAKE_TYPE = 'abrupt' # or 'sigma' or 'linear', type of growth of alternative flying technologies
CDR_COST = unumpy.uarray(150, 100) # cost of CDR plus uncertainty range
COLORS = ["#006d77","#83c5be","#c2e0eb","#ffddd2","#e29578"] # colors for palette
PALETTE = sns.set_palette(sns.color_palette(COLORS))  # make palette

# switch on and off "debug" modus, that makes plot to understand whether everything works correctly
debug = False # or True

first_run = False # or True - switch on to make csv emissions input data
plots_final = False # or False
plots_SI = False # or True to produce SI Figures
output_txt = True # or False
output_xlsx = True # or False
extra_plots = False # or True

#============= MAKE SCENARIOS UP UNTIL 2050=========================
# make emissions input data in csv form
if first_run == True:
    make_first_time_CMIP_aviation_scenarios()
# make historical time series
ERF_2018, aviation_2018, emissions_2018, ERF_factors = import_lee()
# make scenarios of aviation emissions following SSP-RCP scenarios
aviation_df_1, av_SSP2_1 = make_SSP_CMIP_emissions(emissions_2018.join(aviation_2018[['distance','Fuel']]), SCENARIO1)
aviation_df_2, av_SSP2_2 = make_SSP_CMIP_emissions(emissions_2018.join(aviation_2018[['distance','Fuel']]), SCENARIO2)
aviation_df_3, av_SSP2_3 = make_SSP_CMIP_emissions(emissions_2018.join(aviation_2018[['distance','Fuel']]), SCENARIO3)
aviation_df_ref, av_SSP2_ref = make_SSP_CMIP_emissions(emissions_2018.join(aviation_2018[['distance','Fuel']]), REF_SCENARIO)
# calculate Effective Radiative Forcing corresponding to different scenarios of emissions
ERF_df_1 = calculate_ERF(aviation_df_1, ERF_factors)
ERF_df_2 = calculate_ERF(aviation_df_2, ERF_factors)
ERF_df_3 = calculate_ERF(aviation_df_3, ERF_factors)
ERF_df_ref = calculate_ERF(aviation_df_ref, ERF_factors)

#============================ CALCULATE CDR ========================================================
# MAKE BRONZE SCENARIO
bronze_baseline_df_1, bronze_baseline_df_2, bronze_baseline_df_3, bronze_ERF_df_1, bronze_ERF_df_2, bronze_ERF_df_3 = \
    make_target_scenarios(aviation_df_1, aviation_df_2, aviation_df_3, ERF_factors, START_DATE, BASELINE3, intermediate_goal = True)
# Calculate CDR under BRONZE
CDR_bronze_df_1, CDR_bronze_df_2, CDR_bronze_df_3 = make_CDR_scenarios(
    ERF_df_1-bronze_ERF_df_1, ERF_df_2-bronze_ERF_df_2, ERF_df_3-bronze_ERF_df_3,
    aviation_df_1-bronze_baseline_df_1, aviation_df_2-bronze_baseline_df_2, aviation_df_3-bronze_baseline_df_3,
    dt = DELTA_T, start_date=START_DATE)

# Calculate CDR under SILVER
CDR_silver_df_1, CDR_silver_df_2, CDR_silver_df_3 = make_CDR_scenarios(
    ERF_df_1-ERF_df_ref, ERF_df_2-ERF_df_ref, ERF_df_3-ERF_df_ref,
    aviation_df_1-aviation_df_ref, aviation_df_2-aviation_df_ref, aviation_df_3-aviation_df_ref,
    dt = DELTA_T, start_date=START_DATE_REF)

# MAKE GOLD SCNEARIO
gold_baseline_df_1, gold_baseline_df_2, gold_baseline_df_3, gold_ERF_df_1, gold_ERF_df_2, gold_ERF_df_3 = \
    make_target_scenarios(aviation_df_1, aviation_df_2, aviation_df_3, ERF_factors, START_DATE, BASELINE1, intermediate_goal = True)
# Calculate CDR under GOLD
CDR_gold_df_1, CDR_gold_df_2, CDR_gold_df_3 = make_CDR_scenarios(
    ERF_df_1-gold_ERF_df_1, ERF_df_2-gold_ERF_df_2, ERF_df_3-gold_ERF_df_3,
    aviation_df_1-gold_baseline_df_1, aviation_df_2-gold_baseline_df_2, aviation_df_3-gold_baseline_df_3,
    dt = DELTA_T, start_date=START_DATE)


#============================== CALCULATE CDR CO2 ONLY ==========================
# BRONZE
CDR_bronze_CO2only_df_1, CDR_bronze_CO2only_df_2, CDR_bronze_CO2only_df_3 = make_CDR_scenarios(
    ERF_df_1-bronze_ERF_df_1, ERF_df_2-bronze_ERF_df_2, ERF_df_3-bronze_ERF_df_3,
    aviation_df_1-bronze_baseline_df_1, aviation_df_2-bronze_baseline_df_2, aviation_df_3-bronze_baseline_df_3,
    dt = DELTA_T, start_date=START_DATE, CO2only=True)
# SILVER
CDR_silver_CO2only_df_1, CDR_silver_CO2only_df_2, CDR_silver_CO2only_df_3 = make_CDR_scenarios(
    ERF_df_1-ERF_df_ref, ERF_df_2-ERF_df_ref, ERF_df_3-ERF_df_ref,
    aviation_df_1-aviation_df_ref, aviation_df_2-aviation_df_ref, aviation_df_3-aviation_df_ref,
    dt = DELTA_T, start_date=START_DATE_REF, CO2only=True)
# GOLD
CDR_gold_CO2only_df_1, CDR_gold_CO2only_df_2, CDR_gold_CO2only_df_3 = make_CDR_scenarios(
    ERF_df_1-gold_ERF_df_1, ERF_df_2-gold_ERF_df_2, ERF_df_3-gold_ERF_df_3,
    aviation_df_1-gold_baseline_df_1, aviation_df_2-gold_baseline_df_2, aviation_df_3-gold_baseline_df_3,
    dt = DELTA_T, start_date=START_DATE, CO2only=True)


#============================ COMPUTE SCENARIOS' ERF AND T WITH FAIR =============================
# Bronze
fair_bronze_1 = test_CO2_Fair(aviation_df_1, CDR_bronze_df_1, ERF_df_1, bronze_baseline_df_1, bronze_ERF_df_1,
                                start_year= START_DATE, baseline=BASELINE1)
fair_bronze_2 = test_CO2_Fair(aviation_df_2, CDR_bronze_df_2, ERF_df_2, bronze_baseline_df_2, bronze_ERF_df_2,
                                start_year= START_DATE, baseline=BASELINE1)
fair_bronze_3 = test_CO2_Fair(aviation_df_3, CDR_bronze_df_3, ERF_df_3, bronze_baseline_df_3, bronze_ERF_df_3,
                                start_year= START_DATE, baseline=BASELINE1)

# Silver
fair_silver_1 = test_CO2_Fair(aviation_df_1, CDR_silver_df_1, ERF_df_1, aviation_df_ref, ERF_df_ref,
                                start_year= START_DATE_REF, baseline=REF_SCENARIO)
fair_silver_2 = test_CO2_Fair(aviation_df_2, CDR_silver_df_2, ERF_df_2, aviation_df_ref, ERF_df_ref,
                                start_year= START_DATE_REF, baseline=REF_SCENARIO)
fair_silver_3 = test_CO2_Fair(aviation_df_2, CDR_silver_df_2, ERF_df_2, aviation_df_ref, ERF_df_ref,
                                start_year= START_DATE_REF, baseline=REF_SCENARIO)

# Gold
fair_gold_1 = test_CO2_Fair(aviation_df_1, CDR_gold_df_1, ERF_df_1, gold_baseline_df_1, gold_ERF_df_1,
                              start_year= START_DATE, baseline=BASELINE1)
fair_gold_2 = test_CO2_Fair(aviation_df_2, CDR_gold_df_2, ERF_df_2, gold_baseline_df_2, gold_ERF_df_2,
                              start_year= START_DATE, baseline=BASELINE1)
fair_gold_3 = test_CO2_Fair(aviation_df_3, CDR_gold_df_3, ERF_df_3, gold_baseline_df_3, gold_ERF_df_3,
                              start_year= START_DATE, baseline=BASELINE1)

# CO2 only CDR scenarios
# Bronze A
fair_bronze_CO2only_1 = test_CO2_Fair(aviation_df_1, CDR_bronze_CO2only_df_1, ERF_df_1, bronze_baseline_df_1, bronze_ERF_df_1,
                                        start_year= START_DATE, baseline=BASELINE1)
fair_bronze_CO2only_2 = test_CO2_Fair(aviation_df_2, CDR_bronze_CO2only_df_2, ERF_df_2, bronze_baseline_df_2, bronze_ERF_df_2,
                                        start_year= START_DATE, baseline=BASELINE1)
fair_bronze_CO2only_3 = test_CO2_Fair(aviation_df_3, CDR_bronze_CO2only_df_3, ERF_df_3, bronze_baseline_df_3, bronze_ERF_df_3,
                                        start_year= START_DATE, baseline=BASELINE1)
# Silver A
fair_silver_CO2only_1 = test_CO2_Fair(aviation_df_1, CDR_silver_CO2only_df_1, ERF_df_1, aviation_df_ref, ERF_df_ref,
                                        start_year= START_DATE_REF, baseline=REF_SCENARIO)
fair_silver_CO2only_2 = test_CO2_Fair(aviation_df_2, CDR_silver_CO2only_df_2, ERF_df_2, aviation_df_ref, ERF_df_ref,
                                        start_year= START_DATE_REF, baseline=REF_SCENARIO)
fair_silver_CO2only_3 = test_CO2_Fair(aviation_df_2, CDR_silver_CO2only_df_2, ERF_df_2, aviation_df_ref, ERF_df_ref,
                                        start_year= START_DATE_REF, baseline=REF_SCENARIO)
# Gold A
fair_gold_CO2only_1 = test_CO2_Fair(aviation_df_1, CDR_gold_CO2only_df_1, ERF_df_1, gold_baseline_df_1, gold_ERF_df_1,
                                      start_year= START_DATE, baseline=BASELINE1)
fair_gold_CO2only_2 = test_CO2_Fair(aviation_df_2, CDR_gold_CO2only_df_2, ERF_df_2, gold_baseline_df_2, gold_ERF_df_2,
                                      start_year= START_DATE, baseline=BASELINE1)
fair_gold_CO2only_3 = test_CO2_Fair(aviation_df_3, CDR_gold_CO2only_df_3, ERF_df_3, gold_baseline_df_3, gold_ERF_df_3,
                                      start_year= START_DATE, baseline=BASELINE1)


#======================================== CREATE NEW TECH SCENARIOS ==================================================
# calculate emission pathways under technology 1 (SAFs)
aviation_tech1_df_1 = make_emissions_new_tech(aviation_df_1, 2020, UPTAKE_TYPE, TECH_1, DATE_TECH1) # in this scenario: start of tech1 in 2020
aviation_tech1_df_2 = make_emissions_new_tech(aviation_df_2, 2020, UPTAKE_TYPE , TECH_1, DATE_TECH1)
aviation_tech1_df_3 = make_emissions_new_tech(aviation_df_3, 2020, UPTAKE_TYPE , TECH_1, DATE_TECH1)
# calculate emission pathways under technology 2 (E-airplanes)
aviation_tech2_df_1= make_emissions_new_tech(aviation_df_1, 2030, UPTAKE_TYPE , TECH_2, DATE_TECH2) # in this scenario: start of tech2 in 2040
aviation_tech2_df_2 = make_emissions_new_tech(aviation_df_2, 2030, UPTAKE_TYPE , TECH_2, DATE_TECH2)
aviation_tech2_df_3 = make_emissions_new_tech(aviation_df_3, 2030, UPTAKE_TYPE , TECH_2, DATE_TECH2)

# calculate Effective Radiative Forcing under emissions from technology 1 (SAFs)
ERF_tech1_df_1 = calculate_ERF_techs(aviation_tech1_df_1, ERF_factors)
ERF_tech1_df_2 = calculate_ERF_techs(aviation_tech1_df_2, ERF_factors)
ERF_tech1_df_3 = calculate_ERF_techs(aviation_tech1_df_3, ERF_factors)
# calculate Effective Radiative Forcing under emissions from technology 2 (E-airplanes)
ERF_tech2_df_1 = calculate_ERF_techs(aviation_tech2_df_1, ERF_factors)
ERF_tech2_df_2 = calculate_ERF_techs(aviation_tech2_df_2, ERF_factors)
ERF_tech2_df_3 = calculate_ERF_techs(aviation_tech2_df_3, ERF_factors)

#============================ CALCULATE CDR - SAFs ========================================================
# make baseline BRONZE
bronze_baseline_tech1_df_1, bronze_baseline_tech1_df_2, bronze_baseline_tech1_df_3, bronze_ERF_tech1_df_1, bronze_ERF_tech1_df_2, bronze_ERF_tech1_df_3 = \
    make_target_scenarios(aviation_tech1_df_1.best, aviation_tech1_df_2.best, aviation_tech1_df_3.best, ERF_factors,
                          START_DATE, BASELINE3)
# calculate CDR for BRONZE
CDR_bronze_tech1_df_1, CDR_bronze_tech1_df_2, CDR_bronze_tech1_df_3 = make_CDR_scenarios(
    ERF_tech1_df_1-bronze_ERF_tech1_df_1, ERF_tech1_df_2-bronze_ERF_tech1_df_2, ERF_tech1_df_3-bronze_ERF_tech1_df_3,
    aviation_tech1_df_1.best-bronze_baseline_tech1_df_1, aviation_tech1_df_2.best-bronze_baseline_tech1_df_2,
    aviation_tech1_df_3.best-bronze_baseline_tech1_df_3, dt = DELTA_T, start_date=START_DATE)

# BRONZE CO2only
CDR_bronze_CO2only_tech1_df_1, CDR_bronze_CO2only_tech1_df_2, CDR_bronze_CO2only_tech1_df_3 = make_CDR_scenarios(
    ERF_tech1_df_1-bronze_ERF_tech1_df_1, ERF_tech1_df_2-bronze_ERF_tech1_df_2, ERF_tech1_df_3-bronze_ERF_tech1_df_3,
    aviation_tech1_df_1.best-bronze_baseline_tech1_df_1, aviation_tech1_df_2.best-bronze_baseline_tech1_df_2, aviation_tech1_df_3.best-bronze_baseline_tech1_df_3,
    dt = DELTA_T, start_date=START_DATE, CO2only=True)

# calculate CDR for SILVER
CDR_silver_tech1_df_1, CDR_silver_tech1_df_2, CDR_silver_tech1_df_3 = make_CDR_scenarios(
    ERF_tech1_df_1-ERF_df_ref, ERF_tech1_df_2-ERF_df_ref, ERF_tech1_df_3-ERF_df_ref,
    aviation_tech1_df_1.best-aviation_df_ref, aviation_tech1_df_2.best-aviation_df_ref,
    aviation_tech1_df_3.best-aviation_df_ref, dt = DELTA_T, start_date=START_DATE_REF)

# make baseline GOLD
gold_baseline_tech1_df_1, gold_baseline_tech1_df_2, gold_baseline_tech1_df_3, gold_ERF_tech1_df_1, gold_ERF_tech1_df_2, gold_ERF_tech1_df_3 = \
    make_target_scenarios(aviation_tech1_df_1.best, aviation_tech1_df_2.best, aviation_tech1_df_3.best, ERF_factors,
                          START_DATE, BASELINE1, intermediate_goal=True)
# calculate CDR for GOLD
CDR_gold_tech1_df_1, CDR_gold_tech1_df_2, CDR_gold_tech1_df_3 = make_CDR_scenarios(
    ERF_tech1_df_1-gold_ERF_tech1_df_1, ERF_tech1_df_2-gold_ERF_tech1_df_2, ERF_tech1_df_3-gold_ERF_tech1_df_3,
    aviation_tech1_df_1.best-gold_baseline_tech1_df_1, aviation_tech1_df_2.best-gold_baseline_tech1_df_2,
    aviation_tech1_df_3.best-gold_baseline_tech1_df_3, dt = DELTA_T, start_date=START_DATE)


#============================ CALCULATE CDR - E_airplanes ===============================================
# make baseline BRONZE
bronze_baseline_tech2_df_1, bronze_baseline_tech2_df_2, bronze_baseline_tech2_df_3, bronze_ERF_tech2_df_1, bronze_ERF_tech2_df_2, bronze_ERF_tech2_df_3 = \
    make_target_scenarios(aviation_tech2_df_1.best, aviation_tech2_df_2.best, aviation_tech2_df_3.best,
                          ERF_factors, START_DATE, BASELINE3)
# calculate CDR BRONZE
CDR_bronze_tech2_df_1, CDR_bronze_tech2_df_2, CDR_bronze_tech2_df_3 = make_CDR_scenarios(
    ERF_tech2_df_1-bronze_ERF_tech2_df_1, ERF_tech2_df_2-bronze_ERF_tech2_df_2, ERF_tech2_df_3-bronze_ERF_tech2_df_3,
    aviation_tech2_df_1.best-bronze_baseline_tech2_df_1, aviation_tech2_df_2.best-bronze_baseline_tech2_df_2,
    aviation_tech2_df_3.best-bronze_baseline_tech2_df_3, dt = DELTA_T, start_date=START_DATE)

# BRONZE
CDR_bronze_CO2only_tech2_df_1, CDR_bronze_CO2only_tech2_df_2, CDR_bronze_CO2only_tech2_df_3 = make_CDR_scenarios(
    ERF_tech2_df_1-bronze_ERF_tech2_df_1, ERF_tech2_df_2-bronze_ERF_tech2_df_2, ERF_tech2_df_3-bronze_ERF_tech2_df_3,
    aviation_tech2_df_1.best-bronze_baseline_tech2_df_1, aviation_tech2_df_2.best-bronze_baseline_tech2_df_2, aviation_tech2_df_3.best-bronze_baseline_tech2_df_3,
    dt = DELTA_T, start_date=START_DATE, CO2only=True)

# calculate CDR SILVER
CDR_silver_tech2_df_1, CDR_silver_tech2_df_2, CDR_silver_tech2_df_3 = make_CDR_scenarios(
    ERF_tech2_df_1-ERF_df_ref, ERF_tech2_df_2-ERF_df_ref, ERF_tech2_df_3-ERF_df_ref,
    aviation_tech2_df_1.best-aviation_df_ref, aviation_tech2_df_2.best-aviation_df_ref,
    aviation_tech2_df_3.best-aviation_df_ref, dt = DELTA_T, start_date=START_DATE_REF)

# make baseline GOLD
gold_baseline_tech2_df_1, gold_baseline_tech2_df_2, gold_baseline_tech2_df_3, gold_ERF_tech2_df_1, gold_ERF_tech2_df_2, gold_ERF_tech2_df_3 = \
    make_target_scenarios(aviation_tech2_df_1.best, aviation_tech2_df_2.best, aviation_tech2_df_3.best,
                          ERF_factors, START_DATE, BASELINE1, intermediate_goal=True)
# calculate CDR GOLD
CDR_gold_tech2_df_1, CDR_gold_tech2_df_2, CDR_gold_tech2_df_3 = make_CDR_scenarios(
    ERF_tech2_df_1-gold_ERF_tech2_df_1, ERF_tech2_df_2-gold_ERF_tech2_df_2, ERF_tech2_df_3-gold_ERF_tech2_df_3,
    aviation_tech2_df_1.best-gold_baseline_tech2_df_1, aviation_tech2_df_2.best-gold_baseline_tech2_df_2,
    aviation_tech2_df_3.best-gold_baseline_tech2_df_3, dt = DELTA_T, start_date=START_DATE)


#============================ COMPUTE SCENARIOS' ERF AND T WITH FAIR =============================
#================================ SAFs ==================================
# Bronze
fair_bronze_tech1_1 = test_CO2_Fair(aviation_tech1_df_1.best, CDR_bronze_tech1_df_1, ERF_tech1_df_1, bronze_baseline_tech1_df_1, bronze_ERF_tech1_df_1,
                                start_year= START_DATE, baseline=BASELINE1)
fair_bronze_tech1_2 = test_CO2_Fair(aviation_tech1_df_2.best, CDR_bronze_tech1_df_2, ERF_tech1_df_2, bronze_baseline_tech1_df_2, bronze_ERF_tech1_df_2,
                                start_year= START_DATE, baseline=BASELINE1)
fair_bronze_tech1_3 = test_CO2_Fair(aviation_tech1_df_3.best, CDR_bronze_tech1_df_3, ERF_tech1_df_3, bronze_baseline_tech1_df_3, bronze_ERF_tech1_df_3,
                                start_year= START_DATE, baseline=BASELINE1)

# Silver
fair_silver_tech1_1 = test_CO2_Fair(aviation_tech1_df_1.best, CDR_silver_tech1_df_1, ERF_tech1_df_1, aviation_df_ref, ERF_df_ref,
                                      start_year= START_DATE_REF, baseline=REF_SCENARIO)
fair_silver_tech1_2 = test_CO2_Fair(aviation_tech1_df_2.best, CDR_silver_tech1_df_2, ERF_tech1_df_2, aviation_df_ref, ERF_df_ref,
                                      start_year= START_DATE_REF, baseline=REF_SCENARIO)
fair_silver_tech1_3 = test_CO2_Fair(aviation_tech1_df_2.best, CDR_silver_tech1_df_2, ERF_tech1_df_2, aviation_df_ref, ERF_df_ref,
                                      start_year= START_DATE_REF, baseline=REF_SCENARIO)

# Gold
fair_gold_tech1_1 = test_CO2_Fair(aviation_tech1_df_1.best, CDR_gold_tech1_df_1, ERF_tech1_df_1, gold_baseline_tech1_df_1, gold_ERF_tech1_df_1,
                                start_year= START_DATE, baseline=BASELINE1)
fair_gold_tech1_2 = test_CO2_Fair(aviation_tech1_df_2.best, CDR_gold_tech1_df_2, ERF_tech1_df_2, gold_baseline_tech1_df_2, gold_ERF_tech1_df_2,
                                start_year= START_DATE, baseline=BASELINE1)
fair_gold_tech1_3 = test_CO2_Fair(aviation_tech1_df_3.best, CDR_gold_tech1_df_3, ERF_tech1_df_3, gold_baseline_tech1_df_3, gold_ERF_tech1_df_3,
                                start_year= START_DATE, baseline=BASELINE1)


#================================ E-airplanes ==================================
# Bronze
fair_bronze_tech2_1 = test_CO2_Fair(aviation_tech2_df_1.best, CDR_bronze_tech2_df_1, ERF_tech2_df_1, bronze_baseline_tech2_df_1, bronze_ERF_tech2_df_1,
                                start_year= START_DATE, baseline=BASELINE1)
fair_bronze_tech2_2 = test_CO2_Fair(aviation_tech2_df_2.best, CDR_bronze_tech2_df_2, ERF_tech2_df_2, bronze_baseline_tech2_df_2, bronze_ERF_tech2_df_2,
                                start_year= START_DATE, baseline=BASELINE1)
fair_bronze_tech2_3 = test_CO2_Fair(aviation_tech2_df_3.best, CDR_bronze_tech2_df_3, ERF_tech2_df_3, bronze_baseline_tech2_df_3, bronze_ERF_tech2_df_3,
                                start_year= START_DATE, baseline=BASELINE1)

# Silver
fair_silver_tech2_1 = test_CO2_Fair(aviation_tech2_df_1.best, CDR_silver_tech2_df_1, ERF_tech2_df_1, aviation_df_ref, ERF_df_ref,
                                      start_year= START_DATE_REF, baseline=REF_SCENARIO)
fair_silver_tech2_2 = test_CO2_Fair(aviation_tech2_df_2.best, CDR_silver_tech2_df_2, ERF_tech2_df_2, aviation_df_ref, ERF_df_ref,
                                      start_year= START_DATE_REF, baseline=REF_SCENARIO)
fair_silver_tech2_3 = test_CO2_Fair(aviation_tech2_df_2.best, CDR_silver_tech2_df_2, ERF_tech2_df_2, aviation_df_ref, ERF_df_ref,
                                      start_year= START_DATE_REF, baseline=REF_SCENARIO)

# Gold
fair_gold_tech2_1 = test_CO2_Fair(aviation_tech2_df_1.best, CDR_gold_tech2_df_1, ERF_tech2_df_1, gold_baseline_tech2_df_1, gold_ERF_tech2_df_1,
                                start_year= START_DATE, baseline=BASELINE1)
fair_gold_tech2_2 = test_CO2_Fair(aviation_tech2_df_2.best, CDR_gold_tech2_df_2, ERF_tech2_df_2, gold_baseline_tech2_df_2, gold_ERF_tech2_df_2,
                                start_year= START_DATE, baseline=BASELINE1)
fair_gold_tech2_3 = test_CO2_Fair(aviation_tech2_df_3.best, CDR_gold_tech2_df_3, ERF_tech2_df_3, gold_baseline_tech2_df_3, gold_ERF_tech2_df_3,
                                start_year= START_DATE, baseline=BASELINE1)

#=============================== CO2 only =====================================
# CO2 only CDR scenarios
# Bronze Tech 1
fair_bronze_CO2only_tech1_1 = test_CO2_Fair(aviation_tech1_df_1.best, CDR_bronze_CO2only_tech1_df_1, ERF_tech1_df_1, bronze_baseline_tech1_df_1, bronze_ERF_tech1_df_1,
                                        start_year= START_DATE, baseline=BASELINE1)
fair_bronze_CO2only_tech1_2 = test_CO2_Fair(aviation_tech1_df_2.best, CDR_bronze_CO2only_tech1_df_2, ERF_tech1_df_2, bronze_baseline_tech1_df_2, bronze_ERF_tech1_df_2,
                                        start_year= START_DATE, baseline=BASELINE1)
fair_bronze_CO2only_tech1_3 = test_CO2_Fair(aviation_tech1_df_3.best, CDR_bronze_CO2only_tech1_df_3, ERF_tech1_df_3, bronze_baseline_tech1_df_3, bronze_ERF_tech1_df_3,
                                        start_year= START_DATE, baseline=BASELINE1)

# Bronze Tech 2
fair_bronze_CO2only_tech2_1 = test_CO2_Fair(aviation_tech2_df_1.best, CDR_bronze_CO2only_tech2_df_1, ERF_tech2_df_1, bronze_baseline_tech2_df_1, bronze_ERF_tech2_df_1,
                                        start_year= START_DATE, baseline=BASELINE1)
fair_bronze_CO2only_tech2_2 = test_CO2_Fair(aviation_tech2_df_2.best, CDR_bronze_CO2only_tech2_df_2, ERF_tech2_df_2, bronze_baseline_tech2_df_2, bronze_ERF_tech2_df_2,
                                        start_year= START_DATE, baseline=BASELINE1)
fair_bronze_CO2only_tech2_3 = test_CO2_Fair(aviation_tech2_df_3.best, CDR_bronze_CO2only_tech2_df_3, ERF_tech2_df_3, bronze_baseline_tech2_df_3, bronze_ERF_tech2_df_3,
                                        start_year= START_DATE, baseline=BASELINE1)

#=============================== CALCULATE COST PER KM =======================================
# Jet A1
CDRkm_gold_df_1, costkm_gold_df_1 = calculate_cost_perkm(aviation_df_1, CDR_gold_df_1, CDR_COST)
CDRkm_gold_df_2, costkm_gold_df_2 = calculate_cost_perkm(aviation_df_2, CDR_gold_df_2, CDR_COST)
CDRkm_gold_df_3, costkm_gold_df_3 = calculate_cost_perkm(aviation_df_3, CDR_gold_df_3, CDR_COST)
CDRkm_silver_df_1, costkm_silver_df_1 = calculate_cost_perkm(aviation_df_1, CDR_silver_df_1, CDR_COST)
CDRkm_silver_df_2, costkm_silver_df_2 = calculate_cost_perkm(aviation_df_2, CDR_silver_df_2, CDR_COST)
CDRkm_silver_df_3, costkm_silver_df_3 = calculate_cost_perkm(aviation_df_3, CDR_silver_df_3, CDR_COST)
CDRkm_bronze_df_1, costkm_bronze_df_1 = calculate_cost_perkm(aviation_df_1, CDR_bronze_df_1, CDR_COST)
CDRkm_bronze_df_2, costkm_bronze_df_2 = calculate_cost_perkm(aviation_df_2, CDR_bronze_df_2, CDR_COST)
CDRkm_bronze_df_3, costkm_bronze_df_3 = calculate_cost_perkm(aviation_df_3, CDR_bronze_df_3, CDR_COST)
# SAFs
CDRkm_gold_tech1_df_1, costkm_gold_tech1_df_1 = calculate_cost_perkm(aviation_df_1, CDR_gold_tech1_df_1, CDR_COST)
CDRkm_gold_tech1_df_2, costkm_gold_tech1_df_2 = calculate_cost_perkm(aviation_df_2, CDR_gold_tech1_df_2, CDR_COST)
CDRkm_gold_tech1_df_3, costkm_gold_tech1_df_3 = calculate_cost_perkm(aviation_df_3, CDR_gold_tech1_df_3, CDR_COST)
CDRkm_silver_tech1_df_1, costkm_silver_tech1_df_1 = calculate_cost_perkm(aviation_df_1, CDR_silver_tech1_df_1, CDR_COST)
CDRkm_silver_tech1_df_2, costkm_silver_tech1_df_2 = calculate_cost_perkm(aviation_df_2, CDR_silver_tech1_df_2, CDR_COST)
CDRkm_silver_tech1_df_3, costkm_silver_tech1_df_3 = calculate_cost_perkm(aviation_df_3, CDR_silver_tech1_df_3, CDR_COST)
CDRkm_bronze_tech1_df_1, costkm_bronze_tech1_df_1 = calculate_cost_perkm(aviation_df_1, CDR_bronze_tech1_df_1, CDR_COST)
CDRkm_bronze_tech1_df_2, costkm_bronze_tech1_df_2 = calculate_cost_perkm(aviation_df_2, CDR_bronze_tech1_df_2, CDR_COST)
CDRkm_bronze_tech1_df_3, costkm_bronze_tech1_df_3 = calculate_cost_perkm(aviation_df_3, CDR_bronze_tech1_df_3, CDR_COST)
# E-airplanes
CDRkm_gold_tech2_df_1, costkm_gold_tech2_df_1 = calculate_cost_perkm(aviation_df_1, CDR_gold_tech2_df_1, CDR_COST)
CDRkm_gold_tech2_df_2, costkm_gold_tech2_df_2 = calculate_cost_perkm(aviation_df_2, CDR_gold_tech2_df_2, CDR_COST)
CDRkm_gold_tech2_df_3, costkm_gold_tech2_df_3 = calculate_cost_perkm(aviation_df_3, CDR_gold_tech2_df_3, CDR_COST)
CDRkm_silver_tech2_df_1, costkm_silver_tech2_df_1 = calculate_cost_perkm(aviation_df_1, CDR_silver_tech2_df_1, CDR_COST)
CDRkm_silver_tech2_df_2, costkm_silver_tech2_df_2 = calculate_cost_perkm(aviation_df_2, CDR_silver_tech2_df_2, CDR_COST)
CDRkm_silver_tech2_df_3, costkm_silver_tech2_df_3 = calculate_cost_perkm(aviation_df_3, CDR_silver_tech2_df_3, CDR_COST)
CDRkm_bronze_tech2_df_1, costkm_bronze_tech2_df_1 = calculate_cost_perkm(aviation_df_1, CDR_bronze_tech2_df_1, CDR_COST)
CDRkm_bronze_tech2_df_2, costkm_bronze_tech2_df_2 = calculate_cost_perkm(aviation_df_2, CDR_bronze_tech2_df_2, CDR_COST)
CDRkm_bronze_tech2_df_3, costkm_bronze_tech2_df_3 = calculate_cost_perkm(aviation_df_3, CDR_bronze_tech2_df_3, CDR_COST)

#==================================== ANALYSIS OF DIFFERENT METRICS =============================
# CALCULATE CDR with Emission Factor = 2
E_EWF_df_1, E_EWF_df_2, E_EWF_df_3 = make_CDR_scenarios(
    ERF_df_1, ERF_df_2, ERF_df_3,
    aviation_df_ref, aviation_df_2, aviation_df_3,
    dt = DELTA_T, start_date=2010, metric='EWF', EF = 2)
E_EWF_tech1_df_1, E_EWF_tech1_df_2, E_EWF_tech1_df_3 = make_CDR_scenarios(
    ERF_tech1_df_1, ERF_tech1_df_2, ERF_tech1_df_3,
    aviation_tech1_df_1.best, aviation_tech1_df_2.best, aviation_tech1_df_3.best,
    dt = DELTA_T, start_date=2010, metric='EWF', EF = 2)
E_EWF_tech2_df_1, E_EWF_tech2_df_2, E_EWF_tech2_df_3 = make_CDR_scenarios(
    ERF_tech2_df_1, ERF_tech2_df_2, ERF_tech2_df_3,
    aviation_tech2_df_1.best, aviation_tech2_df_2.best, aviation_tech2_df_3.best,
    dt = DELTA_T, start_date=2010, metric='EWF', EF = 2)


# CALCULATE CDR with GWP100
E_GWP100_df_1, E_GWP100_df_2, E_GWP100_df_3 = make_CDR_scenarios(
    ERF_df_1, ERF_df_2, ERF_df_3,
    aviation_df_1, aviation_df_2, aviation_df_3,
    dt = DELTA_T, start_date=2010, metric='GWP100', EF = None)
E_GWP100_tech1_df_1, E_GWP100_tech1_df_2, E_GWP100_tech1_df_3 = make_CDR_scenarios(
    ERF_tech1_df_1, ERF_tech1_df_2, ERF_tech1_df_3,
    aviation_tech1_df_1.best, aviation_tech1_df_2.best,aviation_tech1_df_3.best,
    dt = DELTA_T, start_date=2010, metric='GWP100', EF = 2)
E_GWP100_tech2_df_1, E_GWP100_tech2_df_2, E_GWP100_tech2_df_3 = make_CDR_scenarios(
    ERF_tech2_df_1, ERF_tech2_df_2, ERF_tech2_df_3,
    aviation_tech2_df_1.best, aviation_tech2_df_2.best, aviation_tech2_df_3.best,
    dt = DELTA_T, start_date=2010, metric='GWP100', EF = 2)

#CALCULATE CDR with GWP*
E_GWPstar_df_1, E_GWPstar_df_2, E_GWPstar_df_3 = make_CDR_scenarios(
    ERF_df_1, ERF_df_2, ERF_df_3,
    aviation_df_1, aviation_df_2, aviation_df_3,
    dt = DELTA_T, start_date=2010, metric='GWP*', EF = None)
E_GWPstar_tech1_df_1, E_GWPstar_tech1_df_2, E_GWPstar_tech1_df_3 = make_CDR_scenarios(
    ERF_tech1_df_1, ERF_tech1_df_2, ERF_tech1_df_3,
    aviation_tech1_df_1.best, aviation_tech1_df_2.best,  aviation_tech1_df_3.best,
    dt = DELTA_T, start_date=2010, metric='GWP*', EF = 2)
E_GWPstar_tech2_df_1, E_GWPstar_tech2_df_2, E_GWPstar_tech2_df_3 = make_CDR_scenarios(
    ERF_tech2_df_1, ERF_tech2_df_2, ERF_tech2_df_3,
    aviation_tech2_df_1.best, aviation_tech2_df_2.best, aviation_tech2_df_3.best,
    dt = DELTA_T, start_date=2010, metric='GWP*', EF = 2)

# Make baseline scenarios of ERF for different emissions
ERF_base_df_1 = ERF_df_1.copy()
ERF_base_df_2 = ERF_df_2.copy()
ERF_base_df_3 = ERF_df_3.copy()
ERF_base_tech1_df_1 = ERF_tech1_df_1.copy()
ERF_base_tech1_df_2 = ERF_tech1_df_2.copy()
ERF_base_tech1_df_3 = ERF_tech1_df_3.copy()
ERF_base_tech2_df_1 = ERF_tech2_df_1.copy()
ERF_base_tech2_df_2 = ERF_tech2_df_2.copy()
ERF_base_tech2_df_3 = ERF_tech2_df_3.copy()
# Make ERF = 0 before 2009
ERF_base_df_1[:'2009'] = 0.
ERF_base_df_2[:'2009'] = 0.
ERF_base_df_3[:'2009'] = 0.
ERF_base_tech1_df_1[:'2009'] = 0.
ERF_base_tech1_df_2[:'2009'] = 0.
ERF_base_tech1_df_3[:'2009'] = 0.
ERF_base_tech2_df_1[:'2009'] = 0.
ERF_base_tech2_df_2[:'2009'] = 0.
ERF_base_tech2_df_3[:'2009'] = 0.

# Calculate ERF and temperature outcomes for non-CO2 emissions calculated with different conversion metrics
fair_base_1, fair_base_2, fair_base_3 = make_fair_scenarios(aviation_df_1, aviation_df_2, aviation_df_3,
                                  ERF_base_df_1, ERF_base_df_2, ERF_base_df_3, start_year=1990)
fair_base_tech1_1, fair_base_tech1_2, fair_base_tech1_3 = make_fair_scenarios(aviation_tech1_df_1.best,
                                                                              aviation_tech1_df_2.best,
                                                                              aviation_tech1_df_3.best,
                                  ERF_base_tech1_df_1, ERF_base_tech1_df_2, ERF_base_tech1_df_3, start_year=1990)
fair_base_tech2_1, fair_base_tech2_2, fair_base_tech2_3 = make_fair_scenarios(aviation_tech2_df_1.best,
                                                                              aviation_tech2_df_2.best,
                                                                              aviation_tech2_df_3.best,
                                  ERF_base_tech2_df_1, ERF_base_tech2_df_2, ERF_base_tech2_df_3, start_year=1990)
fair_EWF_1, fair_EWF_2, fair_EWF_3 = make_fair_scenarios(aviation_df_ref, aviation_df_2, aviation_df_3,
                                                         E_CO2eq_1=E_EWF_df_1, E_CO2eq_2=E_EWF_df_2,
                                                         E_CO2eq_3=E_EWF_df_3, start_year= 1990)
fair_EWF_tech1_1, fair_EWF_tech1_2, fair_EWF_tech1_3 = make_fair_scenarios(aviation_tech1_df_1.best,
                                                                           aviation_tech1_df_2.best,
                                                                           aviation_tech1_df_3.best,
                                                                              E_CO2eq_1=E_EWF_tech1_df_1,
                                                                              E_CO2eq_2=E_EWF_tech1_df_2,
                                                                              E_CO2eq_3=E_EWF_tech1_df_3, start_year=1990)
fair_EWF_tech2_1, fair_EWF_tech2_2, fair_EWF_tech2_3 = make_fair_scenarios(aviation_tech2_df_1.best,
                                                                           aviation_tech2_df_2.best,
                                                                           aviation_tech2_df_3.best,
                                                                              E_CO2eq_1=E_EWF_tech2_df_1,
                                                                              E_CO2eq_2=E_EWF_tech2_df_2,
                                                                              E_CO2eq_3=E_EWF_tech2_df_3, start_year=1990)
fair_GWP100_1, fair_GWP100_2, fair_GWP100_3 = make_fair_scenarios(aviation_df_ref, aviation_df_2, aviation_df_3,
                                                                  E_CO2eq_1=E_GWP100_df_1, E_CO2eq_2=E_GWP100_df_2,
                                                                  E_CO2eq_3=E_GWP100_df_3, start_year=1990)
fair_GWP100_tech1_1, fair_GWP100_tech1_2, fair_GWP100_tech1_3 = make_fair_scenarios(aviation_tech1_df_1.best,
                                                                                    aviation_tech1_df_2.best,
                                                                                    aviation_tech1_df_3.best,
                                                                              E_CO2eq_1=E_GWP100_tech1_df_1,
                                                                              E_CO2eq_2=E_GWP100_tech1_df_2,
                                                                              E_CO2eq_3=E_GWP100_tech1_df_3,
                                                                              start_year=1990)
fair_GWP100_tech2_1, fair_GWP100_tech2_2, fair_GWP100_tech2_3 = make_fair_scenarios(aviation_tech2_df_1.best,
                                                                                    aviation_tech2_df_2.best,
                                                                                    aviation_tech2_df_3.best,
                                                                              E_CO2eq_1=E_GWP100_tech2_df_1,
                                                                              E_CO2eq_2=E_GWP100_tech2_df_2,
                                                                              E_CO2eq_3=E_GWP100_tech2_df_3,
                                                                              start_year=1990)
fair_GWPstar_1, fair_GWPstar_2, fair_GWPstar_3 = make_fair_scenarios(aviation_df_ref, aviation_df_2, aviation_df_3,
                                                                     E_CO2eq_1=E_GWPstar_df_1, E_CO2eq_2=E_GWPstar_df_2,
                                                                     E_CO2eq_3=E_GWPstar_df_3, start_year=1990,
                                                                     what = 'GWP*')
fair_GWPstar_tech1_1, fair_GWPstar_tech1_2, fair_GWPstar_tech1_3 = make_fair_scenarios(aviation_tech1_df_1.best,
                                                                                       aviation_tech1_df_2.best,
                                                                                       aviation_tech1_df_3.best,
                                                                              E_CO2eq_1=E_GWPstar_tech1_df_1,
                                                                              E_CO2eq_2=E_GWPstar_tech1_df_2,
                                                                              E_CO2eq_3=E_GWPstar_tech1_df_3,
                                                                              start_year=1990,
                                                                              what='GWP*')
fair_GWPstar_tech2_1, fair_GWPstar_tech2_2, fair_GWPstar_tech2_3 = make_fair_scenarios(aviation_tech2_df_1.best,
                                                                                       aviation_tech2_df_2.best,
                                                                                       aviation_tech2_df_3.best,
                                                                              E_CO2eq_1=E_GWPstar_tech2_df_1,
                                                                              E_CO2eq_2=E_GWPstar_tech2_df_2,
                                                                              E_CO2eq_3=E_GWPstar_tech2_df_3,
                                                                              start_year=1990,
                                                                              what='GWP*')


#================================================================================================
# ============================ MAKE SUMMARY DATAFRAMES FOR VISUALIZATIONs AND EXPORTs ===========
# summary CDR and emissions
meanCDR_summary_alltechs = make_bar_CDR_alltechs_summary(CDR_gold_df_1, CDR_gold_df_2, CDR_gold_df_3,
                                                         CDR_silver_df_1, CDR_silver_df_2, CDR_silver_df_3,
                                                         CDR_bronze_df_1, CDR_bronze_df_2, CDR_bronze_df_3,
                                                         CDR_gold_tech1_df_1, CDR_gold_tech1_df_2,
                                                         CDR_gold_tech1_df_3,
                                                         CDR_silver_tech1_df_1, CDR_silver_tech1_df_2,
                                                         CDR_silver_tech1_df_3,
                                                         CDR_bronze_tech1_df_1, CDR_bronze_tech1_df_2,
                                                         CDR_bronze_tech1_df_3, CDR_gold_tech2_df_1,
                                                         CDR_gold_tech2_df_2, CDR_gold_tech2_df_3,
                                                         CDR_silver_tech2_df_1, CDR_silver_tech2_df_2,
                                                         CDR_silver_tech2_df_3,
                                                         CDR_bronze_tech2_df_1, CDR_bronze_tech2_df_2,
                                                         CDR_bronze_tech2_df_3, SCENARIO1, SCENARIO2, SCENARIO3)
cumulativeCDR_summary_alltechs = make_bar_CDR_alltechs_summary(CDR_gold_df_1, CDR_gold_df_2, CDR_gold_df_3,
                                                               CDR_silver_df_1, CDR_silver_df_2,
                                                               CDR_silver_df_3,
                                                               CDR_bronze_df_1, CDR_bronze_df_2,
                                                               CDR_bronze_df_3, CDR_gold_tech1_df_1,
                                                               CDR_gold_tech1_df_2, CDR_gold_tech1_df_3,
                                                               CDR_silver_tech1_df_1, CDR_silver_tech1_df_2,
                                                               CDR_silver_tech1_df_3,
                                                               CDR_bronze_tech1_df_1, CDR_bronze_tech1_df_2,
                                                               CDR_bronze_tech1_df_3, CDR_gold_tech2_df_1,
                                                               CDR_gold_tech2_df_2, CDR_gold_tech2_df_3,
                                                               CDR_silver_tech2_df_1, CDR_silver_tech2_df_2,
                                                               CDR_silver_tech2_df_3,
                                                               CDR_bronze_tech2_df_1, CDR_bronze_tech2_df_2,
                                                               CDR_bronze_tech2_df_3, SCENARIO1, SCENARIO2,
                                                               SCENARIO3,
                                                               what='cumulative')
positive_cumulativeCDR_summary_alltechs = make_bar_CDR_alltechs_summary(CDR_gold_df_1, CDR_gold_df_2, CDR_gold_df_3,
                                                                        CDR_silver_df_1, CDR_silver_df_2,
                                                                        CDR_silver_df_3,
                                                                        CDR_bronze_df_1, CDR_bronze_df_2,
                                                                        CDR_bronze_df_3, CDR_gold_tech1_df_1,
                                                                        CDR_gold_tech1_df_2, CDR_gold_tech1_df_3,
                                                                        CDR_silver_tech1_df_1, CDR_silver_tech1_df_2,
                                                                        CDR_silver_tech1_df_3,
                                                                        CDR_bronze_tech1_df_1, CDR_bronze_tech1_df_2,
                                                                        CDR_bronze_tech1_df_3, CDR_gold_tech2_df_1,
                                                                        CDR_gold_tech2_df_2, CDR_gold_tech2_df_3,
                                                                        CDR_silver_tech2_df_1, CDR_silver_tech2_df_2,
                                                                        CDR_silver_tech2_df_3,
                                                                        CDR_bronze_tech2_df_1, CDR_bronze_tech2_df_2,
                                                                        CDR_bronze_tech2_df_3, SCENARIO1, SCENARIO2,
                                                                        SCENARIO3, sign = 'positive',
                                                                        what='cumulative')
negative_cumulativeCDR_summary_alltechs = make_bar_CDR_alltechs_summary(CDR_gold_df_1, CDR_gold_df_2, CDR_gold_df_3,
                                                                        CDR_silver_df_1, CDR_silver_df_2,
                                                                        CDR_silver_df_3,
                                                                        CDR_bronze_df_1, CDR_bronze_df_2,
                                                                        CDR_bronze_df_3, CDR_gold_tech1_df_1,
                                                                        CDR_gold_tech1_df_2, CDR_gold_tech1_df_3,
                                                                        CDR_silver_tech1_df_1, CDR_silver_tech1_df_2,
                                                                        CDR_silver_tech1_df_3,
                                                                        CDR_bronze_tech1_df_1, CDR_bronze_tech1_df_2,
                                                                        CDR_bronze_tech1_df_3, CDR_gold_tech2_df_1,
                                                                        CDR_gold_tech2_df_2, CDR_gold_tech2_df_3,
                                                                        CDR_silver_tech2_df_1, CDR_silver_tech2_df_2,
                                                                        CDR_silver_tech2_df_3,
                                                                        CDR_bronze_tech2_df_1, CDR_bronze_tech2_df_2,
                                                                        CDR_bronze_tech2_df_3, SCENARIO1, SCENARIO2,
                                                                        SCENARIO3, sign = 'negative',
                                                                        what='cumulative')

cumulativeEWF_summary_alltechs = make_bar_EWF_alltechs_summary(ERF_df_1, ERF_df_2, ERF_df_3,
                                                               ERF_tech1_df_1, ERF_tech1_df_2, ERF_tech1_df_3,
                                                               ERF_tech2_df_1, ERF_tech2_df_2, ERF_tech2_df_3,
                                                               aviation_df_1, aviation_df_2, aviation_df_3,
                                                               aviation_tech1_df_1, aviation_tech1_df_2, aviation_tech1_df_3,
                                                               aviation_tech2_df_1, aviation_tech2_df_2, aviation_tech2_df_3,
                                                               SCENARIO1, SCENARIO2, SCENARIO3)
# summary Ts
summaryT_alltechs = make_bar_fair_alltechs_summary(fair_gold_1, fair_gold_2, fair_gold_3,
                                                   fair_silver_1, fair_silver_2, fair_silver_3,
                                                   fair_bronze_1, fair_bronze_2, fair_bronze_3,
                                                   fair_gold_tech1_1, fair_gold_tech1_2, fair_gold_tech1_3,
                                                   fair_silver_tech1_1, fair_silver_tech1_2,
                                                   fair_silver_tech1_3,
                                                   fair_bronze_tech1_1, fair_bronze_tech1_2,
                                                   fair_bronze_tech1_3,
                                                   fair_gold_tech2_1, fair_gold_tech2_2, fair_gold_tech2_3,
                                                   fair_silver_tech2_1, fair_silver_tech2_2, fair_silver_tech2_3,
                                                   fair_bronze_tech2_1, fair_bronze_tech2_2, fair_bronze_tech2_3,
                                                   SCENARIO1, SCENARIO2, SCENARIO3,
                                                   what='T')
summary_T_CO2only_df, summary_T_CO2only_err_df = make_summary_T_CO2only(fair_gold_CO2only_1, fair_gold_CO2only_2,
                                                                        fair_gold_CO2only_3,
                                                                        fair_silver_CO2only_1,
                                                                        fair_silver_CO2only_2,
                                                                        fair_silver_CO2only_3,
                                                                        fair_bronze_CO2only_1,
                                                                        fair_bronze_CO2only_2,
                                                                        fair_bronze_CO2only_3,
                                                                        SCENARIO1, SCENARIO2, SCENARIO3)

summary_T_CO2only_alltechs_df, summary_T_CO2only_alltechs_err_df = make_summary_T_CO2only(
    fair_bronze_CO2only_1, fair_bronze_CO2only_2,fair_bronze_CO2only_3,
    fair_bronze_CO2only_tech1_1, fair_bronze_CO2only_tech1_2, fair_bronze_CO2only_tech1_3,
    fair_bronze_CO2only_tech2_1, fair_bronze_CO2only_tech2_2, fair_bronze_CO2only_tech2_3,
    SCENARIO1, SCENARIO2, SCENARIO3, 'Fossil jet fuels', TECH_1, TECH_2)



summary_T_allmetrics = make_bar_T_metrics_summary(fair_base_1, fair_base_2, fair_base_3 ,
                         fair_EWF_1, fair_EWF_2, fair_EWF_3,
                         fair_GWP100_1, fair_GWP100_2, fair_GWP100_3,
                                   fair_GWPstar_1, fair_GWPstar_2, fair_GWPstar_3,
                         scenario1 = 'SSP1-2.6',scenario2 = 'SSP2-4.5',scenario3 = 'SSP5-8.5',
                                                  base = '$\sigma$',
                         metric1 = 'EWF', metric2 = 'GWP100', metric3 = 'GWP*')
summary_T_tech1_allmetrics = make_bar_T_metrics_summary(fair_base_tech1_1, fair_base_tech1_2, fair_base_tech1_3 ,
                         fair_EWF_tech1_1, fair_EWF_tech1_2, fair_EWF_tech1_3,
                         fair_GWP100_tech1_1, fair_GWP100_tech1_2, fair_GWP100_tech1_3,
                                   fair_GWPstar_tech1_1, fair_GWPstar_tech1_2, fair_GWPstar_tech1_3,
                         scenario1 = 'SSP1-2.6',scenario2 = 'SSP2-4.5',scenario3 = 'SSP5-8.5',
                                                  base = '$\sigma$',
                         metric1 = 'EWF', metric2 = 'GWP100', metric3 = 'GWP*')
summary_T_tech2_allmetrics = make_bar_T_metrics_summary(fair_base_tech2_1, fair_base_tech2_2, fair_base_tech2_3 ,
                         fair_EWF_tech2_1, fair_EWF_tech2_2, fair_EWF_tech2_3,
                         fair_GWP100_tech2_1, fair_GWP100_tech2_2, fair_GWP100_tech2_3,
                                   fair_GWPstar_tech2_1, fair_GWPstar_tech2_2, fair_GWPstar_tech2_3,
                         scenario1 = 'SSP1-2.6',scenario2 = 'SSP2-4.5',scenario3 = 'SSP5-8.5',
                                                  base = '$\sigma$',
                         metric1 = 'EWF', metric2 = 'GWP100', metric3 = 'GWP*')


# summary cost per km
mean_costkm_summary_alltechs = make_bar_costkm_alltechs_summary(
    costkm_gold_df_1, costkm_gold_df_2, costkm_gold_df_3,
    costkm_silver_df_1, costkm_silver_df_2, costkm_silver_df_3,
    costkm_bronze_df_1, costkm_bronze_df_2, costkm_bronze_df_3,
    costkm_gold_tech1_df_1, costkm_gold_tech1_df_2, costkm_gold_tech1_df_3,
    costkm_silver_tech1_df_1, costkm_silver_tech1_df_2, costkm_silver_tech1_df_3,
    costkm_bronze_tech1_df_1, costkm_bronze_tech1_df_2, costkm_bronze_tech1_df_3,
    costkm_gold_tech2_df_1, costkm_gold_tech2_df_2, costkm_gold_tech2_df_3,
    costkm_silver_tech2_df_1, costkm_silver_tech2_df_2, costkm_silver_tech2_df_3,
    costkm_bronze_tech2_df_1, costkm_bronze_tech2_df_2, costkm_bronze_tech2_df_3, SCENARIO1, SCENARIO2, SCENARIO3)


#====================================================================================================
#========================= Make plots included in manuscript =====================================

if plots_final == True:

    # Plot Figure 2 (ERF contributions from different aviation species)
    plot_ERF_alltechs_scenarios(ERF_df_1, ERF_df_2, ERF_df_3, ERF_tech1_df_1, ERF_tech1_df_2, ERF_tech1_df_3,
                                ERF_tech2_df_1, ERF_tech2_df_2, ERF_tech2_df_3, SCENARIO1, SCENARIO2, SCENARIO3, REF_SCENARIO, ERF_df_ref,
                                palette='coolwarm')

    # Plot Figure 3 (definitions of climate neutrality)
    plot_climateneutrality_definitions(ERF_df_3, gold_ERF_df_3, ERF_df_ref, bronze_ERF_df_3,
                                       aviation_df_3, gold_baseline_df_3, aviation_df_ref, bronze_baseline_df_3,
                                       'SSP5-8.5', START_DATE, 2100, nonCO2='NOx',
                                       ylabel_nonCO2='NOx emissions (TgN/yr)')

    # Plot Figure 4 (temperature anomaly by 2100 under the different scenarios of future aviation and climate-neutrality targets)
    plot_summary_fair_final(summaryT_alltechs, fair_silver_1, summary_T_CO2only_alltechs_df, summary_T_CO2only_alltechs_err_df, what='new')

    # Plot Figure 5 (temperature change under the different scenarios of future aviation and climate-neutrality targets)
    compare_fair_aviation_alltech(fair_gold_1, fair_gold_3, fair_silver_1, fair_silver_3,
                          fair_bronze_1, fair_bronze_3, fair_gold_tech1_1, fair_gold_tech1_3,
                          fair_silver_tech1_1, fair_silver_tech1_3,
                          fair_bronze_tech1_1, fair_bronze_tech1_3,
                                  fair_gold_tech2_1, fair_gold_tech2_3,
                          fair_silver_tech2_1, fair_silver_tech2_3,
                          fair_bronze_tech2_1, fair_bronze_tech2_3,
                                  scenario1='SSP1-2.6', scenario3='SSP5-8.5',
                                  low_lim1=-0.08, up_lim1=0.52,
                                  low_lim2=-0.05, up_lim2=0.27,
                                  low_lim3= -0.05, up_lim3=0.27,
                                  fair_ref=fair_silver_1,
                                  palette=PALETTE, what='T')

    # Plot Figure 6 (mean CDR rates and cumulative CDR)
    plot_CDR_rates_multi(meanCDR_summary_alltechs, cumulativeCDR_summary_alltechs, positive_cumulativeCDR_summary_alltechs,
                         negative_cumulativeCDR_summary_alltechs,
                         tech1=TECH_1, tech2=TECH_2, scenario = 'A', plots = 'new')

#======================== EXPORT DATA =========================================
if output_xlsx == True:
    # Emissions input data
    with pd.ExcelWriter('Outputs/input_rawdata_emissions.xlsx', datetime_format="YYYY-MM-DD") as writer:
        aviation_df_1.to_excel(writer, index_label='Date', sheet_name='E_SSP1_26')
        aviation_df_2.to_excel(writer, index_label='Date', sheet_name='E_SSP2_45')
        aviation_df_3.to_excel(writer, index_label='Date', sheet_name='E_SSP5_85')
        aviation_df_ref.to_excel(writer, index_label='Date', sheet_name='E_SSP1_19')
        aviation_tech1_df_1.best.to_excel(writer, index_label='Date', sheet_name='E_SSP1_26_SAFs')
        aviation_tech1_df_2.best.to_excel(writer, index_label='Date', sheet_name='E_SSP2_45_SAFs')
        aviation_tech1_df_3.best.to_excel(writer, index_label='Date', sheet_name='E_SSP5_85_SAFs')
        aviation_tech2_df_1.best.to_excel(writer, index_label='Date', sheet_name='E_SSP1_26_Eairplanes')
        aviation_tech2_df_2.best.to_excel(writer, index_label='Date', sheet_name='E_SSP2_45_Eairplanes')
        aviation_tech2_df_3.best.to_excel(writer, index_label='Date', sheet_name='E_SSP5_85_Eairplanes')

    # Demand and energy efficiency input data
    with pd.ExcelWriter('Outputs/output_rawdata_sharmina_demand.xlsx', datetime_format="YYYY-MM-DD") as writer:
        av_SSP2_1.to_excel(writer, index_label='Date', sheet_name='demand_SSP1_26')
        av_SSP2_2.to_excel(writer, index_label='Date', sheet_name='demand_SSP2_45')
        av_SSP2_3.to_excel(writer, index_label='Date', sheet_name='demand_SSP5_85')
        av_SSP2_ref.to_excel(writer, index_label='Date', sheet_name='demand_SSP1_19')

    # Temperature change by 2100
    with pd.ExcelWriter('Outputs/output_rawdata_T.xlsx', datetime_format="YYYY-MM-DD") as writer:
        summaryT_alltechs.to_excel(writer, sheet_name='T_2100_degreeC')
        summary_T_CO2only_alltechs_df.to_excel(writer, sheet_name='T_2100_degreeC_CO2only')
        summary_T_allmetrics.to_excel(writer, sheet_name='T_2100_degreeC_allmetrics_JetA1')
        summary_T_tech1_allmetrics.to_excel(writer, sheet_name='T_2100_degreeC_allmetrics_SAFs')
        summary_T_tech2_allmetrics.to_excel(writer, sheet_name='T_2100_allmetrics_Eairplanes')

    T_SSPs = pd.DataFrame({'SSP1-2.6 Gold': fair_gold_1.T_avCDR,
                      'SSP2-4.5 Gold': fair_gold_2.T_avCDR,
                      'SSP5-8.5 Gold': fair_gold_3.T_avCDR,
                      'SSP1-2.6 Silver': fair_silver_1.T_avCDR,
                      'SSP2-4.5 Silver': fair_silver_2.T_avCDR,
                      'SSP5-8.5 Silver': fair_silver_3.T_avCDR,
                      'SSP1-2.6 Bronze': fair_bronze_1.T_avCDR,
                      'SSP2-4.5 Bronze': fair_bronze_2.T_avCDR,
                      'SSP5-8.5 Bronze': fair_bronze_3.T_avCDR
                      },
                     index = pd.to_datetime(np.arange(1940, 2101), format = '%Y'))
    T_SSPs_tech1 = pd.DataFrame({'SSP1-2.6 Gold': fair_gold_tech1_1.T_avCDR,
                           'SSP2-4.5 Gold': fair_gold_tech1_2.T_avCDR,
                           'SSP5-8.5 Gold': fair_gold_tech1_3.T_avCDR,
                           'SSP1-2.6 Silver': fair_silver_tech1_1.T_avCDR,
                           'SSP2-4.5 Silver': fair_silver_tech1_2.T_avCDR,
                           'SSP5-8.5 Silver': fair_silver_tech1_3.T_avCDR,
                           'SSP1-2.6 Bronze': fair_bronze_tech1_1.T_avCDR,
                           'SSP2-4.5 Bronze': fair_bronze_tech1_2.T_avCDR,
                           'SSP5-8.5 Bronze': fair_bronze_tech1_3.T_avCDR
                           },
                          index=pd.to_datetime(np.arange(1940, 2101), format='%Y'))
    T_SSPs_tech2 = pd.DataFrame({'SSP1-2.6 Gold': fair_gold_tech2_1.T_avCDR,
                           'SSP2-4.5 Gold': fair_gold_tech2_2.T_avCDR,
                           'SSP5-8.5 Gold': fair_gold_tech2_3.T_avCDR,
                           'SSP1-2.6 Silver': fair_silver_tech2_1.T_avCDR,
                           'SSP2-4.5 Silver': fair_silver_tech2_2.T_avCDR,
                           'SSP5-8.5 Silver': fair_silver_tech2_3.T_avCDR,
                           'SSP1-2.6 Bronze': fair_bronze_tech2_1.T_avCDR,
                           'SSP2-4.5 Bronze': fair_bronze_tech2_2.T_avCDR,
                           'SSP5-8.5 Bronze': fair_bronze_tech2_3.T_avCDR
                           },
                          index=pd.to_datetime(np.arange(1940, 2101), format='%Y'))

    # Temperature time series
    with pd.ExcelWriter('Outputs/output_rawdata_T_timeseries.xlsx', datetime_format="YYYY-MM-DD") as writer:
        T_SSPs.to_excel(writer, index_label = 'Date', sheet_name='T_degreeC_SSPs')
        T_SSPs_tech1.to_excel(writer, index_label = 'Date', sheet_name='T_degreeC_SSPs_SAFs')
        T_SSPs_tech2.to_excel(writer, index_label = 'Date', sheet_name='T_degreeC_SSPs_Eairplanes')

    # CDR mean and cumulative rates
    with pd.ExcelWriter('Outputs/output_rawdata_CDR.xlsx', datetime_format="YYYY-MM-DD") as writer:
        meanCDR_summary_alltechs.to_excel(writer, sheet_name='meanCDR_MtCO2_year')
        cumulativeCDR_summary_alltechs.to_excel(writer, sheet_name='cumulativeCDR_2100_MtCO2')
        positive_cumulativeCDR_summary_alltechs.to_excel(writer, sheet_name='positive_cCDR_2100_MtCO2')
        negative_cumulativeCDR_summary_alltechs.to_excel(writer, sheet_name='negative_cCDR_2100_MtCO2')

    # Cost per kilometer flown
    with pd.ExcelWriter('Outputs/output_rawdata_cost_per_km.xlsx', datetime_format="YYYY-MM-DD") as writer:
        mean_costkm_summary_alltechs.to_excel(writer, sheet_name='CostCDR_per_km_USdollar')

    # Temperature outcomes using different conversion metrics
    T_allmetrics = pd.DataFrame({'SSP1-2.6 real': fair_base_1.T_aviation,
                           'SSP2-4.5 real': fair_base_2.T_aviation,
                           'SSP5-8.5 real': fair_base_3.T_aviation,
                           'SSP1-2.6 EWF': fair_EWF_1.T_aviation,
                           'SSP2-4.5 EWF': fair_EWF_2.T_aviation,
                           'SSP5-8.5 EWF': fair_EWF_3.T_aviation,
                           'SSP1-2.6 GWP100': fair_GWP100_1.T_aviation,
                           'SSP2-4.5 GWP100': fair_GWP100_2.T_aviation,
                           'SSP5-8.5 GWP100': fair_GWP100_3.T_aviation,
                           'SSP1-2.6 GWP*': fair_GWPstar_1.T_aviation,
                           'SSP2-4.5 GWP*': fair_GWPstar_2.T_aviation,
                           'SSP5-8.5 GWP*': fair_GWPstar_3.T_aviation,
                           },
                          index=pd.to_datetime(np.arange(1940, 2101), format='%Y'))
    T_allmetrics_tech1 = pd.DataFrame({'SSP1-2.6 real': fair_base_tech1_1.T_aviation,
                           'SSP2-4.5 real': fair_base_tech1_2.T_aviation,
                           'SSP5-8.5 real': fair_base_tech1_3.T_aviation,
                           'SSP1-2.6 EWF': fair_EWF_tech1_1.T_aviation,
                           'SSP2-4.5 EWF': fair_EWF_tech1_2.T_aviation,
                           'SSP5-8.5 EWF': fair_EWF_tech1_3.T_aviation,
                           'SSP1-2.6 GWP100': fair_GWP100_tech1_1.T_aviation,
                           'SSP2-4.5 GWP100': fair_GWP100_tech1_2.T_aviation,
                           'SSP5-8.5 GWP100': fair_GWP100_tech1_3.T_aviation,
                           'SSP1-2.6 GWP*': fair_GWPstar_tech1_1.T_aviation,
                           'SSP2-4.5 GWP*': fair_GWPstar_tech1_2.T_aviation,
                           'SSP5-8.5 GWP*': fair_GWPstar_tech1_3.T_aviation,
                           },
                          index=pd.to_datetime(np.arange(1940, 2101), format='%Y'))
    T_allmetrics_tech2 = pd.DataFrame({'SSP1-2.6 real': fair_base_tech2_1.T_aviation,
                           'SSP2-4.5 real': fair_base_tech2_2.T_aviation,
                           'SSP5-8.5 real': fair_base_tech2_3.T_aviation,
                           'SSP1-2.6 EWF': fair_EWF_tech2_1.T_aviation,
                           'SSP2-4.5 EWF': fair_EWF_tech2_2.T_aviation,
                           'SSP5-8.5 EWF': fair_EWF_tech2_3.T_aviation,
                           'SSP1-2.6 GWP100': fair_GWP100_tech2_1.T_aviation,
                           'SSP2-4.5 GWP100': fair_GWP100_tech2_2.T_aviation,
                           'SSP5-8.5 GWP100': fair_GWP100_tech2_3.T_aviation,
                           'SSP1-2.6 GWP*': fair_GWPstar_tech2_1.T_aviation,
                           'SSP2-4.5 GWP*': fair_GWPstar_tech2_2.T_aviation,
                           'SSP5-8.5 GWP*': fair_GWPstar_tech2_3.T_aviation,
                           },
                          index=pd.to_datetime(np.arange(1940, 2101), format='%Y'))

    # Export temperatures under different conversion metrics
    with pd.ExcelWriter('Outputs/output_rawdata_T_allmetrics.xlsx', datetime_format="YYYY-MM-DD") as writer:
        T_allmetrics.to_excel(writer, index_label = 'Date', sheet_name='T_degreeC_allmetrics')
        T_allmetrics_tech1.to_excel(writer, index_label='Date', sheet_name='T_degreeC_allmetrics_SAFs')
        T_allmetrics_tech2.to_excel(writer, index_label='Date', sheet_name='T_degreeC_allmetrics_Eairplanes')

    # EWF aviation
    cumulativeEWF_summary_alltechs.to_excel('Outputs/EWF_aviation.xlsx')

#======================== PLOTS SUPPLEMENTARY INFORMATION =========================================
if plots_SI == True:

    # plot input emissions from different technology and demand scenarios
    plot_input_scenarios_extended_alltechs(aviation_df_1, aviation_df_2, aviation_df_3, aviation_df_ref,
                                           aviation_tech1_df_1.best, aviation_tech1_df_2.best, aviation_tech1_df_3.best,
                                           aviation_tech2_df_1.best, aviation_tech2_df_2.best, aviation_tech2_df_3.best,
                                           col1='CO2', col2='NOx', col3='Contrail', col4='BC',
                                           y1='TgCO$_2$', y2='TgNOx', y3='km', y4='TgBC',
                                           scenario1='SSP1-2.6', scenario2='SSP2-4.5',
                                           scenario3='SSP5-8.5', ref_scenario='SSP1-1.9')
    plot_input_scenarios_extended_alltechs(aviation_df_1, aviation_df_2, aviation_df_3, aviation_df_ref,
                                           aviation_tech1_df_1.best, aviation_tech1_df_2.best, aviation_tech1_df_3.best,
                                           aviation_tech2_df_1.best, aviation_tech2_df_2.best, aviation_tech2_df_3.best,
                                           col1='SO2', col2='H2O', col3='Fuel', col4='Distance',
                                           y1='TgSO$_2$', y2='TgH$_2$O', y3='Tg', y4='km',
                                           scenario1='SSP1-2.6', scenario2='SSP2-4.5',
                                           scenario3='SSP5-8.5', ref_scenario='SSP1-1.9', what='ptII')

    # plot total CO2eq emissions vs. total CO2 emissions under different scenarios
    plot_summary_EWF(cumulativeEWF_summary_alltechs, size_p=7, add=0.26, palette=PALETTE, colors=COLORS, EWF_low=1.3, EWF_high=2.9)

    # compare temperature outcomes different metrics
    compare_fair_aviation_allmetrics(fair_base_1, fair_base_3,
                                     fair_EWF_1, fair_EWF_3,
                                     fair_GWP100_1, fair_GWP100_3,
                                     fair_GWPstar_1,  fair_GWPstar_3,
                                     fair_base_tech1_1, fair_base_tech1_3,
                                     fair_EWF_tech1_1, fair_EWF_tech1_3,
                                     fair_GWP100_tech1_1,  fair_GWP100_tech1_3,
                                     fair_GWPstar_tech1_1, fair_GWPstar_tech1_3,
                                     fair_base_tech2_1, fair_base_tech2_3,
                                     fair_EWF_tech2_1, fair_EWF_tech2_3,
                                     fair_GWP100_tech2_1, fair_GWP100_tech2_3,
                                     fair_GWPstar_tech2_1, fair_GWPstar_tech2_3,
                                     scenario1='SSP1-2.6', scenario3='SSP5-8.5',
                                     low_lim1=-0.01, up_lim1=0.55,
                                     low_lim2=-0.01, up_lim2=0.25,
                                     low_lim3= -0.01, up_lim3=0.25,
                                     palette=PALETTE, what='T')


    COLORS_LONG = ["#006D77",
                   "#0A6E76",
                   "#136F76",
                   "#1D7175",
                   "#267275",
                   "#307374",
                   "#3A7473",
                   "#437573",
                   "#4D7772",
                   "#567871",
                   "#607971",
                   "#697A70",
                   "#737C70",
                   "#7D7D6F",
                   "#867E6E",
                   "#907F6E",
                   "#99806D",
                   "#A3826C",
                   "#AD836C",
                   "#B6846B",
                   "#C0856B",
                   "#C9866A",
                   "#D38869",
                   "#DC8969",
                   "#E68A68",
                   "#E67E6C"]
    PALETTE_LONG = sns.set_palette(sns.color_palette(COLORS_LONG))

    compare_Toutcomes_deltat(ERF_df_2, gold_ERF_df_2, ERF_df_ref, bronze_ERF_df_2,
                             aviation_df_2, gold_baseline_df_2, aviation_df_ref, bronze_baseline_df_2,
                             SCENARIO2, PALETTE_LONG, START_DATE, 2100)

    compare_fair_aviation_alltech(CDR_gold_df_1, CDR_gold_df_3,
                                  CDR_silver_df_1, CDR_silver_df_3,
                                  CDR_bronze_df_1, CDR_bronze_df_3,
                                  CDR_gold_tech1_df_1, CDR_gold_tech1_df_3,
                                  CDR_silver_tech1_df_1, CDR_silver_tech1_df_3,
                                  CDR_bronze_tech1_df_1, CDR_bronze_tech1_df_3,
                                  CDR_gold_tech2_df_1, CDR_gold_tech2_df_3,
                                  CDR_silver_tech2_df_1, CDR_silver_tech2_df_3,
                                  CDR_bronze_tech2_df_1, CDR_bronze_tech2_df_3,
                                  scenario1='SSP1-2.6', scenario3='SSP5-8.5',
                                  low_lim1= -15, up_lim1=40,
                                  low_lim2= -10, up_lim2=21,
                                  low_lim3= -30, up_lim3= 30,
                                  palette=PALETTE, what='CDR')





if extra_plots == True:
    # Summary plot of CDR
    plot_summaryCDR_alltechs(meanCDR_summary_alltechs,
                             tech1=TECH_1, tech2=TECH_2, what='mean', scenario='A', plot_type='point', size_p=7, plots='new')
    plot_summaryCDR_alltechs(cumulativeCDR_summary_alltechs, positive_cumulativeCDR_summary_alltechs,
                             negative_cumulativeCDR_summary_alltechs,
                             tech1=TECH_1, tech2=TECH_2, what='cumulative', scenario='A', plots='new')

    # CDR cost
    plot_summaryCDR_alltechs(mean_costkm_summary_alltechs,
                             tech1=TECH_1, tech2=TECH_2, what='cost', scenario='A', plot_type='point', size_p=7, plots = 'old')

    T2100_tech2_sensitivity_transitiondate = sensitivity_transition_date(aviation_df_1, aviation_df_2, aviation_df_3,
    aviation_df_ref, ERF_df_ref, 2030,
    UPTAKE_TYPE, TECH_2, ERF_factors,
    trans_date1 = 2090, trans_date2 = 2080, trans_date3 = 2100)

    T2100_tech1_sensitivity_transitiondate = sensitivity_transition_date(aviation_df_1, aviation_df_2, aviation_df_3,
    aviation_df_ref, ERF_df_ref, 2020,
    UPTAKE_TYPE, TECH_1, ERF_factors, trans_date1=2060, trans_date2=2050,
                                                                         trans_date3=2070)



#================================= OUTPUT TEXT ============================================
if output_txt == True:
    output_file = open("Outputs/Outputs.txt", "w")
    output_file.write(
        "Max. non-CO2 ERF contributions:"
        + "\n"
        + "SSP1-2.6 (FF): "
        + str(np.max(ERF_df_1['non-CO2']/ERF_df_1['Tot']))
        + " in year "
        + str(1990 + (np.where(ERF_df_1['non-CO2']/ERF_df_1['Tot'] == (np.max((ERF_df_1['non-CO2']/ERF_df_1['Tot'])))
         ))[0])
        + " interval: "
        + str((unumpy.nominal_values(ERF_df_1['non-CO2']['2056'])-unumpy.std_devs(ERF_df_1['non-CO2']['2056']))/(unumpy.nominal_values(ERF_df_1['Tot']['2056'])-unumpy.std_devs(ERF_df_1['Tot']['2056'])))
        + "-"
        + str((unumpy.nominal_values(ERF_df_1['non-CO2']['2056'])+unumpy.std_devs(ERF_df_1['non-CO2']['2056']))/(unumpy.nominal_values(ERF_df_1['Tot']['2056'])+unumpy.std_devs(ERF_df_1['Tot']['2056'])))
        + "\n"
        + "SSP1-2.6 (FF) in 2100: "
        + str(ERF_df_1['non-CO2']['2100'] / ERF_df_1['Tot']['2100'])
        + " interval: "
        + str((unumpy.nominal_values(ERF_df_1['non-CO2']['2100']) - unumpy.std_devs(ERF_df_1['non-CO2']['2100'])) / (
                    unumpy.nominal_values(ERF_df_1['Tot']['2100']) - unumpy.std_devs(ERF_df_1['Tot']['2100'])))
        + "-"
        + str((unumpy.nominal_values(ERF_df_1['non-CO2']['2100']) + unumpy.std_devs(ERF_df_1['non-CO2']['2100'])) / (
                    unumpy.nominal_values(ERF_df_1['Tot']['2100']) + unumpy.std_devs(ERF_df_1['Tot']['2100'])))
        + "\n"
        + "Max % non-CO2 SSP5-8.5 (FF): "
        + str(np.max(ERF_df_3['non-CO2'] / ERF_df_3['Tot']))
        + " in year "
        + str(
            1990 + (np.where(ERF_df_3['non-CO2'] / ERF_df_3['Tot'] == (np.max((ERF_df_3['non-CO2'] / ERF_df_3['Tot'])))
                             ))[0])
        + " interval: "
        + str((unumpy.nominal_values(ERF_df_3['non-CO2']['2051']) - unumpy.std_devs(ERF_df_3['non-CO2']['2051'])) / (
                    unumpy.nominal_values(ERF_df_3['Tot']['2051']) - unumpy.std_devs(ERF_df_3['Tot']['2051'])))
        + "-"
        + str((unumpy.nominal_values(ERF_df_3['non-CO2']['2051']) + unumpy.std_devs(ERF_df_3['non-CO2']['2051'])) / (
                    unumpy.nominal_values(ERF_df_3['Tot']['2051']) + unumpy.std_devs(ERF_df_3['Tot']['2051'])))
        + "\n"
        + "SSP1-2.6 (SAFs): "
        + str(np.max(ERF_tech1_df_1['non-CO2'] / ERF_tech1_df_1['Tot']))
        + " in year "
        + str(
            1990 + (np.where(ERF_tech1_df_1['non-CO2'] / ERF_tech1_df_1['Tot'] == (np.max((ERF_tech1_df_1['non-CO2'] / ERF_tech1_df_1['Tot'])))
                             ))[0])
        + " interval: "
        + str((unumpy.nominal_values(ERF_tech1_df_1['non-CO2']['2051']) - unumpy.std_devs(ERF_tech1_df_1['non-CO2']['2051'])) / (
                unumpy.nominal_values(ERF_tech1_df_1['Tot']['2051']) - unumpy.std_devs(ERF_tech1_df_1['Tot']['2051'])))
        + "-"
        + str((unumpy.nominal_values(ERF_tech1_df_1['non-CO2']['2051']) + unumpy.std_devs(ERF_tech1_df_1['non-CO2']['2051'])) / (
                unumpy.nominal_values(ERF_tech1_df_1['Tot']['2051']) + unumpy.std_devs(ERF_tech1_df_1['Tot']['2051'])))
        + "\n"
        + "SSP5-8.5 (SAFs): "
        + str(np.max(ERF_tech1_df_3['non-CO2'] / ERF_tech1_df_3['Tot']))
        + " in year "
        + str(
            1990 + (np.where(ERF_tech1_df_3['non-CO2'] / ERF_tech1_df_3['Tot'] == (
                np.max((ERF_tech1_df_3['non-CO2'] / ERF_tech1_df_3['Tot'])))
                             ))[0])
        + " interval: "
        + str((unumpy.nominal_values(ERF_tech1_df_3['non-CO2']['2100']) - unumpy.std_devs(
            ERF_tech1_df_3['non-CO2']['2100'])) / (
                      unumpy.nominal_values(ERF_tech1_df_3['Tot']['2100']) - unumpy.std_devs(
                  ERF_tech1_df_3['Tot']['2100'])))
        + "-"
        + str((unumpy.nominal_values(ERF_tech1_df_3['non-CO2']['2100']) + unumpy.std_devs(
            ERF_tech1_df_3['non-CO2']['2100'])) / (
                      unumpy.nominal_values(ERF_tech1_df_3['Tot']['2100']) + unumpy.std_devs(
                  ERF_tech1_df_3['Tot']['2100'])))
        + "\n"

    )
    output_file.write(
        ""
        "CO2 ERF contributions:"
        + "\n"
        + "2018: "
        + str(ERF_df_2['CO2']['2018'] / ERF_df_2['Tot']['2018'] )
        + " interval: "
        + str((unumpy.nominal_values(ERF_df_2['CO2']['2018']) - unumpy.std_devs(
            ERF_df_2['CO2']['2018'])) / (
                      unumpy.nominal_values(ERF_df_2['Tot']['2018']) - unumpy.std_devs(
                  ERF_df_2['Tot']['2018'])))
        + "-"
        + str((unumpy.nominal_values(ERF_df_2['CO2']['2018']) + unumpy.std_devs(
            ERF_df_2['CO2']['2018'])) / (
                      unumpy.nominal_values(ERF_df_2['Tot']['2018']) + unumpy.std_devs(
                  ERF_df_2['Tot']['2018'])))
        + "\n"
        + "2100 SSP5-85 (FF): "
        + str(ERF_df_3['CO2']['2100'] / ERF_df_3['Tot']['2100'])
        + " interval: "
        + str((unumpy.nominal_values(ERF_df_3['CO2']['2100']) - unumpy.std_devs(
            ERF_df_3['CO2']['2100'])) / (
                      unumpy.nominal_values(ERF_df_3['Tot']['2100']) - unumpy.std_devs(
                  ERF_df_3['Tot']['2100'])))
        + "-"
        + str((unumpy.nominal_values(ERF_df_3['CO2']['2100']) + unumpy.std_devs(
            ERF_df_3['CO2']['2100'])) / (
                      unumpy.nominal_values(ERF_df_3['Tot']['2100']) + unumpy.std_devs(
                  ERF_df_3['Tot']['2100'])))
        + "\n"
        + "2100 SSP5-85 (SAF): "
        + str(ERF_tech1_df_3['CO2']['2100'] / ERF_tech1_df_3['Tot']['2100'])
        + " interval: "
        + str((unumpy.nominal_values(ERF_tech1_df_3['CO2']['2100']) - unumpy.std_devs(
            ERF_tech1_df_3['CO2']['2100'])) / (
                      unumpy.nominal_values(ERF_tech1_df_3['Tot']['2100']) - unumpy.std_devs(
                  ERF_tech1_df_3['Tot']['2100'])))
        + "-"
        + str((unumpy.nominal_values(ERF_tech1_df_3['CO2']['2100']) + unumpy.std_devs(
            ERF_tech1_df_3['CO2']['2100'])) / (
                      unumpy.nominal_values(ERF_tech1_df_3['Tot']['2100']) + unumpy.std_devs(
                  ERF_tech1_df_3['Tot']['2100'])))
        + "\n"
        +"Contrail ERF contributions:"
        + "\n"
        + "2018: "
        + str(ERF_df_2['Contrails and C-C']['2018'] / ERF_df_2['Tot']['2018'] )
        + " interval: "
        + str((unumpy.nominal_values(ERF_df_2['Contrails and C-C']['2018']) - unumpy.std_devs(
            ERF_df_2['Contrails and C-C']['2018'])) / (
                      unumpy.nominal_values(ERF_df_2['Tot']['2018']) - unumpy.std_devs(
                  ERF_df_2['Tot']['2018'])))
        + "-"
        + str((unumpy.nominal_values(ERF_df_2['Contrails and C-C']['2018']) + unumpy.std_devs(
            ERF_df_2['Contrails and C-C']['2018'])) / (
                      unumpy.nominal_values(ERF_df_2['Tot']['2018']) + unumpy.std_devs(
                  ERF_df_2['Tot']['2018'])))
        + "\n"
        + "2100 SSP5-85 (FF): "
        + str(ERF_df_3['Contrails and C-C']['2100'] / ERF_df_3['Tot']['2100'])
        + " interval: "
        + str((unumpy.nominal_values(ERF_df_3['Contrails and C-C']['2100']) - unumpy.std_devs(
            ERF_df_3['Contrails and C-C']['2100'])) / (
                      unumpy.nominal_values(ERF_df_3['Tot']['2100']) - unumpy.std_devs(
                  ERF_df_3['Tot']['2100'])))
        + "-"
        + str((unumpy.nominal_values(ERF_df_3['Contrails and C-C']['2100']) + unumpy.std_devs(
            ERF_df_3['Contrails and C-C']['2100'])) / (
                      unumpy.nominal_values(ERF_df_3['Tot']['2100']) + unumpy.std_devs(
                  ERF_df_3['Tot']['2100'])))
        + "\n"
        + "2100 SSP5-85 (SAF): "
        + str(ERF_tech1_df_3['Contrails and C-C']['2100'] / ERF_tech1_df_3['Tot']['2100'])
        + " interval: "
        + str((unumpy.nominal_values(ERF_tech1_df_3['Contrails and C-C']['2100']) - unumpy.std_devs(
            ERF_tech1_df_3['Contrails and C-C']['2100'])) / (
                      unumpy.nominal_values(ERF_tech1_df_3['Tot']['2100']) - unumpy.std_devs(
                  ERF_tech1_df_3['Tot']['2100'])))
        + "-"
        + str((unumpy.nominal_values(ERF_tech1_df_3['Contrails and C-C']['2100']) + unumpy.std_devs(
            ERF_tech1_df_3['Contrails and C-C']['2100'])) / (
                      unumpy.nominal_values(ERF_tech1_df_3['Tot']['2100']) + unumpy.std_devs(
                  ERF_tech1_df_3['Tot']['2100'])))
        + "\n"
    )

    output_file.write(
         "Paris-compatible aviation warming by 2100: "
         + str(round(fair_silver_1.T_baseline[-1], 3))
         + "\n"
         + "FF by 2100 SSP1-2.6: "
         + str(round(fair_gold_1.T_aviation[-1], 3))
         + "±"
         + str(round((fair_gold_1.T_aviation[-1]-fair_gold_1.T_aviation_lower[-1]), 3))
         + "\n"
         + " & SSP5-8.5: "
         + str(round(fair_gold_3.T_aviation[-1], 3))
         + "±"
         + str(round((fair_gold_3.T_aviation[-1]-fair_gold_3.T_aviation_lower[-1]), 3))
         + "\n"
         + "SAF by 2100 SSP1-2.6: "
         + str(round(fair_gold_tech1_1.T_aviation[-1], 3))
         + "±"
         + str(round((fair_gold_tech1_1.T_aviation[-1] - fair_gold_tech1_1.T_aviation_lower[-1]), 3))
         + "\n"
         + " & SSP5-8.5: "
         + str(round(fair_gold_tech1_3.T_aviation[-1], 3))
         + "±"
         + str(round((fair_gold_tech1_3.T_aviation[-1] - fair_gold_tech1_3.T_aviation_lower[-1]), 3))
         + "\n"
         + "E-airplanes by 2100 SSP1-2.6: "
         + str(round(fair_gold_tech2_1.T_aviation[-1], 3))
         + "±"
         + str(round((fair_gold_tech2_1.T_aviation[-1] - fair_gold_tech2_1.T_aviation_lower[-1]), 3))
         + "\n"
         + " & SSP5-8.5: "
         + str(round(fair_gold_tech2_3.T_aviation[-1], 3))
         + "±"
         + str(round((fair_gold_tech2_3.T_aviation[-1] - fair_gold_tech2_3.T_aviation_lower[-1]), 3))
         + "\n"
         + "CO2-neutrality mitigation:"
         + "\n"
         + " & SSP1-2.6: "
         + str(round((fair_bronze_CO2only_1.T_avCDR[-1] - fair_bronze_CO2only_1.T_aviation[-1]) / fair_bronze_CO2only_1.T_aviation[-1], 2))
         + " ("
         + str(round((fair_bronze_CO2only_1.T_avCDR_lower[-1] - fair_bronze_CO2only_1.T_aviation_lower[-1]) / fair_bronze_CO2only_1.T_aviation_lower[-1], 2))
         + "-"
         + str(round((fair_bronze_CO2only_1.T_avCDR_upper[-1] - fair_bronze_CO2only_1.T_aviation_upper[-1]) / fair_bronze_CO2only_1.T_aviation_upper[-1], 2))
         + ")"
         + "\n"
         + " & SSP5-8.5: "
         + str(round((fair_bronze_CO2only_3.T_avCDR[-1] - fair_bronze_CO2only_3.T_aviation[-1]) / fair_bronze_CO2only_3.T_aviation[-1], 2))
         + " ("
         + str(round((fair_bronze_CO2only_3.T_avCDR_lower[-1] - fair_bronze_CO2only_3.T_aviation_lower[-1]) / fair_bronze_CO2only_3.T_aviation_lower[-1], 2))
         + "-"
         + str(round((fair_bronze_CO2only_3.T_avCDR_upper[-1] - fair_bronze_CO2only_3.T_aviation_upper[-1]) / fair_bronze_CO2only_3.T_aviation_upper[-1], 2))
         + ")"
         + "\n"
    )
    output_file.write(
         "Jet A1 $\Delta$T by 2100:"
         + "\n"
         + str(summaryT_alltechs[summaryT_alltechs['Technology'] == 'Jet A1']
               .drop(['Technology', 'noCDR_fair_std', 'noCDR_fair'], axis=1))
         + "\n"
         + TECH_1 + " $\Delta$T by 2100:"
         + "\n"
         + str(summaryT_alltechs[summaryT_alltechs['Technology'] == TECH_1]
               .drop(['Technology', 'noCDR_fair', 'noCDR_fair_std'], axis=1))
         + "\n"
         + TECH_2 + " $\Delta$T by 2100:"
         + "\n"
         + str(summaryT_alltechs[summaryT_alltechs['Technology'] == TECH_2]
               .drop(['Technology', 'noCDR_fair', 'noCDR_fair_std'], axis=1))
         + "\n"
         + " $\Delta$T by 2100 with Co2 only:"
         + "\n"
         + str(summary_T_CO2only_df)
         + "\n"
         + str(summary_T_CO2only_err_df)
    )

    output_file.write(
        "Maximum temperature increase under different climate neutralities:"
        + "\n"
        + "Gold SSP1-2.6 (Jet A1): " + str(round(np.max(fair_gold_1.T_avCDR), 3))
        + " (SAFs): " + str(round(np.max(fair_gold_tech1_1.T_avCDR), 3))
        + " (E-airplanes): " + str(round(np.max(fair_gold_tech2_1.T_avCDR), 3))
        + "\n"
        + "Silver SSP1-2.6 (Jet A1): " + str(round(np.max(fair_silver_1.T_avCDR), 3))
        + " (SAFs): " + str(round(np.max(fair_silver_tech1_1.T_avCDR), 3))
        + " (E-airplanes): " + str(round(np.max(fair_silver_tech2_1.T_avCDR), 3))
        + "\n"
        + "Bronze SSP1-2.6 (Jet A1): " + str(round(np.max(fair_bronze_1.T_avCDR), 3))
        + " (SAFs): " + str(round(np.max(fair_bronze_tech1_1.T_avCDR), 3))
        + " (E-airplanes): " + str(round(np.max(fair_bronze_tech2_1.T_avCDR), 3))
        + "\n"
        + "Gold SSP2-4.5 (Jet A1): " + str(round(np.max(fair_gold_2.T_avCDR), 3))
        + " (SAFs): " + str(round(np.max(fair_gold_tech1_2.T_avCDR), 3))
        + " (E-airplanes): " + str(round(np.max(fair_gold_tech2_2.T_avCDR), 3))
        + "\n"
        + "Silver SSP2-4.5 (Jet A1): " + str(round(np.max(fair_silver_2.T_avCDR), 3))
        + " (SAFs): " + str(round(np.max(fair_silver_tech1_2.T_avCDR), 3))
        + " (E-airplanes): " + str(round(np.max(fair_silver_tech2_2.T_avCDR), 3))
        + "\n"
        + "Bronze SSP2-4.5 (Jet A1): " + str(round(np.max(fair_bronze_2.T_avCDR), 3))
        + " (SAFs): " + str(round(np.max(fair_bronze_tech1_2.T_avCDR), 3))
        + " (E-airplanes): " + str(round(np.max(fair_bronze_tech2_2.T_avCDR), 3))
        + "\n"
        + "Gold SSP5-8.5 (Jet A1): " + str(round(np.max(fair_gold_3.T_avCDR), 3))
        + " (SAFs): " + str(round(np.max(fair_gold_tech1_3.T_avCDR), 3))
        + " (E-airplanes): " + str(round(np.max(fair_gold_tech2_3.T_avCDR), 3))
        + "\n"
        + "Silver SSP5-8.5 (Jet A1): " + str(round(np.max(fair_silver_3.T_avCDR), 3))
        + " (SAFs): " + str(round(np.max(fair_silver_tech1_3.T_avCDR), 3))
        + " (E-airplanes): " + str(round(np.max(fair_silver_tech2_3.T_avCDR), 3))
        + "\n"
        + "Bronze SSP5-8.5 (Jet A1): " + str(round(np.max(fair_bronze_3.T_avCDR), 3))
        + " (SAFs): " + str(round(np.max(fair_bronze_tech1_3.T_avCDR), 3))
        + " (E-airplanes): " + str(round(np.max(fair_bronze_tech2_3.T_avCDR), 3))
        + "\n"
    )

    output_file.write(
        "Maximum temperature decrease relative to current aviation warming under different climate neutralities:"
        + "\n"
        + "Gold SSP1-2.6 (Jet A1): " + str(
            round(np.min(fair_gold_1.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (SAFs): " + str(
            round(np.min(fair_gold_tech1_1.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (E-airplanes): " + str(
            round(np.min(fair_gold_tech2_1.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + "\n"
        + "Silver SSP1-2.6 (Jet A1): " + str(
            round(np.min(fair_silver_1.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (SAFs): " + str(
            round(np.min(fair_silver_tech1_1.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (E-airplanes): " + str(
            round(np.min(fair_silver_tech2_1.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + "\n"
        + "Bronze SSP1-2.6 (Jet A1): " + str(
            round(np.min(fair_bronze_1.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (SAFs): " + str(
            round(np.min(fair_bronze_tech1_1.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (E-airplanes): " + str(
            round(np.min(fair_bronze_tech2_1.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + "\n"
        + "Gold SSP2-4.5 (Jet A1): " + str(
            round(np.min(fair_gold_2.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (SAFs): " + str(
            round(np.min(fair_gold_tech1_2.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (E-airplanes): " + str(
            round(np.min(fair_gold_tech2_2.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + "\n"
        + "Silver SSP2-4.5 (Jet A1): " + str(
            round(np.min(fair_silver_2.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (SAFs): " + str(
            round(np.min(fair_gold_1.T_avCDR[2019 - 1940] - fair_silver_tech1_2.T_avCDR[2019 - 1940:]), 3))
        + " (E-airplanes): " + str(
            round(np.min(fair_gold_1.T_avCDR[2019 - 1940] - fair_silver_tech2_2.T_avCDR[2019 - 1940:]), 3))
        + "\n"
        + "Bronze SSP2-4.5 (Jet A1): " + str(
            round(np.min(fair_bronze_2.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (SAFs): " + str(
            round(np.min(fair_bronze_tech1_2.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (E-airplanes): " + str(
            round(np.min(fair_bronze_tech2_2.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + "\n"
        + "Gold SSP5-8.5 (Jet A1): " + str(
            round(np.min(fair_gold_3.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (SAFs): " + str(
            round(np.min(fair_gold_tech1_3.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (E-airplanes): " + str(
            round(np.min(fair_gold_tech2_3.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + "\n"
        + "Silver SSP5-8.5 (Jet A1): " + str(
            round(np.min(fair_silver_3.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (SAFs): " + str(
            round(np.min(fair_silver_tech1_3.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (E-airplanes): " + str(
            round(np.min(fair_silver_tech2_3.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + "\n"
        + "Bronze SSP5-8.5 (Jet A1): " + str(
            round(np.min(fair_bronze_3.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (SAFs): " + str(
            round(np.min(fair_bronze_tech1_3.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + " (E-airplanes): " + str(
            round(np.min(fair_bronze_tech2_3.T_avCDR[2019 - 1940:] - fair_gold_1.T_avCDR[2019 - 1940]), 3))
        + "\n"
    )

    output_file.write(
        "Mean CDR rates:"
        + "\n"
        + " CDR rates under Jet A1:"
        + "\n"
        + str(meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1']
              .drop(['Technology', 'summary_CO2_CDR_std', 'summary_CO2_CDR'], axis=1))
        + "\n"
        + "Change in CDR rates under " + TECH_1 + " :"
        + "\n"
        + str(((meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == TECH_1]['summary_Tot_CDR'].values -
                meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1']['summary_Tot_CDR'].values)/
                meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1']['summary_Tot_CDR'].values)[:6])
        + " Interval : "
        + str((((meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == TECH_1]['summary_Tot_CDR'].values -
                 meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == TECH_1]['summary_Tot_CDR_std'].values) -
                 (meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1'][
                    'summary_Tot_CDR'].values - meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1'][
                    'summary_Tot_CDR_std'].values)) /
                 (meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1']['summary_Tot_CDR'].values-
                  meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1'][
                      'summary_Tot_CDR_std'].values))[:6])
        + " - "
        + str((((meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == TECH_1]['summary_Tot_CDR'].values +
                 meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == TECH_1]['summary_Tot_CDR_std'].values) -
                 (meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1'][
                    'summary_Tot_CDR'].values + meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1'][
                    'summary_Tot_CDR_std'].values)) /
                 (meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1']['summary_Tot_CDR'].values +
                  meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1'][
                      'summary_Tot_CDR_std'].values))[:6])
        + "\n"
        + "Change in CDR rates under " + TECH_2 + " :"
        + "\n"
        + str((((meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == TECH_2]['summary_Tot_CDR'].values -
                meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1']['summary_Tot_CDR'].values)/meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1']['summary_Tot_CDR'].values))[:6])
        + " Interval : "
        + str((((meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == TECH_2]['summary_Tot_CDR'].values -
                 meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == TECH_2]['summary_Tot_CDR_std'].values) -
                 (meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1'][
                    'summary_Tot_CDR'].values - meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1'][
                    'summary_Tot_CDR_std'].values)) /
                 (meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1']['summary_Tot_CDR'].values-
                  meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1'][
                      'summary_Tot_CDR_std'].values))[:6])
        + " - "
        + str((((meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == TECH_2]['summary_Tot_CDR'].values +
                 meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == TECH_2]['summary_Tot_CDR_std'].values) -
                 (meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1'][
                    'summary_Tot_CDR'].values + meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1'][
                    'summary_Tot_CDR_std'].values)) /
                 (meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1']['summary_Tot_CDR'].values +
                  meanCDR_summary_alltechs[meanCDR_summary_alltechs['Technology'] == 'Jet A1'][
                      'summary_Tot_CDR_std'].values))[:6])
        + "\n"
    )

    output_file.write(
        "Cumulative CDR rates:"
        + "\n"
        + " CDR rates under Jet A1:"
        + "\n"
        + str(cumulativeCDR_summary_alltechs[cumulativeCDR_summary_alltechs['Technology'] == 'Jet A1']
              .drop(['Technology', 'summary_CO2_CDR_std', 'summary_CO2_CDR'], axis=1))
        + "\n"
        + " Tot/CO2 CDR rates under Jet A1:"
        + "\n"
        + str(cumulativeCDR_summary_alltechs.loc[(cumulativeCDR_summary_alltechs['Technology'] == 'Jet A1')]['summary_Tot_CDR'] \
              / cumulativeCDR_summary_alltechs.loc[(cumulativeCDR_summary_alltechs['Technology'] == 'Jet A1')]['summary_CO2_CDR'])
        + "\n"
        + "Cumulative CDR rates under " + TECH_1 + " :"
        + "\n"
        + str(cumulativeCDR_summary_alltechs[cumulativeCDR_summary_alltechs['Technology'] == TECH_1]
              .drop(['Technology', 'summary_CO2_CDR_std', 'summary_CO2_CDR'], axis=1))
        + "\n"
        + "Cumulative CDR rates under " + TECH_2 + " :"
        + "\n"
        + str(cumulativeCDR_summary_alltechs[cumulativeCDR_summary_alltechs['Technology'] == TECH_2]
              .drop(['Technology', 'summary_CO2_CDR_std', 'summary_CO2_CDR'], axis=1))
        + "\n"
    )

    output_file.write(
        "Mean cost per km of CDR:"
        + "\n"
        + " Cost under Jet A1:"
        + "\n"
        + str(mean_costkm_summary_alltechs[mean_costkm_summary_alltechs['Technology'] == 'Jet A1']
              .drop(['Technology'], axis=1))
        + "\n"
        + " Cost under " + TECH_1 + " :"
        + "\n"
        + str(mean_costkm_summary_alltechs[mean_costkm_summary_alltechs['Technology'] == TECH_1]
              .drop(['Technology'], axis=1))
        + "\n"
        + " Cost under " + TECH_2 + " :"
        + "\n"
        + str(mean_costkm_summary_alltechs[mean_costkm_summary_alltechs['Technology'] == TECH_2]
              .drop(['Technology'], axis=1))
        + "\n"
    )

    mean_costZHNY_summary_alltechs_A = calculate_cost_passenger(mean_costkm_summary_alltechs,
                                                                km=6326, passengers=350)

    output_file.write(
        "Mean cost ZH-NY of CDR:"
        + "\n"
        + " Cost under Jet A1:"
        + "\n"
        + str(mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1']
              .drop(['Technology'], axis=1))
        + "\n"
        + " Cost under " + TECH_1 + " :"
        + "\n"
        + str(mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == TECH_1]
              .drop(['Technology'], axis=1))
        + "\n"
        + " Cost under " + TECH_2 + " :"
        + "\n"
        + str(mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == TECH_2]
              .drop(['Technology'], axis=1))
        + "\n"
        + "Change in cost: "
        + "\n"
        + " SAFs - "
        + str(
            (mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == TECH_1]['summary_Tot_costkm'].values
             - mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                 'summary_Tot_costkm'].values) / mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                 'summary_Tot_costkm'].values)
        + "("
        + str(
            ((mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == TECH_1]['summary_Tot_costkm'].values
            + mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == TECH_1]['summary_Tot_costkm_std'].values)
             - (mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                 'summary_Tot_costkm'].values + mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                 'summary_Tot_costkm_std'].values)) / (mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                 'summary_Tot_costkm'].values+mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                 'summary_Tot_costkm_std'].values))
        + "-"
        + str(
            ((mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == TECH_1][
                  'summary_Tot_costkm'].values
              - mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == TECH_1][
                  'summary_Tot_costkm_std'].values)
             - (mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                    'summary_Tot_costkm'].values -
                mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                    'summary_Tot_costkm_std'].values)) / (
                        mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                            'summary_Tot_costkm'].values -
                        mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                            'summary_Tot_costkm_std'].values))
        + "\n"
        + " E-airplanes - "
        + str(
            (mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == TECH_2][
                 'summary_Tot_costkm'].values
             - mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                 'summary_Tot_costkm'].values) /
            mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                'summary_Tot_costkm'].values)
        + "("
        + str(
            ((mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == TECH_2][
                  'summary_Tot_costkm'].values
              + mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == TECH_2][
                  'summary_Tot_costkm_std'].values)
             - (mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                    'summary_Tot_costkm'].values +
                mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                    'summary_Tot_costkm_std'].values)) / (
                        mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                            'summary_Tot_costkm'].values +
                        mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                            'summary_Tot_costkm_std'].values))
        + "-"
        + str(
            ((mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == TECH_2][
                  'summary_Tot_costkm'].values
              - mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == TECH_2][
                  'summary_Tot_costkm_std'].values)
             - (mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                    'summary_Tot_costkm'].values -
                mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                    'summary_Tot_costkm_std'].values)) / (
                    mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                        'summary_Tot_costkm'].values -
                    mean_costZHNY_summary_alltechs_A[mean_costZHNY_summary_alltechs_A['Technology'] == 'Jet A1'][
                        'summary_Tot_costkm_std'].values))

    )

    output_file.close()










