import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import seaborn as sns
import pyam
colors = pyam.plotting.PYAM_COLORS
pd.DataFrame({'name': list(colors.keys()), 'color': list(colors.values())})

sns.set_style(style='white')
from functions import *

#plt.style.use('plot_style.txt')

# Define colour palette
bau = "#4393c3"
bau_shade = "#77b1d4"
decrease = "#f46d43"
decrease_shade = "#fd9a7c"
const = "#737373"
const_shade = "#d9d9d9"
const_shadier = "#EDEDED"

bronze = "#824A02"
bronze_shade = "#f5a564"
silver = "#A7A7AD"
silver_shade = "#b8b8b8"
gold = "#daa520"
gold_shade = "#fff266"

scenario1 = 'SSP1-2.6'
scenario2 = 'SSP2-4.5'
scenario3 = 'SSP5-8.5'
ref_scenario = 'SSP1-1.9'

baseline_1 = 'Gold'
baseline_2 = 'Silver'
baseline_3 = 'Bronze'
baseline_4 = 'EWF'

col_scenario1 = colors['AR6-'+scenario1]
col_scenario1_shade = '#005bb3'
col_scenario2 = colors['AR6-'+scenario2]
col_scenario2_shade = '#f8ab51'
col_scenario3 = colors['AR6-'+scenario3]
col_scenario3_shade = '#fe0004'
col_scenarioref = colors['AR6-'+ref_scenario]
col_scenarioref_shade = '#00a9cf'

TECH_1 = 'Zero-CO$_2$ fuels'
TECH_2 = 'No-emissions aircraft'

#============================= FINAL PLOTS =============================================

def plot_summary_EWF(summary_df, size_p = 7, add = 0.17, palette = 'BrBG_r', EWF_high = 2, EWF_low = 1.7, colors = None):
    """
    plot the relationship between total CO2-equivalent emissions and CO2 emissions only (which is equal to
    the Emissions Weighting Factor) for different scenarios of future aviation's demand and technology.
    :param summary_df:
    :param size_p:
    :param add:
    :param palette:
    :param EWF_high:
    :param EWF_low:
    :param colors:
    :return:
    """

    fig, ax = plt.subplots(figsize=(3.5, 3))

    y = 'EWF'
    y_std = 'EWF_std'
    x_tech1 = np.array([0, 1, 2])
    x_tech0 = x_tech1 - np.array([add, add, add])
    x_tech2 = x_tech1 + np.array([add, add, add])

    y_ff = summary_df.loc[(summary_df['Technology'] == 'Jet A1')]
    y_saf = summary_df.loc[(summary_df['Technology'] == TECH_1)]
    y_e = summary_df.loc[(summary_df['Technology'] == TECH_2)]

    x = np.array([-1.5, 3])
    y_low = np.array([EWF_low, EWF_low])
    y_high = np.array([EWF_high, EWF_high])
    y_best1 = np.array([1.7, 1.7])
    y_best2 = np.array([2, 2])

    ax.fill_between(x, y_low, y_high, color = const_shadier, label='EWF='+str(EWF_low)+'-'+str(EWF_high))
    ax.fill_between(x, y_best1, y_best2, color=const_shade, label='EWF=' + str(EWF_low) + '-' + str(EWF_high))
    ax.hlines(1, -1.5, 3, linestyles='dashed', color = const)


    ax.errorbar(x=x_tech0, y=y_ff[y],
                 yerr=y_ff[y_std],
                 fmt='none', c='black',
                 capsize=2)
    ax.errorbar(x=x_tech1, y=y_saf[y],
                 yerr=y_saf[y_std],
                 fmt='none', c='black',
                 capsize=2)
    ax.errorbar(x=x_tech2, y=y_e[y],
                 yerr=y_e[y_std],
                 fmt='none', c='black',
                 capsize=2)


    sns.swarmplot(ax=ax, x='Scenario', y=y, hue='Technology', data=summary_df,
                  palette=palette,
                  dodge=True, size=size_p)



    legend_elements = [
        #Patch(facecolor=const_shade, label='EWF range'),
        #Patch(facecolor=const, label=' EWF best '),
          #Line2D([0], [0], linestyle='--', color=const),
        Line2D([0], [0], marker='o', ls='none', markeredgecolor = 'none', label='Fossil jet fuels',
               markerfacecolor=colors[0], markersize=size_p),
        Line2D([0], [0], marker='o', ls='none', markeredgecolor='none', label=TECH_1,
               markerfacecolor=colors[1], markersize=size_p),
        Line2D([0], [0], marker='o', ls='none', markeredgecolor='none', label=TECH_2,
               markerfacecolor=colors[2], markersize=size_p),
    ]

    ax.legend(handles = legend_elements, ncol=1, frameon = False)



    ax.set_xticks(x_tech1)
    ax.set_xticklabels(('SSP1-2.6', 'SSP2-4.5', 'SSP5-8.5'))
    ax.set_ylabel('Total CO$_{2eq}$/CO$_2$ emissions')

    plt.yticks(np.arange(0, 16, step=2))


    fig.tight_layout()
    fig.savefig("Figures/comparison_EWF_GWPstar.png", dpi=850, bbox_inches="tight")

def make_single_summary_fair(summary_df, fair_ref, fair_CO2only, fair_CO2only_err, col, col_shade, ax=None,
                             baseline1=baseline_1, baseline2=baseline_2, baseline3=baseline_3,
                             tech1=TECH_1, tech2=TECH_2):
    colors = sns.dark_palette(col_shade, reverse=True)
    # sns.dark_palette(col, 5, reverse=True)
    ax = ax or plt.gca()
    add = 0.2
    x = np.array([0, 1, 2, 3])
    x_ff = x + np.array([-add, -add, -add, -add])
    x_tech1 = x
    x_tech2 = x + np.array([add, add, add, add])
    x_CO2only = np.array([1 - 2 * add, 2 - 2 * add, 3 - 2 * add])
    target = fair_ref.T_baseline[-1]
    x_ticks = np.array(['no climate' + '\n' + ' neutrality', baseline1, baseline2, baseline3])
    df_ff = summary_df.loc[summary_df['Technology'] == 'Jet A1']
    df_tech1 = summary_df.loc[summary_df['Technology'] == tech1]
    df_tech2 = summary_df.loc[summary_df['Technology'] == tech2]
    y_ff = np.array((float(df_ff['noCDR_fair'].loc[(df_ff['Climate neutrality'] == baseline1)]),
                     float(df_ff['summary_fair'].loc[(df_ff['Climate neutrality'] == baseline1)]),
                     float(df_ff['summary_fair'].loc[(df_ff['Climate neutrality'] == baseline2)]),
                     float(df_ff['summary_fair'].loc[(df_ff['Climate neutrality'] == baseline3)])))
    y_err_ff = np.array((float(df_ff['noCDR_fair_std'].loc[(df_ff['Climate neutrality'] == baseline1)]),
                         float(df_ff['summary_fair_std'].loc[(df_ff['Climate neutrality'] == baseline1)]),
                         float(df_ff['summary_fair_std'].loc[(df_ff['Climate neutrality'] == baseline2)]),
                         float(df_ff['summary_fair_std'].loc[(df_ff['Climate neutrality'] == baseline3)])))
    y_tech1 = np.array((float(df_tech1['noCDR_fair'].loc[(df_tech1['Climate neutrality'] == baseline1)]),
                        float(df_tech1['summary_fair'].loc[(df_tech1['Climate neutrality'] == baseline1)]),
                        float(df_tech1['summary_fair'].loc[(df_tech1['Climate neutrality'] == baseline2)]),
                        float(df_tech1['summary_fair'].loc[(df_tech1['Climate neutrality'] == baseline3)])))
    y_err_tech1 = np.array((float(df_tech1['noCDR_fair_std'].loc[(df_tech1['Climate neutrality'] == baseline1)]),
                            float(df_tech1['summary_fair_std'].loc[(df_tech1['Climate neutrality'] == baseline1)]),
                            float(df_tech1['summary_fair_std'].loc[(df_tech1['Climate neutrality'] == baseline2)]),
                            float(df_tech1['summary_fair_std'].loc[(df_tech1['Climate neutrality'] == baseline3)])))
    y_tech2 = np.array((float(df_tech2['noCDR_fair'].loc[(df_tech2['Climate neutrality'] == baseline1)]),
                        float(df_tech2['summary_fair'].loc[(df_tech2['Climate neutrality'] == baseline1)]),
                        float(df_tech2['summary_fair'].loc[(df_tech2['Climate neutrality'] == baseline2)]),
                        float(df_tech2['summary_fair'].loc[(df_tech2['Climate neutrality'] == baseline3)])))
    y_err_tech2 = np.array((float(df_tech2['noCDR_fair_std'].loc[(df_tech2['Climate neutrality'] == baseline1)]),
                            float(df_tech2['summary_fair_std'].loc[(df_tech2['Climate neutrality'] == baseline1)]),
                            float(df_tech2['summary_fair_std'].loc[(df_tech2['Climate neutrality'] == baseline2)]),
                            float(df_tech2['summary_fair_std'].loc[(df_tech2['Climate neutrality'] == baseline3)])))

    ax.hlines(target, -0.5, 3.3, linestyles="dashed", color=const, alpha=0.7, label='1.5°C-compatible')

    ax.errorbar(x=x_ff, y=y_ff, yerr=y_err_ff, fmt='none', c='black',
                capsize=2)
    ax.errorbar(x=x_tech1, y=y_tech1, yerr=y_err_tech1, fmt='none', c='black',
                capsize=2)
    ax.errorbar(x=x_tech2, y=y_tech2, yerr=y_err_tech2, fmt='none', c='black',
                capsize=2)
    ax.errorbar(x=x_tech2, y=y_tech2, yerr=y_err_tech2, fmt='none', c='black',
                capsize=2)
    ax.errorbar(x=x_CO2only, y=fair_CO2only, yerr=fair_CO2only_err, fmt='none', c='black',
                capsize=2)
    ax.plot(x_ff, y_ff, 'o', color=colors[0], label='Fossil jet fuels')
    ax.plot(x_tech1, y_tech1, 'o', color=colors[1], label=tech1)
    ax.plot(x_tech2, y_tech2, 'o', color=colors[2], label=tech2)
    ax.plot(x_CO2only, fair_CO2only, 'o', color=const, label='only CO$_2$')
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks, rotation=20)
    ax.set_ylabel('Aviation $\Delta$T in 2100 (°C)')
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, ncol=1, frameon=False)


def make_ssp_summary_fair_new(summary_df, fair_ref, fair_CO2only, fair_CO2only_err, col, col_shade, ylbl, ax=None,
                             baseline1=baseline_1, baseline2=baseline_2, baseline3=baseline_3,
                             tech1=TECH_1, tech2=TECH_2, labels = True, add = 0.2, ):
    colors = sns.dark_palette(col_shade, reverse=True)
    # sns.dark_palette(col, 5, reverse=True)
    ax = ax or plt.gca()
    x = np.array([-0.4, 0.8, 2, 3.2, 4.4])
    x_ff = x + np.array([-add, -add, -add, -add, -add])
    x_tech1 = x
    x_tech2 = x + np.array([add, add, add, add, add])
    target = fair_ref.T_baseline[-1]
    x_ticks = np.array(['No' + '\n' + 'offset', 'CO$_2$ only'+ '\n' + 'offset', baseline1+ '\n' + 'offset',
                        baseline2+ '\n' + 'offset', baseline3+ '\n' + 'offset'])
    df_ff = summary_df.loc[summary_df['Technology'] == 'Jet A1']
    df_tech1 = summary_df.loc[summary_df['Technology'] == tech1]
    df_tech2 = summary_df.loc[summary_df['Technology'] == tech2]
    y_ff = np.array((float(df_ff['noCDR_fair'].loc[(df_ff['Climate neutrality'] == baseline1)]),
                     float(fair_CO2only[0]),
                     float(df_ff['summary_fair'].loc[(df_ff['Climate neutrality'] == baseline1)]),
                     float(df_ff['summary_fair'].loc[(df_ff['Climate neutrality'] == baseline2)]),
                     float(df_ff['summary_fair'].loc[(df_ff['Climate neutrality'] == baseline3)])))
    y_err_ff = np.array((float(df_ff['noCDR_fair_std'].loc[(df_ff['Climate neutrality'] == baseline1)]),
                         float(fair_CO2only_err[0]),
                         float(df_ff['summary_fair_std'].loc[(df_ff['Climate neutrality'] == baseline1)]),
                         float(df_ff['summary_fair_std'].loc[(df_ff['Climate neutrality'] == baseline2)]),
                         float(df_ff['summary_fair_std'].loc[(df_ff['Climate neutrality'] == baseline3)])))
    y_tech1 = np.array((float(df_tech1['noCDR_fair'].loc[(df_tech1['Climate neutrality'] == baseline1)]),
                        float(fair_CO2only[1]),
                        float(df_tech1['summary_fair'].loc[(df_tech1['Climate neutrality'] == baseline1)]),
                        float(df_tech1['summary_fair'].loc[(df_tech1['Climate neutrality'] == baseline2)]),
                        float(df_tech1['summary_fair'].loc[(df_tech1['Climate neutrality'] == baseline3)])))
    y_err_tech1 = np.array((float(df_tech1['noCDR_fair_std'].loc[(df_tech1['Climate neutrality'] == baseline1)]),
                            float(fair_CO2only_err[1]),
                            float(df_tech1['summary_fair_std'].loc[(df_tech1['Climate neutrality'] == baseline1)]),
                            float(df_tech1['summary_fair_std'].loc[(df_tech1['Climate neutrality'] == baseline2)]),
                            float(df_tech1['summary_fair_std'].loc[(df_tech1['Climate neutrality'] == baseline3)])))
    y_tech2 = np.array((float(df_tech2['noCDR_fair'].loc[(df_tech2['Climate neutrality'] == baseline1)]),
                        float(fair_CO2only[2]),
                        float(df_tech2['summary_fair'].loc[(df_tech2['Climate neutrality'] == baseline1)]),
                        float(df_tech2['summary_fair'].loc[(df_tech2['Climate neutrality'] == baseline2)]),
                        float(df_tech2['summary_fair'].loc[(df_tech2['Climate neutrality'] == baseline3)])))
    y_err_tech2 = np.array((float(df_tech2['noCDR_fair_std'].loc[(df_tech2['Climate neutrality'] == baseline1)]),
                            float(fair_CO2only_err[2]),
                            float(df_tech2['summary_fair_std'].loc[(df_tech2['Climate neutrality'] == baseline1)]),
                            float(df_tech2['summary_fair_std'].loc[(df_tech2['Climate neutrality'] == baseline2)]),
                            float(df_tech2['summary_fair_std'].loc[(df_tech2['Climate neutrality'] == baseline3)])))

    ax.hlines(target, -0.7, 4.7, linestyles="dashed", color=const, alpha=0.7, label='1.5°C-compatible')

    ax.errorbar(x=x_ff, y=y_ff, yerr=y_err_ff, fmt='none', c='black',
                capsize=2)
    ax.errorbar(x=x_tech1, y=y_tech1, yerr=y_err_tech1, fmt='none', c='black',
                capsize=2)
    ax.errorbar(x=x_tech2, y=y_tech2, yerr=y_err_tech2, fmt='none', c='black',
                capsize=2)
    ax.errorbar(x=x_tech2, y=y_tech2, yerr=y_err_tech2, fmt='none', c='black',
                capsize=2)
    ax.plot(x_ff, y_ff, 'o', markersize=7,  color=colors[0], label='Fossil jet fuels')
    ax.plot(x_tech1, y_tech1, 'o', markersize=7, color=colors[1], label=tech1)
    ax.plot(x_tech2, y_tech2, 'o', markersize=7,  color=colors[2], label=tech2)
    if labels == True:
        ax.set_xticks(x)
        ax.set_xticklabels(x_ticks, rotation=20)
        #ax.set_xlabel('Offsetting scheme')
    else:
        ax.set_xticks([])
    ax.set_ylabel(ylabel=ylbl)
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, ncol=1, frameon=False)





def plot_summary_fair_final(summary_df, fair_ref, fair_CO2only, fair_CO2only_err,
                            scenario1='SSP1_26', scenario2='SSP2_45', scenario3='SSP5_85', what='new'):
    summary_1 = summary_df.loc[(summary_df['Scenario'] == scenario1)]
    summary_2 = summary_df.loc[(summary_df['Scenario'] == scenario2)]
    summary_3 = summary_df.loc[(summary_df['Scenario'] == scenario3)]
    CO2only_1 = fair_CO2only.loc[scenario1].values
    CO2only_2 = fair_CO2only.loc[scenario2].values
    CO2only_3 = fair_CO2only.loc[scenario3].values
    CO2only_err_1 = fair_CO2only_err.loc[scenario1].values
    CO2only_err_2 = fair_CO2only_err.loc[scenario2].values
    CO2only_err_3 = fair_CO2only_err.loc[scenario3].values


    if what == 'new':
        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(7.2, 6.8), tight_layout = True)
        plt.tight_layout(w_pad=0.5)
        ax1, ax3 = (
            ax.flatten()[0],
            ax.flatten()[1],
        )
        make_ssp_summary_fair_new(summary_1, fair_ref, CO2only_1, CO2only_err_1, col_scenario1, col_scenario1_shade,
                                 ax=ax1,ylbl=None,
                                 baseline1=baseline_1, baseline2=baseline_2, baseline3=baseline_3,
                                 tech1=TECH_1, tech2=TECH_2, labels= False, add = 0.25)
        make_ssp_summary_fair_new(summary_3, fair_ref, CO2only_3, CO2only_err_3, col_scenario3, col_scenario3_shade,
                                 ax=ax3, ylbl = None,
                                 baseline1=baseline_1, baseline2=baseline_2, baseline3=baseline_3,
                                 tech1=TECH_1, tech2=TECH_2, add = 0.25)
        ax1.text(0.05, 0.05, 'SSP1-2.6', transform=ax1.transAxes, color = col_scenario1, fontweight='bold', va="bottom", ha="left")
        ax3.text(0.05, 0.05, 'SSP5-8.5', transform=ax3.transAxes, color = col_scenario3, fontweight='bold', va="bottom", ha="left")

        fig.supylabel('Aviation temperature change in 2100 (°C)')
        #box = ax1.get_position()
        #ax1.set_position([box.x0, box.y0, box.width, box.height])
        #box = ax3.get_position()
        #ax3.set_position([box.x0, box.y0, box.width, box.height])

    else:
        fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(4, 10))
        ax1, ax2, ax3 = (
            ax.flatten()[0],
            ax.flatten()[1],
            ax.flatten()[2],
        )
        make_single_summary_fair(summary_1, fair_ref, CO2only_1, CO2only_err_1, col_scenario1, col_scenario1_shade,
                                 ax=ax1,ylbl = 'Aviation $\Delta$T in 2100 (°C)',
                                 baseline1=baseline_1, baseline2=baseline_2, baseline3=baseline_3,
                                 tech1=TECH_1, tech2=TECH_2)
        make_single_summary_fair(summary_2, fair_ref, CO2only_2, CO2only_err_2, col_scenario2, col_scenario2_shade,
                                 ax=ax2, ylbl = 'Aviation $\Delta$T in 2100 (°C)',
                                 baseline1=baseline_1, baseline2=baseline_2, baseline3=baseline_3,
                                 tech1=TECH_1, tech2=TECH_2)
        make_single_summary_fair(summary_3, fair_ref, CO2only_3, CO2only_err_3, col_scenario3, col_scenario3_shade,
                                 ax=ax3, ylbl = 'Aviation $\Delta$T in 2100 (°C)',
                                 baseline1=baseline_1, baseline2=baseline_2, baseline3=baseline_3,
                                 tech1=TECH_1, tech2=TECH_2)

        ax1.text(0.9, 0.05, 'SSP1-2.6', transform=ax1.transAxes, va="bottom", ha="right")
        ax2.text(0.9, 0.05, 'SSP2-4.5', transform=ax2.transAxes, va="bottom", ha="right")
        ax3.text(0.9, 0.05, 'SSP5-8.5', transform=ax3.transAxes, va="bottom", ha="right")

        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width, box.height * 0.95])
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width, box.height * 0.95])
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width, box.height * 0.95])


    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['font.size'] = 13
    fig.savefig("Figures/fair_outcomes_comparison_final.pdf", dpi=600, bbox_inches="tight")

def compare_fair_aviation_alltech(fair1_gold, fair3_gold, fair1_silver, fair3_silver,
                          fair1_bronze, fair3_bronze, fair1_tech1_gold, fair3_tech1_gold,
                                  fair1_tech1_silver, fair3_tech1_silver,
                          fair1_tech1_bronze, fair3_tech1_bronze,
                                  fair1_tech2_gold, fair3_tech2_gold,
                                  fair1_tech2_silver, fair3_tech2_silver,
                          fair1_tech2_bronze, fair3_tech2_bronze,
                          scenario1, scenario3, what = 'T', end_year = 2100, start_year = 1990,
                          CDR_type = 'A', low_lim1 = None, up_lim1 = None, low_lim2 = None, up_lim2 = None,
                                  low_lim3 = None, up_lim3 = None, fair_ref = None,
                          fair1_EWF = None, fair3_EWF= None,
                              palette = 'BrBG_r', l_width = 2):
    """
    Plot concentrations, forcing, and temperatures under CDR policy and references
    :param fair1_gold:
    :param fair3_gold:
    :param fair1_silver:
    :param fair3_silver:
    :param fair1_bronze:
    :param fair3_bronze:
    :param fair1_tech1_gold:
    :param fair3_tech1_gold:
    :param fair1_tech1_silver:
    :param fair3_tech1_silver:
    :param fair1_tech1_bronze:
    :param fair3_tech1_bronze:
    :param fair1_tech2_gold:
    :param fair3_tech2_gold:
    :param fair1_tech2_silver:
    :param fair3_tech2_silver:
    :param fair1_tech2_bronze:
    :param fair3_tech2_bronze:
    :param scenario1:
    :param scenario3:
    :param what:
    :param end_year:
    :param start_year:
    :param CDR_type:
    :param low_lim1:
    :param up_lim1:
    :param low_lim2:
    :param up_lim2:
    :param low_lim3:
    :param up_lim3:
    :param fair_ref:
    :param fair1_EWF:
    :param fair2_EWF:
    :param fair3_EWF:
    :param palette:
    :param l_width:
    :return:
    """
    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(7, 7.3))
    ax1, ax4, ax2, ax5, ax3, ax6= (
        ax.flatten()[0],
        ax.flatten()[1],
        ax.flatten()[2],
        ax.flatten()[3],
        ax.flatten()[4],
        ax.flatten()[5]
    )

    colors = sns.color_palette([gold, silver, bronze, decrease, 'black' ])
    colors_shade = sns.color_palette([gold_shade, silver_shade, bronze_shade, decrease_shade, 'grey'])

    if what == 'T':
        dates = np.arange(1940, end_year + 1)
        ax1.fill_between(dates, fair1_gold.T_aviation_upper, fair1_gold.T_aviation_lower, color =colors_shade[4], alpha = 0.5)
        ax1.fill_between(dates, fair1_gold.T_avCDR_upper, fair1_gold.T_avCDR_lower, color=colors_shade[0], alpha=0.5)
        ax1.fill_between(dates, fair1_silver.T_avCDR_upper, fair1_silver.T_avCDR_lower, color=colors_shade[1], alpha=0.5)
        ax1.fill_between(dates, fair1_bronze.T_avCDR_upper, fair1_bronze.T_avCDR_lower, color=colors_shade[2], alpha=0.5)
        if fair1_EWF is not None:
            ax1.fill_between(dates, fair1_EWF.T_avCDR_upper, fair1_EWF.T_avCDR_lower, color=colors[3], alpha=0.3)
        if fair_ref is not None:
            ax1.plot(dates, fair_ref.T_baseline, '--k', label = '1.5°C-compatible')
        ax1.plot(dates, fair1_gold.T_aviation, color=colors[4], label='No neutrality', linewidth=l_width)
        ax1.plot(dates, fair1_gold.T_avCDR, color=colors[0], label=baseline_1, linewidth = l_width)
        ax1.plot(dates, fair1_silver.T_avCDR, color=colors[1], label=baseline_2, linewidth = l_width)
        ax1.plot(dates, fair1_bronze.T_avCDR, color=colors[2], label=baseline_3, linewidth = l_width)
        if fair1_EWF is not None:
            ax1.plot(dates, fair1_EWF.T_avCDR, color=colors[3], label=baseline_4, linewidth = l_width)

        ax2.fill_between(dates, fair1_tech1_gold.T_aviation_upper, fair1_tech1_gold.T_aviation_lower, color=colors_shade[4], alpha=0.5)
        ax2.fill_between(dates, fair1_tech1_gold.T_avCDR_upper, fair1_tech1_gold.T_avCDR_lower, color=colors_shade[0], alpha=0.5)
        ax2.fill_between(dates, fair1_tech1_silver.T_avCDR_upper, fair1_tech1_silver.T_avCDR_lower, color=colors_shade[1], alpha=0.5)
        ax2.fill_between(dates, fair1_tech1_bronze.T_avCDR_upper, fair1_tech1_bronze.T_avCDR_lower, color=colors_shade[2], alpha=0.5)
        if fair_ref is not None:
            ax2.plot(dates, fair_ref.T_baseline, '--k', label='1.5°C-compatible')
        ax2.plot(dates, fair1_tech1_gold.T_aviation, color=colors[4], label='No neutrality', linewidth=l_width)
        ax2.plot(dates, fair1_tech1_gold.T_avCDR, color=colors[0], label=baseline_1, linewidth=l_width)
        ax2.plot(dates, fair1_tech1_silver.T_avCDR, color=colors[1], label=baseline_2, linewidth=l_width)
        ax2.plot(dates, fair1_tech1_bronze.T_avCDR, color=colors[2], label=baseline_3, linewidth=l_width)

        ax3.fill_between(dates, fair1_tech2_gold.T_aviation_upper, fair1_tech2_gold.T_aviation_lower, color=colors_shade[4], alpha=0.5)
        ax3.fill_between(dates, fair1_tech2_gold.T_avCDR_upper, fair1_tech2_gold.T_avCDR_lower, color=colors_shade[0], alpha=0.5)
        ax3.fill_between(dates, fair1_tech2_silver.T_avCDR_upper, fair1_tech2_silver.T_avCDR_lower, color=colors_shade[1], alpha=0.5)
        ax3.fill_between(dates, fair1_tech2_bronze.T_avCDR_upper, fair1_tech2_bronze.T_avCDR_lower, color=colors_shade[2], alpha=0.5)
        if fair_ref is not None:
            ax3.plot(dates, fair_ref.T_baseline, '--k', label='1.5°C-compatible')
        ax3.plot(dates, fair1_tech2_gold.T_aviation, color=colors[4], label='No neutrality', linewidth=l_width)
        ax3.plot(dates, fair1_tech2_gold.T_avCDR, color=colors[0], label=baseline_1, linewidth=l_width)
        ax3.plot(dates, fair1_tech2_silver.T_avCDR, color=colors[1], label=baseline_2, linewidth=l_width)
        ax3.plot(dates, fair1_tech2_bronze.T_avCDR, color=colors[2], label=baseline_3, linewidth=l_width)

        ax4.fill_between(dates, fair3_gold.T_aviation_upper, fair3_gold.T_aviation_lower, color =colors_shade[4], alpha = 0.5)
        ax4.fill_between(dates, fair3_gold.T_avCDR_upper, fair3_gold.T_avCDR_lower, color=colors_shade[0], alpha=0.5)
        ax4.fill_between(dates, fair3_silver.T_avCDR_upper, fair3_silver.T_avCDR_lower, color=colors_shade[1], alpha=0.5)
        ax4.fill_between(dates, fair3_bronze.T_avCDR_upper, fair3_bronze.T_avCDR_lower, color=colors_shade[2], alpha=0.5)
        if fair3_EWF is not None:
            ax4.fill_between(dates, fair3_EWF.T_avCDR_upper, fair3_EWF.T_avCDR_lower, color=colors[3], alpha=0.3)
        if fair_ref is not None:
            ax4.plot(dates, fair_ref.T_baseline, '--k', label = '1.5°C-compatible')
        ax4.plot(dates, fair3_gold.T_aviation, color=colors[4], label='No neutrality', linewidth=l_width)
        ax4.plot(dates, fair3_gold.T_avCDR, color=colors[0], label=baseline_1, linewidth = l_width)
        ax4.plot(dates, fair3_silver.T_avCDR, color=colors[1], label=baseline_2, linewidth = l_width)
        ax4.plot(dates, fair3_bronze.T_avCDR, color=colors[2], label=baseline_3, linewidth = l_width)
        if fair3_EWF is not None:
            ax4.plot(dates, fair3_EWF.T_avCDR, color=colors[3], label=baseline_4, linewidth = l_width)

        ax5.fill_between(dates, fair3_tech1_gold.T_aviation_upper, fair3_tech1_gold.T_aviation_lower, color=colors_shade[4],
                         alpha=0.5)
        ax5.fill_between(dates, fair3_tech1_gold.T_avCDR_upper, fair3_tech1_gold.T_avCDR_lower, color=colors_shade[0],
                         alpha=0.5)
        ax5.fill_between(dates, fair3_tech1_silver.T_avCDR_upper, fair3_tech1_silver.T_avCDR_lower, color=colors_shade[1],
                         alpha=0.5)
        ax5.fill_between(dates, fair3_tech1_bronze.T_avCDR_upper, fair3_tech1_bronze.T_avCDR_lower, color=colors_shade[2],
                         alpha=0.5)
        if fair_ref is not None:
            ax5.plot(dates, fair_ref.T_baseline, '--k', label='1.5°C-compatible')
        ax5.plot(dates, fair3_tech1_gold.T_aviation, color=colors[4], label='No neutrality', linewidth=l_width)
        ax5.plot(dates, fair3_tech1_gold.T_avCDR, color=colors[0], label=baseline_1, linewidth=l_width)
        ax5.plot(dates, fair3_tech1_silver.T_avCDR, color=colors[1], label=baseline_2, linewidth=l_width)
        ax5.plot(dates, fair3_tech1_bronze.T_avCDR, color=colors[2], label=baseline_3, linewidth=l_width)

        ax6.fill_between(dates, fair3_tech2_gold.T_aviation_upper, fair3_tech2_gold.T_aviation_lower, color=colors_shade[4],
                         alpha=0.5)
        ax6.fill_between(dates, fair3_tech2_gold.T_avCDR_upper, fair3_tech2_gold.T_avCDR_lower, color=colors_shade[0],
                         alpha=0.5)
        ax6.fill_between(dates, fair3_tech2_silver.T_avCDR_upper, fair3_tech2_silver.T_avCDR_lower, color=colors_shade[1],
                         alpha=0.5)
        ax6.fill_between(dates, fair3_tech2_bronze.T_avCDR_upper, fair3_tech2_bronze.T_avCDR_lower, color=colors_shade[2],
                         alpha=0.5)
        if fair_ref is not None:
            ax6.plot(dates, fair_ref.T_baseline, '--k', label='1.5°C-compatible')
        ax6.plot(dates, fair3_tech2_gold.T_aviation, color=colors[4], label='No neutrality', linewidth=l_width)
        ax6.plot(dates, fair3_tech2_gold.T_avCDR, color=colors[0], label=baseline_1, linewidth=l_width)
        ax6.plot(dates, fair3_tech2_silver.T_avCDR, color=colors[1], label=baseline_2, linewidth=l_width)
        ax6.plot(dates, fair3_tech2_bronze.T_avCDR, color=colors[2], label=baseline_3, linewidth=l_width)

        ylabel = "Aviation temperature change in 2100 (°C)"

        ax4.legend(bbox_to_anchor=(0.5, -0.05), loc="lower center",
                   bbox_transform=fig.transFigure, ncol=3, frameon=False)

    elif what == 'CDR':
        dates = np.arange(1990, end_year + 1)
        CDR_1_gold_upper = (unumpy.nominal_values(fair1_gold['Tot'])+unumpy.std_devs(fair1_gold['Tot']))/1000
        CDR_1_gold_lower = (unumpy.nominal_values(fair1_gold['Tot'])-unumpy.std_devs(fair1_gold['Tot']))/1000
        CDR_1_gold = (unumpy.nominal_values(fair1_gold['Tot']))/1000
        CDR_1_silver_upper = (unumpy.nominal_values(fair1_silver['Tot'])+unumpy.std_devs(fair1_silver['Tot']))/1000
        CDR_1_silver_lower = (unumpy.nominal_values(fair1_silver['Tot'])-unumpy.std_devs(fair1_silver['Tot']))/1000
        CDR_1_silver = (unumpy.nominal_values(fair1_silver['Tot']))/1000
        CDR_1_bronze_upper = (unumpy.nominal_values(fair1_bronze['Tot'])+unumpy.std_devs(fair1_bronze['Tot']))/1000
        CDR_1_bronze_lower = (unumpy.nominal_values(fair1_bronze['Tot'])-unumpy.std_devs(fair1_bronze['Tot']))/1000
        CDR_1_bronze = (unumpy.nominal_values(fair1_bronze['Tot']))/1000
        CDR_1_CO2only = (unumpy.nominal_values(fair1_bronze['CO2']))/1000
        CDR_3_gold_upper = (unumpy.nominal_values(fair3_gold['Tot'])+unumpy.std_devs(fair3_gold['Tot']))/1000
        CDR_3_gold_lower = (unumpy.nominal_values(fair3_gold['Tot'])-unumpy.std_devs(fair3_gold['Tot']))/1000
        CDR_3_gold = (unumpy.nominal_values(fair3_gold['Tot']))/1000
        CDR_3_silver_upper = (unumpy.nominal_values(fair3_silver['Tot'])+unumpy.std_devs(fair3_silver['Tot']))/1000
        CDR_3_silver_lower = (unumpy.nominal_values(fair3_silver['Tot'])-unumpy.std_devs(fair3_silver['Tot']))/1000
        CDR_3_silver = (unumpy.nominal_values(fair3_silver['Tot']))/1000
        CDR_3_bronze_upper = (unumpy.nominal_values(fair3_bronze['Tot'])+unumpy.std_devs(fair3_bronze['Tot']))/1000
        CDR_3_bronze_lower = (unumpy.nominal_values(fair3_bronze['Tot'])-unumpy.std_devs(fair3_bronze['Tot']))/1000
        CDR_3_bronze = (unumpy.nominal_values(fair3_bronze['Tot']))/1000
        CDR_3_CO2only = (unumpy.nominal_values(fair3_bronze['CO2']))/1000
        CDR_1_tech1_gold_upper = (unumpy.nominal_values(fair1_tech1_gold['Tot'])+unumpy.std_devs(fair1_tech1_gold['Tot']))/1000
        CDR_1_tech1_gold_lower = (unumpy.nominal_values(fair1_tech1_gold['Tot'])-unumpy.std_devs(fair1_tech1_gold['Tot']))/1000
        CDR_1_tech1_gold = (unumpy.nominal_values(fair1_tech1_gold['Tot']))/1000
        CDR_1_tech1_silver_upper = (unumpy.nominal_values(fair1_tech1_silver['Tot'])+unumpy.std_devs(fair1_tech1_silver['Tot']))/1000
        CDR_1_tech1_silver_lower = (unumpy.nominal_values(fair1_tech1_silver['Tot'])-unumpy.std_devs(fair1_tech1_silver['Tot']))/1000
        CDR_1_tech1_silver = (unumpy.nominal_values(fair1_tech1_silver['Tot']))/1000
        CDR_1_tech1_bronze_upper = (unumpy.nominal_values(fair1_tech1_bronze['Tot'])+unumpy.std_devs(fair1_tech1_bronze['Tot']))/1000
        CDR_1_tech1_bronze_lower = (unumpy.nominal_values(fair1_tech1_bronze['Tot'])-unumpy.std_devs(fair1_tech1_bronze['Tot']))/1000
        CDR_1_tech1_bronze = (unumpy.nominal_values(fair1_tech1_bronze['Tot']))/1000
        CDR_1_tech1_CO2only = (unumpy.nominal_values(fair1_tech1_bronze['CO2']))/1000
        CDR_3_tech1_gold_upper = (unumpy.nominal_values(fair3_tech1_gold['Tot'])+unumpy.std_devs(fair3_tech1_gold['Tot']))/1000
        CDR_3_tech1_gold_lower = (unumpy.nominal_values(fair3_tech1_gold['Tot'])-unumpy.std_devs(fair3_tech1_gold['Tot']))/1000
        CDR_3_tech1_gold = (unumpy.nominal_values(fair3_tech1_gold['Tot']))/1000
        CDR_3_tech1_silver_upper = (unumpy.nominal_values(fair3_tech1_silver['Tot'])+unumpy.std_devs(fair3_tech1_silver['Tot']))/1000
        CDR_3_tech1_silver_lower = (unumpy.nominal_values(fair3_tech1_silver['Tot'])-unumpy.std_devs(fair3_tech1_silver['Tot']))/1000
        CDR_3_tech1_silver = (unumpy.nominal_values(fair3_tech1_silver['Tot']))/1000
        CDR_3_tech1_bronze_upper = (unumpy.nominal_values(fair3_tech1_bronze['Tot'])+unumpy.std_devs(fair3_tech1_bronze['Tot']))/1000
        CDR_3_tech1_bronze_lower = (unumpy.nominal_values(fair3_tech1_bronze['Tot'])-unumpy.std_devs(fair3_tech1_bronze['Tot']))/1000
        CDR_3_tech1_bronze = (unumpy.nominal_values(fair3_tech1_bronze['Tot']))/1000
        CDR_3_tech1_CO2only = (unumpy.nominal_values(fair3_tech1_bronze['CO2']))/1000

        CDR_1_tech2_gold_upper = (unumpy.nominal_values(fair1_tech2_gold['Tot'])+unumpy.std_devs(fair1_tech2_gold['Tot']))/1000
        CDR_1_tech2_gold_lower = (unumpy.nominal_values(fair1_tech2_gold['Tot'])-unumpy.std_devs(fair1_tech2_gold['Tot']))/1000
        CDR_1_tech2_gold = (unumpy.nominal_values(fair1_tech2_gold['Tot']))/1000
        CDR_1_tech2_silver_upper = (unumpy.nominal_values(fair1_tech2_silver['Tot'])+unumpy.std_devs(fair1_tech2_silver['Tot']))/1000
        CDR_1_tech2_silver_lower = (unumpy.nominal_values(fair1_tech2_silver['Tot'])-unumpy.std_devs(fair1_tech2_silver['Tot']))/1000
        CDR_1_tech2_silver = (unumpy.nominal_values(fair1_tech2_silver['Tot']))/1000
        CDR_1_tech2_bronze_upper = (unumpy.nominal_values(fair1_tech2_bronze['Tot'])+unumpy.std_devs(fair1_tech2_bronze['Tot']))/1000
        CDR_1_tech2_bronze_lower = (unumpy.nominal_values(fair1_tech2_bronze['Tot'])-unumpy.std_devs(fair1_tech2_bronze['Tot']))/1000
        CDR_1_tech2_bronze = (unumpy.nominal_values(fair1_tech2_bronze['Tot']))/1000
        CDR_1_tech2_CO2only = (unumpy.nominal_values(fair1_tech2_bronze['CO2']))/1000
        CDR_3_tech2_gold_upper = (unumpy.nominal_values(fair3_tech2_gold['Tot'])+unumpy.std_devs(fair3_tech2_gold['Tot']))/1000
        CDR_3_tech2_gold_lower = (unumpy.nominal_values(fair3_tech2_gold['Tot'])-unumpy.std_devs(fair3_tech2_gold['Tot']))/1000
        CDR_3_tech2_gold = (unumpy.nominal_values(fair3_tech2_gold['Tot']))/1000
        CDR_3_tech2_silver_upper = (unumpy.nominal_values(fair3_tech2_silver['Tot'])+unumpy.std_devs(fair3_tech2_silver['Tot']))/1000
        CDR_3_tech2_silver_lower = (unumpy.nominal_values(fair3_tech2_silver['Tot'])-unumpy.std_devs(fair3_tech2_silver['Tot']))/1000
        CDR_3_tech2_silver = (unumpy.nominal_values(fair3_tech2_silver['Tot']))/1000
        CDR_3_tech2_bronze_upper = (unumpy.nominal_values(fair3_tech2_bronze['Tot'])+unumpy.std_devs(fair3_tech2_bronze['Tot']))/1000
        CDR_3_tech2_bronze_lower = (unumpy.nominal_values(fair3_tech2_bronze['Tot'])-unumpy.std_devs(fair3_tech2_bronze['Tot']))/1000
        CDR_3_tech2_bronze = (unumpy.nominal_values(fair3_tech2_bronze['Tot']))/1000
        CDR_3_tech2_CO2only = (unumpy.nominal_values(fair3_tech2_bronze['CO2']))/1000
        
        ax1.fill_between(dates, CDR_1_gold_upper, CDR_1_gold_lower, color=colors_shade[0], alpha=0.5)
        ax1.fill_between(dates, CDR_1_silver_upper, CDR_1_silver_lower,  color=colors_shade[1], alpha=0.5)
        ax1.fill_between(dates, CDR_1_bronze_upper, CDR_1_bronze_lower, color=colors_shade[2], alpha=0.5)
        ax1.plot(dates, CDR_1_CO2only, color=colors[4], label='CO$_2$ only', linewidth=l_width)
        ax1.plot(dates, CDR_1_gold, color=colors[0], label=baseline_1, linewidth = l_width)
        ax1.plot(dates, CDR_1_silver, color=colors[1], label=baseline_2, linewidth = l_width)
        ax1.plot(dates, CDR_1_bronze, color=colors[2], label=baseline_3, linewidth = l_width)

        ax2.fill_between(dates, CDR_1_tech1_gold_upper, CDR_1_tech1_gold_lower, color=colors_shade[0], alpha=0.5)
        ax2.fill_between(dates, CDR_1_tech1_silver_upper, CDR_1_tech1_silver_lower, color=colors_shade[1], alpha=0.5)
        ax2.fill_between(dates, CDR_1_tech1_bronze_upper, CDR_1_tech1_bronze_lower, color=colors_shade[2], alpha=0.5)
        ax2.plot(dates, CDR_1_tech1_CO2only, color=colors[4], label='No neutrality', linewidth=l_width)
        ax2.plot(dates, CDR_1_tech1_gold, color=colors[0], label=baseline_1, linewidth=l_width)
        ax2.plot(dates, CDR_1_tech1_silver, color=colors[1], label=baseline_2, linewidth=l_width)
        ax2.plot(dates, CDR_1_tech1_bronze, color=colors[2], label=baseline_3, linewidth=l_width)

        ax3.fill_between(dates, CDR_1_tech2_gold_upper, CDR_1_tech2_gold_lower, color=colors_shade[0], alpha=0.5)
        ax3.fill_between(dates, CDR_1_tech2_silver_upper, CDR_1_tech2_silver_lower, color=colors_shade[1], alpha=0.5)
        ax3.fill_between(dates, CDR_1_tech2_bronze_upper, CDR_1_tech2_bronze_lower, color=colors_shade[2], alpha=0.5)
        ax3.plot(dates, CDR_1_tech2_CO2only, color=colors[4], label='No neutrality', linewidth=l_width)
        ax3.plot(dates, CDR_1_tech2_gold, color=colors[0], label=baseline_1, linewidth=l_width)
        ax3.plot(dates, CDR_1_tech2_silver, color=colors[1], label=baseline_2, linewidth=l_width)
        ax3.plot(dates, CDR_1_tech2_bronze, color=colors[2], label=baseline_3, linewidth=l_width)

        ax4.fill_between(dates, CDR_3_gold_upper, CDR_3_gold_lower, color=colors_shade[0], alpha=0.5)
        ax4.fill_between(dates, CDR_3_silver_upper, CDR_3_silver_lower,  color=colors_shade[1], alpha=0.5)
        ax4.fill_between(dates, CDR_3_bronze_upper, CDR_3_bronze_lower, color=colors_shade[2], alpha=0.5)
        ax4.plot(dates, CDR_3_CO2only, color=colors[4], label='CO$_2$ only', linewidth=l_width)
        ax4.plot(dates, CDR_3_gold, color=colors[0], label=baseline_1, linewidth = l_width)
        ax4.plot(dates, CDR_3_silver, color=colors[1], label=baseline_2, linewidth = l_width)
        ax4.plot(dates, CDR_3_bronze, color=colors[2], label=baseline_3, linewidth = l_width)

        ax5.fill_between(dates, CDR_3_tech1_gold_upper, CDR_3_tech1_gold_lower, color=colors_shade[0], alpha=0.5)
        ax5.fill_between(dates, CDR_3_tech1_silver_upper, CDR_3_tech1_silver_lower, color=colors_shade[1], alpha=0.5)
        ax5.fill_between(dates, CDR_3_tech1_bronze_upper, CDR_3_tech1_bronze_lower, color=colors_shade[2], alpha=0.5)
        ax5.plot(dates, CDR_3_tech1_CO2only, color=colors[4], label='No neutrality', linewidth=l_width)
        ax5.plot(dates, CDR_3_tech1_gold, color=colors[0], label=baseline_1, linewidth=l_width)
        ax5.plot(dates, CDR_3_tech1_silver, color=colors[1], label=baseline_2, linewidth=l_width)
        ax5.plot(dates, CDR_3_tech1_bronze, color=colors[2], label=baseline_3, linewidth=l_width)

        ax6.fill_between(dates, CDR_3_tech2_gold_upper, CDR_3_tech2_gold_lower, color=colors_shade[0], alpha=0.5)
        ax6.fill_between(dates, CDR_3_tech2_silver_upper, CDR_3_tech2_silver_lower, color=colors_shade[1], alpha=0.5)
        ax6.fill_between(dates, CDR_3_tech2_bronze_upper, CDR_3_tech2_bronze_lower, color=colors_shade[2], alpha=0.5)
        ax6.plot(dates, CDR_3_tech2_CO2only, color=colors[4], label='No neutrality', linewidth=l_width)
        ax6.plot(dates, CDR_3_tech2_gold, color=colors[0], label=baseline_1, linewidth=l_width)
        ax6.plot(dates, CDR_3_tech2_silver, color=colors[1], label=baseline_2, linewidth=l_width)
        ax6.plot(dates, CDR_3_tech2_bronze, color=colors[2], label=baseline_3, linewidth=l_width)

        ax4.legend(bbox_to_anchor=(0.5, -0.035), loc="lower center",
                   bbox_transform=fig.transFigure, ncol=2, frameon=False)

        ylabel = "CO$_2$ removal rates (GtCO$_2$/yr)"



    ax1.text(0.05, 0.9, 'Fossil jet fuels', transform=ax1.transAxes, va="top", ha="left")
    ax2.text(0.05, 0.9, TECH_1, transform=ax2.transAxes, va="top", ha="left")
    ax3.text(0.05, 0.9, TECH_2, transform=ax3.transAxes, va="top", ha="left")
    ax4.text(0.05, 0.9, 'Fossil jet fuels', transform=ax4.transAxes, va="top", ha="left")
    ax5.text(0.05, 0.9, TECH_1, transform=ax5.transAxes, va="top", ha="left")
    ax6.text(0.05, 0.9, TECH_2, transform=ax6.transAxes, va="top", ha="left")

    ax1.text(0.5, 1.1, scenario1, transform=ax1.transAxes, va="top", ha="center")
    ax4.text(0.5, 1.1, scenario3, transform=ax4.transAxes, va="top", ha="center")

    #ax1.set_ylabel(ylabel)
    ax2.set_ylabel(ylabel)
    #ax3.set_ylabel(ylabel)
    ax1.set_xlim(start_year, end_year)
    ax2.set_xlim(start_year, end_year)
    ax3.set_xlim(start_year, end_year)
    ax4.set_xlim(start_year, end_year)
    ax5.set_xlim(start_year, end_year)
    ax6.set_xlim(start_year, end_year)

    ax1.set_ylim(low_lim1, up_lim1)
    ax2.set_ylim(low_lim1, up_lim2)
    ax3.set_ylim(low_lim3, up_lim3)
    ax4.set_ylim(low_lim1, up_lim1)
    ax5.set_ylim(low_lim2, up_lim2)
    ax6.set_ylim(low_lim3, up_lim3)

    ax4.set_yticks([])
    ax5.set_yticks([])
    ax6.set_yticks([])


    plt.tight_layout(pad=0.3)
    if fair1_EWF is not None:
        plt.savefig("Figures/comparison_fair_alltechs_EWF_"+what+"_scenarios.pdf", dpi=600, bbox_inches="tight")
    else:
        plt.savefig("Figures/comparison_fair_alltechs_" + what + "_scenarios.pdf", dpi=600, bbox_inches="tight")


def plot_summaryCDR_alltechs(summary_df_original, positive_df_original = None, negative_df_original = None,
                             scenario1='SSP1_26', scenario2='SSP2_45', scenario3='SSP5_85',
                             tech1=TECH_1, tech2=TECH_2, what = 'mean', scenario = 'A', plot_type = 'bar', CO2only = False,
                             size_p = 7, route = 'ZH-NY', km_route = 6326, passengers_route = 350, plots = 'new', ax = None):
    """
    Plot CDR mean or cumulative rates
    :param summary_df_original:
    :param positive_df_original:
    :param negative_df_original:
    :param scenario1:
    :param scenario2:
    :param scenario3:
    :param tech1:
    :param tech2:
    :param what:
    :param scenario:
    :param plot_type:
    :param CO2only:
    :param size_p:
    :param route:
    :param km_route:
    :param passengers_route:
    :return:
    """
    if plots == 'new':
        if what == 'mean':
            fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(3.5, 5.75), tight_layout = True)
        elif what == 'cumulative':
            fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(5.25, 5.75), tight_layout = True)
            plt.tight_layout(w_pad=0.5)
        ax1, ax3 = (
            ax.flatten()[0],
            ax.flatten()[1],
        )

    else:
        fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(3.75, 11))
        ax1, ax2, ax3 = (
            ax.flatten()[0],
            ax.flatten()[1],
            ax.flatten()[2],
        )
    if what == 'mean':
        ylabel = 'mean CO$_2$ removal rates (GtCO$_2$/yr)'
        txt = 'mean CDR'
        x = 'Climate neutrality'
        hue = 'Technology'
    elif what == 'cumulative':
        ylabel = 'cumulative CO$_2$ removal (GtCO$_2$)'
        txt = 'cumulative CDR'
        x = 'Climate neutrality'
        hue = 'Technology'
    elif what == 'EWF':
        ylabel = 'Emissions Weighting Factor'
        txt = 'EWF'
        x = 'Climate neutrality'
        hue = 'Technology'
    elif what == 'rates':
        ylabel = 'CO$_2$ removal (MtCO$_2$/yr)'
        txt = 'CO$_2$ removal'
        x = 'Year'
        hue = 'Climate neutrality'
    elif what == 'cost':
        txt = 'CO$_2$ removal'
        x = 'Climate neutrality'
        hue = 'Technology'
        ylabel = '$\Delta$cost on route ' + route + ' (US\$)'

    x_ticks_names = np.array([baseline_1, baseline_2, baseline_3])
    x_ticks = np.array([0, 1, 2])


    summary_df = summary_df_original.copy()
    if positive_df_original is not None:
        positive_df = positive_df_original.copy()
    if negative_df_original is not None:
       negative_df = negative_df_original.copy()


    if what != 'cost':
        if what == 'EWF':
            summary_df['summary_Tot_CDR'] = summary_df_original['EWF'].copy()
            summary_df['summary_Tot_CDR_std'] = summary_df_original['EWF_std'].copy()
        else:
            summary_df['summary_Tot_CDR'] = summary_df['summary_Tot_CDR'].copy() / 1000
            summary_df['summary_CO2_CDR'] = summary_df['summary_CO2_CDR'].copy() / 1000
            summary_df['summary_Tot_CDR_std'] = summary_df['summary_Tot_CDR_std'].copy() / 1000
        if positive_df_original is not None:
            positive_df['summary_Tot_CDR'] = positive_df['summary_Tot_CDR'].copy() / 1000
            positive_df['summary_CO2_CDR'] = positive_df['summary_CO2_CDR'].copy() / 1000
            positive_df['summary_Tot_CDR_std'] = positive_df['summary_Tot_CDR_std'].copy() / 1000
        if negative_df_original is not None:
            negative_df['summary_Tot_CDR'] = negative_df['summary_Tot_CDR'].copy() / 1000
            negative_df['summary_CO2_CDR'] = negative_df['summary_CO2_CDR'].copy() / 1000
            negative_df['summary_Tot_CDR_std'] = negative_df['summary_Tot_CDR_std'].copy() / 1000
    elif what == 'cost':
        summary_df['summary_Tot_CDR'] = summary_df[
                                               'summary_Tot_costkm'] * km_route / passengers_route  # tot extra cost per passenger
        summary_df['summary_Tot_CDR_std'] = summary_df[
                                                   'summary_Tot_costkm_std'] * km_route / passengers_route  # tot extra cost per passenger

    if plot_type == 'bar':
        add = 0.27
        x_std = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]) + np.array([-add, -add, -add, 0, 0, 0, add, add, add])
    else:
        add = 0.27
        x_std = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]) + np.array([-add, -add, -add, 0, 0, 0, add, add, add])


    if plot_type == 'bar':
        if positive_df_original is None:
            sns.barplot(ax=ax1, x=x, y='summary_Tot_CDR', hue=hue, data=summary_df[summary_df["Scenario"] == scenario1],
                        # color = col_scenario1)
                        palette=sns.dark_palette(col_scenario1_shade, reverse=True))
            if plots != 'new':
                sns.barplot(ax=ax2, x=x, y='summary_Tot_CDR', hue=hue,
                            data=summary_df[summary_df["Scenario"] == scenario2],
                            # color = col_scenario2)
                            palette=sns.dark_palette(col_scenario2_shade, reverse=True))
            sns.barplot(ax=ax3, x=x, y='summary_Tot_CDR', hue=hue,
                        data=summary_df[summary_df["Scenario"] == scenario3],
                        # color = col_scenario2)
                        palette=sns.dark_palette(col_scenario3_shade, reverse=True))
        else:
            sns.barplot(ax=ax1, x=x, y='summary_Tot_CDR', hue=hue, data=positive_df[positive_df["Scenario"] == scenario1],
                        # color = col_scenario1)
                        palette=sns.dark_palette(col_scenario1_shade, reverse=True))
            sns.barplot(ax=ax1, x=x, y='summary_Tot_CDR', hue=hue,
                        data=negative_df[negative_df["Scenario"] == scenario1],
                        # color = col_scenario1)
                        palette=sns.dark_palette(const_shadier, reverse=True))
            if plots != 'new':
                sns.barplot(ax=ax2, x=x, y='summary_Tot_CDR', hue=hue,
                            data=positive_df[positive_df["Scenario"] == scenario2],
                            # color = col_scenario2)
                            palette=sns.dark_palette(col_scenario2_shade, reverse=True))
                sns.barplot(ax=ax2, x=x, y='summary_Tot_CDR', hue=hue,
                            data=negative_df[negative_df["Scenario"] == scenario2],
                            # color = col_scenario2)
                            palette=sns.dark_palette(const_shadier, reverse=True))
            sns.barplot(ax=ax3, x=x, y='summary_Tot_CDR', hue=hue,
                        data=positive_df[positive_df["Scenario"] == scenario3],
                        # color = col_scenario2)
                        palette=sns.dark_palette(col_scenario3_shade, reverse=True))
            sns.barplot(ax=ax3, x=x, y='summary_Tot_CDR', hue=hue,
                        data=negative_df[negative_df["Scenario"] == scenario3],
                        # color = col_scenario2)
                        palette=sns.dark_palette(const_shadier, reverse=True))

        if CO2only == True:
            sns.barplot(ax=ax1, x=x, y='summary_CO2_CDR', hue=hue,
                        data=summary_df[summary_df["Scenario"] == scenario1],
                        # color = col_scenario1_shade)
                        palette=sns.dark_palette(col_scenario1_shade, reverse=True))
            if plots != 'new':
                sns.barplot(ax=ax2, x=x, y='summary_CO2_CDR', hue=hue,
                            data=summary_df[summary_df["Scenario"] == scenario2],
                            # color = col_scenario2_shade)
                            palette=sns.dark_palette(col_scenario2_shade, reverse=True))
            sns.barplot(ax=ax3, x=x, y='summary_CO2_CDR', hue=hue,
                        data=summary_df[summary_df["Scenario"] == scenario3],
                        # color = col_scenario2_shade)
                        palette=sns.dark_palette(col_scenario3_shade, reverse=True))
        if tech2 is not None:
            if CO2only == False:
                if positive_df_original is None:
                    legend_elements_1 = [Patch(facecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[0],
                                               label='Fossil jet fuels'),
                                         Patch(facecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[1],
                                               label=TECH_1),
                                         Patch(facecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[2],
                                               label=TECH_2),
                                         ]
                    legend_elements_2 = [Patch(facecolor=sns.dark_palette(col_scenario2_shade, reverse=True)[0],
                                               label='Fossil jet fuels'),
                                         Patch(facecolor=sns.dark_palette(col_scenario2_shade, reverse=True)[1],
                                               label=TECH_1),
                                         Patch(facecolor=sns.dark_palette(col_scenario2_shade, reverse=True)[2],
                                               label=TECH_2),
                                         ]
                    legend_elements_3 = [Patch(facecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[0],
                                               label='Fossil jet fuels'),
                                         Patch(facecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[1],
                                               label=TECH_1),
                                         Patch(facecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[2],
                                               label=TECH_2),
                                         ]
                elif positive_df_original is not None:
                    colors1 = [[sns.dark_palette(col_scenario1_shade, reverse=True)[0],
                                sns.dark_palette(const_shadier, reverse=True)[0]],
                               [sns.dark_palette(col_scenario1_shade, reverse=True)[1],
                                sns.dark_palette(const_shadier, reverse=True)[1]],
                               [sns.dark_palette(col_scenario1_shade, reverse=True)[2],
                                sns.dark_palette(const_shadier, reverse=True)[2]]]
                    colors2 = [[sns.dark_palette(col_scenario2_shade, reverse=True)[0],
                                sns.dark_palette(const_shadier, reverse=True)[0]],
                               [sns.dark_palette(col_scenario2_shade, reverse=True)[1],
                                sns.dark_palette(const_shadier, reverse=True)[1]],
                               [sns.dark_palette(col_scenario2_shade, reverse=True)[2],
                                sns.dark_palette(const_shadier, reverse=True)[2]]]
                    colors3 = [[sns.dark_palette(col_scenario3_shade, reverse=True)[0],
                                sns.dark_palette(const_shadier, reverse=True)[0]],
                               [sns.dark_palette(col_scenario3_shade, reverse=True)[1],
                                sns.dark_palette(const_shadier, reverse=True)[1]],
                               [sns.dark_palette(col_scenario3_shade, reverse=True)[2],
                                sns.dark_palette(const_shadier, reverse=True)[2]]]
                    categories = ['Fossil jet fuels (+/-)',
                                  TECH_1+' (+/-)',
                                  TECH_2+' (+/-)']
                    legend_dict1 = dict(zip(categories, colors1))
                    legend_dict2 = dict(zip(categories, colors2))
                    legend_dict3 = dict(zip(categories, colors3))
                    legend_elements_1 = []
                    legend_elements_2 = []
                    legend_elements_3 = []
                    for cat, col in legend_dict1.items():
                        legend_elements_1.append([mpatches.Patch(facecolor=c, label=cat) for c in col])
                    if plots != 'new':
                        for cat, col in legend_dict2.items():
                            legend_elements_2.append([mpatches.Patch(facecolor=c, label=cat) for c in col])
                    for cat, col in legend_dict3.items():
                        legend_elements_3.append([mpatches.Patch(facecolor=c, label=cat) for c in col])
                    legend_element_4 = [Line2D([0], [0], marker='o', ls='none', markeredgecolor = 'none',
                                               label='Net CDR',
                                                markerfacecolor= 'k',
                                                markersize=size_p)]


            else:
                legend_elements_1 = [Patch(facecolor=sns.dark_palette(col_scenario1, reverse=True)[0],
                                           label='Fossil jet fuels'),
                                     Patch(facecolor=sns.dark_palette(col_scenario1, reverse=True)[1],
                                           label=TECH_1),
                                     Patch(facecolor=sns.dark_palette(col_scenario1, reverse=True)[2],
                                           label=TECH_2),
                                     Patch(facecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[0],
                                           label='Fossil jet fuels (CO$_2$ only)'),
                                     Patch(facecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[1],
                                           label=TECH_1+' (CO$_2$ only)'),
                                     Patch(facecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[2],
                                           label=TECH_2+' (CO$_2$ only)'),
                                     ]
                legend_elements_2 = [Patch(facecolor=sns.dark_palette(col_scenario2, reverse=True)[0],
                                           label='Fossil jet fuels'),
                                     Patch(facecolor=sns.dark_palette(col_scenario2, reverse=True)[1],
                                           label=TECH_1),
                                     Patch(facecolor=sns.dark_palette(col_scenario2, reverse=True)[2],
                                           label=TECH_2),
                                     Patch(facecolor=sns.dark_palette(col_scenario2_shade, reverse=True)[0],
                                           label='Fossil jet fuels (CO$_2$ only)'),
                                     Patch(facecolor=sns.dark_palette(col_scenario2_shade, reverse=True)[1],
                                           label=TECH_1 + ' (CO$_2$ only)'),
                                     Patch(facecolor=sns.dark_palette(col_scenario2_shade, reverse=True)[2],
                                           label=TECH_2 + ' (CO$_2$ only)'),
                                     ]
                legend_elements_3 = [Patch(facecolor=sns.dark_palette(col_scenario3, reverse=True)[0],
                                           label='Fossil jet fuels'),
                                     Patch(facecolor=sns.dark_palette(col_scenario3, reverse=True)[1],
                                           label=TECH_1),
                                     Patch(facecolor=sns.dark_palette(col_scenario3, reverse=True)[2],
                                           label=TECH_2),
                                     Patch(facecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[0],
                                           label='Fossil jet fuels (CO$_2$ only)'),
                                     Patch(facecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[1],
                                           label=TECH_1 + ' (CO$_2$ only)'),
                                     Patch(facecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[2],
                                           label=TECH_2 + ' (CO$_2$ only)'),
                                     ]


        elif tech2 is None:
            legend_elements_1 = [Patch(facecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[0], label='Tot'),
                                 Patch(facecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[1],
                                       label='Tot' + TECH_1),
                                 Patch(facecolor=sns.dark_palette(col_scenario1, reverse=True)[0], label='CO$_2$'),
                                 Patch(facecolor=sns.dark_palette(col_scenario1, reverse=True)[1],
                                       label='CO$_2$' + TECH_1),
                                 ]
            legend_elements_2 = [Patch(facecolor=sns.dark_palette(col_scenario2_shade, reverse=True)[0], label='Tot'),
                                 Patch(facecolor=sns.dark_palette(col_scenario2_shade, reverse=True)[1],
                                       label='Tot' + TECH_1),
                                 Patch(facecolor=sns.dark_palette(col_scenario2, reverse=True)[0], label='CO$_2$'),
                                 Patch(facecolor=sns.dark_palette(col_scenario2, reverse=True)[1],
                                       label='CO$_2$' + TECH_1),
                                 ]
            legend_elements_3 = [Patch(facecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[0], label='Tot'),
                                 Patch(facecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[1],
                                       label='Tot' + TECH_1),
                                 Patch(facecolor=sns.dark_palette(col_scenario3, reverse=True)[0], label='CO$_2$'),
                                 Patch(facecolor=sns.dark_palette(col_scenario3, reverse=True)[1],
                                       label='CO$_2$' + TECH_1),
                                 ]

    elif plot_type == 'point':
        if what != 'cost' and what != 'EWF':
            ax1.hlines(0, -0.4, 2.4, linestyles='dashed', colors= const_shade)
            if plots != 'new':
                ax2.hlines(0, -0.4, 2.4, linestyles='dashed', colors=const_shade)
            ax3.hlines(0, -0.4, 2.4, linestyles='dashed', colors=const_shade)
        elif what == 'EWF':
            ax1.hlines(2, -0.4, 2.4, linestyles='dashed', colors=const_shade)
            if plots != 'new':
                ax2.hlines(2, -0.4, 2.4, linestyles='dashed', colors=const_shade)
            ax3.hlines(2, -0.4, 2.4, linestyles='dashed', colors=const_shade)
        sns.swarmplot(ax=ax1, x=x, y='summary_Tot_CDR', hue=hue, data=summary_df[summary_df["Scenario"] == scenario1],
                    #color = col_scenario1)
                    palette=sns.dark_palette(col_scenario1_shade, reverse=True),
                    dodge = True, size = size_p)
        if plots != 'new':
            sns.swarmplot(ax=ax2, x=x, y='summary_Tot_CDR', hue=hue,
                        data=summary_df[summary_df["Scenario"] == scenario2],
                        dodge=True, size = size_p,
                        #color = col_scenario2)
                        palette=sns.dark_palette(col_scenario2_shade, reverse=True))
        sns.swarmplot(ax=ax3, x=x, y='summary_Tot_CDR', hue=hue,
                    data=summary_df[summary_df["Scenario"] == scenario3],
                    dodge=True, size = size_p,
                    palette=sns.dark_palette(col_scenario3_shade, reverse=True))

        legend_elements_1 = [
            Line2D([0], [0], marker='o', color='w', label='Fossil jet fuels',
                   markerfacecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[0], markersize= size_p),
            Line2D([0], [0], marker='o', color='w', label=TECH_1,
                   markerfacecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[1], markersize = size_p),
            Line2D([0], [0], marker='o', color='w', label=TECH_2,
                   markerfacecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[2], markersize = size_p),
            ]
        legend_elements_2 = [
            Line2D([0], [0], marker='o', color='w', label='Fossil jet fuels',
                   markerfacecolor=sns.dark_palette(col_scenario2_shade, reverse=True)[0], markersize = size_p),
            Line2D([0], [0], marker='o', color='w', label=TECH_1,
                   markerfacecolor=sns.dark_palette(col_scenario2_shade, reverse=True)[1], markersize = size_p),
            Line2D([0], [0], marker='o', color='w', label=TECH_2,
                   markerfacecolor=sns.dark_palette(col_scenario2_shade, reverse=True)[2], markersize = size_p),
            ]
        legend_elements_3 = [
            Line2D([0], [0], marker='o', color='w', label='Fossil jet fuels',
                   markerfacecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[0], markersize = size_p),
            Line2D([0], [0], marker='o', color='w', label=TECH_1,
                   markerfacecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[1], markersize = size_p),
            Line2D([0], [0], marker='o', color='w', label=TECH_2,
                   markerfacecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[2], markersize = size_p),
            ]
        legend_element_5 = [
            Line2D([0], [0], linestyle = '--', color=const_shade, label='EWF = 2'),
        ]

    if positive_df_original is not None:
        ax1.errorbar(x=x_std, y=summary_df[summary_df["Scenario"] == scenario1]['summary_Tot_CDR'],
                     yerr=summary_df[summary_df["Scenario"] == scenario1]['summary_Tot_CDR_std'],
                     fmt = 'o', c='black', label = 'cumulative CDR',
                     capsize=2)
        if plots != 'new':
            ax2.errorbar(x=x_std, y=summary_df[summary_df["Scenario"] == scenario2]['summary_Tot_CDR'],
                         yerr=summary_df[summary_df["Scenario"] == scenario2]['summary_Tot_CDR_std'],
                         c='black', fmt = 'o', label = 'cumulative CDR',
                         capsize=2)
        ax3.errorbar(x=x_std, y=summary_df[summary_df["Scenario"] == scenario3]['summary_Tot_CDR'],
                     yerr=summary_df[summary_df["Scenario"] == scenario3]['summary_Tot_CDR_std'],
                     c='black', fmt = 'o', label = 'cumulative CDR',
                     capsize=2)

    elif positive_df_original is None:
        ax1.errorbar(x=x_std, y=summary_df[summary_df["Scenario"] == scenario1]['summary_Tot_CDR'],
                     yerr=summary_df[summary_df["Scenario"] == scenario1]['summary_Tot_CDR_std'],
                     c='black', fmt='none',
                     capsize=2)
        if plots != 'new':
            ax2.errorbar(x=x_std, y=summary_df[summary_df["Scenario"] == scenario2]['summary_Tot_CDR'],
                         yerr=summary_df[summary_df["Scenario"] == scenario2]['summary_Tot_CDR_std'],
                         c='black', fmt='none',
                         capsize=2)

        ax3.errorbar(x=x_std, y=summary_df[summary_df["Scenario"] == scenario3]['summary_Tot_CDR'],
                     yerr=summary_df[summary_df["Scenario"] == scenario3]['summary_Tot_CDR_std'],
                     c='black', fmt='none',
                     capsize=2)



    if plots != 'new':
        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)
        ax3.set_ylabel(ylabel)
        ax1.set(xlabel=None)
        ax2.set(xlabel=None)
        ax3.set(xlabel=None)
    else:
        ax1.set_ylabel(ylabel=None)
        ax3.set_ylabel(ylabel=None)
        ax1.set(xlabel=None)
        ax3.set(xlabel=None)
        #plt.tick_params(labelcolor='none', axis='y', which='both', top=False, bottom=False, left=False, right=False)
        fig.supylabel(ylabel)

    ax1.set_xticks(x_ticks)
    if plots != 'new':
        ax2.set_xticks(x_ticks)
    ax3.set_xticks(x_ticks)

    if plots != 'new':
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.95, box.height])
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.95, box.height])

    ax1.set_xticklabels(x_ticks_names)
    if plots != 'new':
        ax2.set_xticklabels(x_ticks_names)
    ax3.set_xticklabels(x_ticks_names)

    if positive_df_original is not None:
        legend1 = ax1.legend(handles = legend_element_4, loc="upper right", frameon = False)
        if plots != 'new':
            legend2 = ax2.legend(handles = legend_element_4, loc="upper right", frameon=False)
        legend3 = ax3.legend(handles = legend_element_4,  loc="upper right", frameon=False)

        ax1.legend(handles=legend_elements_1, labels = categories, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0,
                ncol=1, frameon = False , handler_map = {list: HandlerTuple(None)})
        if plots != 'new':
            ax2.legend(handles=legend_elements_2, labels = categories, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0,
                    ncol=1, frameon = False, handler_map = {list: HandlerTuple(None)})
        ax3.legend(handles=legend_elements_3, labels = categories, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0,
                ncol=1, frameon = False, handler_map = {list: HandlerTuple(None)})

        ax1.add_artist(legend1)
        if plots != 'new':
            ax2.add_artist(legend2)
        ax3.add_artist(legend3)

    else:
        if what == 'EWF':
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])
            if plots != 'new':
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])
            box = ax3.get_position()
            ax3.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])

            legend1 = ax1.legend(handles=legend_element_5, loc="upper right", frameon=False)
            if plots != 'new':
                legend2 = ax2.legend(handles=legend_element_5, loc="upper right", frameon=False)
            legend3 = ax3.legend(handles=legend_element_5, loc="upper right", frameon=False)

            ax1.legend(handles=legend_elements_1, bbox_to_anchor=(0.5, -0.35), loc="lower center",
                       ncol=3, frameon=False)
            if plots != 'new':
                ax2.legend(handles=legend_elements_2, bbox_to_anchor=(0.5, -0.35), loc="lower center",
                        ncol=3, frameon=False)
            ax3.legend(handles=legend_elements_3, bbox_to_anchor=(0.5, -0.35), loc="lower center",
                       ncol=3, frameon=False)
            ax1.add_artist(legend1)
            if plots != 'new':
                ax2.add_artist(legend2)
            ax3.add_artist(legend3)
        else:
            ax1.legend(handles=legend_elements_1, loc="upper right", ncol=1, frameon=False)
            if plots != 'new':
                ax2.legend(handles=legend_elements_2, loc="upper right", ncol=1, frameon=False)
            ax3.legend(handles=legend_elements_3, loc="upper right", ncol=1, frameon=False)


    if plots != 'new':
        ax2.text(0.05, 0.05, 'SSP2-4.5', transform=ax2.transAxes, color=col_scenario2, fontweight='bold', va="bottom", ha="left")
    ax1.text(0.05, 0.05, 'SSP1-2.6', transform=ax1.transAxes, color=col_scenario1, fontweight='bold', va="bottom",
             ha="left")
    ax3.text(0.05, 0.05, 'SSP5-8.5', transform=ax3.transAxes, color=col_scenario3, fontweight='bold', va="bottom",
             ha="left")
    #fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')

    #fig.tight_layout()
    fig.savefig("Figures/compare_"+what+"_CDR_alltechs_" + scenario + "_" + plot_type+"_"+plots+".png", dpi=850, bbox_inches="tight")


#================================= SUPPLEMENTARY FIGURES ===================================
def plot_input_scenarios_extended_alltechs(df1, df2, df3, df_ref, df1_tech1, df2_tech1, df3_tech1,
                                           df1_tech2, df2_tech2, df3_tech2, col1, col2, col3, col4,
                                           y1, y2, y3, y4, scenario1, scenario2, scenario3, ref_scenario, what='ptI'):
    fig, ax = plt.subplots(ncols=3, nrows=4, figsize=(9, 11))
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = (
        ax.flatten()[0],
        ax.flatten()[1],
        ax.flatten()[2],
        ax.flatten()[3],
        ax.flatten()[4],
        ax.flatten()[5],
        ax.flatten()[6],
        ax.flatten()[7],
        ax.flatten()[8],
        ax.flatten()[9],
        ax.flatten()[10],
        ax.flatten()[11],
    )
    ax1.plot(df1[col1],label=scenario1, color=col_scenario1 )
    ax1.plot(df2[col1], label=scenario2, color=col_scenario2)
    ax1.plot(df3[col1], label=scenario3, color = col_scenario3)
    ax1.plot(df_ref[col1], label=ref_scenario, color= col_scenarioref)
    ax2.plot(df1_tech1[col1],label=scenario1, color=col_scenario1 )
    ax2.plot(df2_tech1[col1], label=scenario2, color=col_scenario2)
    ax2.plot(df3_tech1[col1], label=scenario3, color = col_scenario3)
    ax2.plot(df_ref[col1], label=ref_scenario, color=col_scenarioref)
    ax3.plot(df1_tech2[col1],label=scenario1, color=col_scenario1 )
    ax3.plot(df2_tech2[col1], label=scenario2, color=col_scenario2)
    ax3.plot(df3_tech2[col1], label=scenario3, color = col_scenario3)
    ax3.plot(df_ref[col1], label=ref_scenario, color=col_scenarioref)
    ax4.plot(df1[col2],label=scenario1, color=col_scenario1 )
    ax4.plot(df2[col2], label=scenario2, color=col_scenario2)
    ax4.plot(df3[col2], label=scenario3, color = col_scenario3)
    ax4.plot(df_ref[col2], label=ref_scenario, color= col_scenarioref)
    ax5.plot(df1_tech1[col2],label=scenario1, color=col_scenario1 )
    ax5.plot(df2_tech1[col2], label=scenario2, color=col_scenario2)
    ax5.plot(df3_tech1[col2], label=scenario3, color = col_scenario3)
    ax5.plot(df_ref[col2], label=ref_scenario, color=col_scenarioref)
    ax6.plot(df1_tech2[col2],label=scenario1, color=col_scenario1 )
    ax6.plot(df2_tech2[col2], label=scenario2, color=col_scenario2)
    ax6.plot(df3_tech2[col2], label=scenario3, color = col_scenario3)
    ax6.plot(df_ref[col2], label=ref_scenario, color=col_scenarioref)
    ax7.plot(df1[col3],label=scenario1, color=col_scenario1 )
    ax7.plot(df2[col3], label=scenario2, color=col_scenario2)
    ax7.plot(df3[col3], label=scenario3, color = col_scenario3)
    ax7.plot(df_ref[col3], label=ref_scenario, color= col_scenarioref)
    ax8.plot(df1_tech1[col3],label=scenario1, color=col_scenario1 )
    ax8.plot(df2_tech1[col3], label=scenario2, color=col_scenario2)
    ax8.plot(df3_tech1[col3], label=scenario3, color = col_scenario3)
    ax8.plot(df_ref[col3], label=ref_scenario, color=col_scenarioref)
    ax9.plot(df1_tech2[col3],label=scenario1, color=col_scenario1 )
    ax9.plot(df2_tech2[col3], label=scenario2, color=col_scenario2)
    ax9.plot(df3_tech2[col3], label=scenario3, color = col_scenario3)
    ax9.plot(df_ref[col3], label=ref_scenario, color=col_scenarioref)
    ax10.plot(df1[col4],label=scenario1, color=col_scenario1 )
    ax10.plot(df2[col4], label=scenario2, color=col_scenario2)
    ax10.plot(df3[col4], label=scenario3, color = col_scenario3)
    ax10.plot(df_ref[col4], label=ref_scenario, color= col_scenarioref)
    ax11.plot(df1_tech1[col4],label=scenario1, color=col_scenario1 )
    ax11.plot(df2_tech1[col4], label=scenario2, color=col_scenario2)
    ax11.plot(df3_tech1[col4], label=scenario3, color = col_scenario3)
    ax11.plot(df_ref[col4], label=ref_scenario, color=col_scenarioref)
    ax12.plot(df1_tech2[col4],label=scenario1, color=col_scenario1 )
    ax12.plot(df2_tech2[col4], label=scenario2, color=col_scenario2)
    ax12.plot(df3_tech2[col4], label=scenario3, color = col_scenario3)
    ax12.plot(df_ref[col4], label=ref_scenario, color=col_scenarioref)

    ax1.text(0.5, 1.1, 'Fossil jet fuels',  transform=ax1.transAxes, va="top", ha="center" )
    ax2.text(0.5, 1.1, TECH_1, transform=ax2.transAxes, va="top", ha="center")
    ax3.text(0.5, 1.1, TECH_2, transform=ax3.transAxes, va="top", ha="center")

    ax1.set_ylabel(y1)
    ax4.set_ylabel(y2)
    ax7.set_ylabel(y3)
    ax10.set_ylabel(y4)

    if what == 'ptI':
        ax1.text(0.05, 0.95, 'CO$_2$', transform=ax1.transAxes, va="top", ha="left")
        ax2.text(0.05, 0.95, 'CO$_2$', transform=ax2.transAxes, va="top", ha="left")
        ax3.text(0.05, 0.95, 'CO$_2$', transform=ax3.transAxes, va="top", ha="left")
        ax4.text(0.05, 0.95, 'NO$_x$', transform=ax4.transAxes, va="top", ha="left")
        ax5.text(0.05, 0.95, 'NO$_x$', transform=ax5.transAxes, va="top", ha="left")
        ax6.text(0.05, 0.95, 'NO$_x$', transform=ax6.transAxes, va="top", ha="left")
        ax7.text(0.05, 0.95, 'Contrails', transform=ax7.transAxes, va="top", ha="left")
        ax8.text(0.05, 0.95, 'Contrails', transform=ax8.transAxes, va="top", ha="left")
        ax9.text(0.05, 0.95, 'Contrails', transform=ax9.transAxes, va="top", ha="left")
        ax10.text(0.05, 0.95, 'Soot', transform=ax10.transAxes, va="top", ha="left")
        ax11.text(0.05, 0.95, 'Soot', transform=ax11.transAxes, va="top", ha="left")
        ax12.text(0.05, 0.95, 'Soot', transform=ax12.transAxes, va="top", ha="left")

        ax1.set_ylim(-5,4300)
        ax2.set_ylim(-5, 4300)
        ax3.set_ylim(-5, 4300)
        ax4.set_ylim(0,5)
        ax5.set_ylim(0, 5)
        ax6.set_ylim(0, 5)
        ax7.set_ylim(0,8*10**11)
        ax8.set_ylim(0, 8 * 10 ** 11)
        ax9.set_ylim(0, 8 * 10 ** 11)
        ax10.set_ylim(-0.001,0.02)
        ax11.set_ylim(-0.001, 0.02)
        ax12.set_ylim(-0.001, 0.02)

    else:
        ax1.text(0.05, 0.95, 'SO$_2$', transform=ax1.transAxes, va="top", ha="left")
        ax2.text(0.05, 0.95, 'SO$_2$', transform=ax2.transAxes, va="top", ha="left")
        ax3.text(0.05, 0.95, 'SO$_2$', transform=ax3.transAxes, va="top", ha="left")
        ax4.text(0.05, 0.95, 'H$_2$O', transform=ax4.transAxes, va="top", ha="left")
        ax5.text(0.05, 0.95, 'H$_2$O', transform=ax5.transAxes, va="top", ha="left")
        ax6.text(0.05, 0.95, 'H$_2$O', transform=ax6.transAxes, va="top", ha="left")
        ax7.text(0.05, 0.95, 'Fuel', transform=ax7.transAxes, va="top", ha="left")
        ax8.text(0.05, 0.95, 'Fuel', transform=ax8.transAxes, va="top", ha="left")
        ax9.text(0.05, 0.95, 'Fuel', transform=ax9.transAxes, va="top", ha="left")
        ax10.text(0.05, 0.95, 'Distance', transform=ax10.transAxes, va="top", ha="left")
        ax11.text(0.05, 0.95, 'Distance', transform=ax11.transAxes, va="top", ha="left")
        ax12.text(0.05, 0.95, 'Distance', transform=ax12.transAxes, va="top", ha="left")

        ax1.set_ylim(-0.05,0.82)
        ax2.set_ylim(-0.05,0.82)
        ax3.set_ylim(-0.05,0.82)
        ax4.set_ylim(-0.05*10**3,1.8*10**3)
        ax5.set_ylim(-0.05*10**3,1.8*10**3)
        ax6.set_ylim(-0.05*10**3,1.8*10**3)
        ax7.set_ylim(-10, 1400)
        ax8.set_ylim(-10, 1400)
        ax9.set_ylim(-10, 1400)
        ax10.set_ylim(-10**6,6.2*10**11)
        ax11.set_ylim(-10**6,6.2*10**11)
        ax12.set_ylim(-10**6,6.2*10**11)


    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width* 0.95, box.height * 0.95])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.95, box.height * 0.95])
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.95, box.height * 0.95])
    box = ax4.get_position()
    ax4.set_position([box.x0, box.y0, box.width * 0.95, box.height * 0.95])
    box = ax5.get_position()
    ax5.set_position([box.x0, box.y0, box.width * 0.95, box.height * 0.95])
    box = ax6.get_position()
    ax6.set_position([box.x0, box.y0, box.width * 0.95, box.height * 0.95])
    box = ax7.get_position()
    ax7.set_position([box.x0, box.y0, box.width* 0.95, box.height * 0.95])
    box = ax8.get_position()
    ax8.set_position([box.x0, box.y0, box.width * 0.95, box.height * 0.95])
    box = ax9.get_position()
    ax9.set_position([box.x0, box.y0, box.width * 0.95, box.height * 0.95])
    box = ax10.get_position()
    ax10.set_position([box.x0, box.y0, box.width * 0.95, box.height * 0.95])
    box = ax11.get_position()
    ax11.set_position([box.x0, box.y0, box.width * 0.95, box.height * 0.95])
    box = ax12.get_position()
    ax12.set_position([box.x0, box.y0, box.width * 0.95, box.height * 0.95])

    handles, labels = ax12.get_legend_handles_labels()

    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.04), loc="lower center",
                ncol=4, frameon=False)

    plt.tight_layout()
    plt.rcParams["axes.labelsize"] = 10
    plt.rc('xtick', labelsize = 9)
    plt.rc('ytick', labelsize=9)
    fig.savefig("Figures/Input_"+what+"_"+ col1 + "_" + col2 + "_" + col3 + "_" + col4 + ".png", dpi=850, bbox_inches="tight")


def compare_fair_aviation_allmetrics(fair_base_1, fair_base_3,
                                     fair_EWF_1, fair_EWF_3,
                                     fair_GWP100_1, fair_GWP100_3,
                                     fair_GWPstar_1, fair_GWPstar_3,
                                     fair_base_tech1_1, fair_base_tech1_3,
                                     fair_EWF_tech1_1, fair_EWF_tech1_3,
                                     fair_GWP100_tech1_1, fair_GWP100_tech1_3,
                                     fair_GWPstar_tech1_1, fair_GWPstar_tech1_3,
                                     fair_base_tech2_1,  fair_base_tech2_3,
                                     fair_EWF_tech2_1, fair_EWF_tech2_3,
                                     fair_GWP100_tech2_1, fair_GWP100_tech2_3,
                                     fair_GWPstar_tech2_1, fair_GWPstar_tech2_3,
                          scenario1, scenario3, what = 'T', end_year = 2100, start_year = 1990,
                          CDR_type = 'A', low_lim1 = None, up_lim1 = None, low_lim2 = None, up_lim2 = None,
                                  low_lim3 = None, up_lim3 = None,
                                     label1 = '$\sigma$', label2 ='EWF', label3 = 'GWP100', label4 = 'GWPstar',
                              palette = 'BrBG_r', l_width = 2):
    # Plot concentrations, forcing, and temperatures under CDR policy and references
    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(5, 6))
    ax1, ax4, ax2, ax5, ax3, ax6= (
        ax.flatten()[0],
        ax.flatten()[1],
        ax.flatten()[2],
        ax.flatten()[3],
        ax.flatten()[4],
        ax.flatten()[5]
    )
    dates = np.arange(1940, end_year+1)
    colors = sns.color_palette(palette, 5)

    if what == 'T':
        ax1.fill_between(dates, fair_base_1.T_aviation_upper, fair_base_1.T_aviation_lower, color =colors[4], alpha = 0.5)
        ax1.fill_between(dates, fair_EWF_1.T_aviation_upper, fair_EWF_1.T_aviation_lower, color=colors[0], alpha=0.5)
        ax1.fill_between(dates, fair_GWP100_1.T_aviation_upper, fair_GWP100_1.T_aviation_lower, color=colors[1], alpha=0.5)
        ax1.fill_between(dates, fair_GWPstar_1.T_aviation_upper, fair_GWPstar_1.T_aviation_lower, color=colors[2], alpha=0.5)

        ax1.plot(dates, fair_base_1.T_aviation, color=colors[4], label=label1, linewidth=l_width)
        ax1.plot(dates, fair_EWF_1.T_aviation, color=colors[0], label=label2, linewidth = l_width)
        ax1.plot(dates, fair_GWP100_1.T_aviation, color=colors[1], label=label3, linewidth = l_width)
        ax1.plot(dates, fair_GWPstar_1.T_aviation, color=colors[2], label=label4, linewidth = l_width)


        ax2.fill_between(dates, fair_base_tech1_1.T_aviation_upper, fair_base_tech1_1.T_aviation_lower, color =colors[4], alpha = 0.5)
        ax2.fill_between(dates, fair_EWF_tech1_1.T_aviation_upper, fair_EWF_tech1_1.T_aviation_lower, color=colors[0], alpha=0.5)
        ax2.fill_between(dates, fair_GWP100_tech1_1.T_aviation_upper, fair_GWP100_tech1_1.T_aviation_lower, color=colors[1], alpha=0.5)
        ax2.fill_between(dates, fair_GWPstar_tech1_1.T_aviation_upper, fair_GWPstar_tech1_1.T_aviation_lower, color=colors[2], alpha=0.5)

        ax2.plot(dates, fair_base_tech1_1.T_aviation, color=colors[4], label=label1, linewidth=l_width)
        ax2.plot(dates, fair_EWF_tech1_1.T_aviation, color=colors[0], label=label2, linewidth = l_width)
        ax2.plot(dates, fair_GWP100_tech1_1.T_aviation, color=colors[1], label=label3, linewidth = l_width)
        ax2.plot(dates, fair_GWPstar_tech1_1.T_aviation, color=colors[2], label=label4, linewidth = l_width)

        ax3.fill_between(dates, fair_base_tech2_1.T_aviation_upper, fair_base_tech2_1.T_aviation_lower, color =colors[4], alpha = 0.5)
        ax3.fill_between(dates, fair_EWF_tech2_1.T_aviation_upper, fair_EWF_tech2_1.T_aviation_lower, color=colors[0], alpha=0.5)
        ax3.fill_between(dates, fair_GWP100_tech2_1.T_aviation_upper, fair_GWP100_tech2_1.T_aviation_lower, color=colors[1], alpha=0.5)
        ax3.fill_between(dates, fair_GWPstar_tech2_1.T_aviation_upper, fair_GWPstar_tech2_1.T_aviation_lower, color=colors[2], alpha=0.5)

        ax3.plot(dates, fair_base_tech2_1.T_aviation, color=colors[4], label=label1, linewidth=l_width)
        ax3.plot(dates, fair_EWF_tech2_1.T_aviation, color=colors[0], label=label2, linewidth = l_width)
        ax3.plot(dates, fair_GWP100_tech2_1.T_aviation, color=colors[1], label=label3, linewidth = l_width)
        ax3.plot(dates, fair_GWPstar_tech2_1.T_aviation, color=colors[2], label=label4, linewidth = l_width)

        ax4.fill_between(dates, fair_base_3.T_aviation_upper, fair_base_3.T_aviation_lower, color=colors[4], alpha=0.5)
        ax4.fill_between(dates, fair_EWF_3.T_aviation_upper, fair_EWF_3.T_aviation_lower, color=colors[0], alpha=0.5)
        ax4.fill_between(dates, fair_GWP100_3.T_aviation_upper, fair_GWP100_3.T_aviation_lower, color=colors[1],
                         alpha=0.5)
        ax4.fill_between(dates, fair_GWPstar_3.T_aviation_upper, fair_GWPstar_3.T_aviation_lower, color=colors[2],
                         alpha=0.5)

        ax4.plot(dates, fair_base_3.T_aviation, color=colors[4], label=label1, linewidth=l_width)
        ax4.plot(dates, fair_EWF_3.T_aviation, color=colors[0], label=label2, linewidth=l_width)
        ax4.plot(dates, fair_GWP100_3.T_aviation, color=colors[1], label=label3, linewidth=l_width)
        ax4.plot(dates, fair_GWPstar_3.T_aviation, color=colors[2], label=label4, linewidth=l_width)

        ax5.fill_between(dates, fair_base_tech1_3.T_aviation_upper, fair_base_tech1_3.T_aviation_lower, color=colors[4],
                         alpha=0.5)
        ax5.fill_between(dates, fair_EWF_tech1_3.T_aviation_upper, fair_EWF_tech1_3.T_aviation_lower, color=colors[0],
                         alpha=0.5)
        ax5.fill_between(dates, fair_GWP100_tech1_3.T_aviation_upper, fair_GWP100_tech1_3.T_aviation_lower,
                         color=colors[1], alpha=0.5)
        ax5.fill_between(dates, fair_GWPstar_tech1_3.T_aviation_upper, fair_GWPstar_tech1_3.T_aviation_lower,
                         color=colors[2], alpha=0.5)

        ax5.plot(dates, fair_base_tech1_3.T_aviation, color=colors[4], label=label1, linewidth=l_width)
        ax5.plot(dates, fair_EWF_tech1_3.T_aviation, color=colors[0], label=label2, linewidth=l_width)
        ax5.plot(dates, fair_GWP100_tech1_3.T_aviation, color=colors[1], label=label3, linewidth=l_width)
        ax5.plot(dates, fair_GWPstar_tech1_3.T_aviation, color=colors[2], label=label4, linewidth=l_width)

        ax6.fill_between(dates, fair_base_tech2_3.T_aviation_upper, fair_base_tech2_3.T_aviation_lower, color=colors[4],
                         alpha=0.5)
        ax6.fill_between(dates, fair_EWF_tech2_3.T_aviation_upper, fair_EWF_tech2_3.T_aviation_lower, color=colors[0],
                         alpha=0.5)
        ax6.fill_between(dates, fair_GWP100_tech2_3.T_aviation_upper, fair_GWP100_tech2_3.T_aviation_lower,
                         color=colors[1], alpha=0.5)
        ax6.fill_between(dates, fair_GWPstar_tech2_3.T_aviation_upper, fair_GWPstar_tech2_3.T_aviation_lower,
                         color=colors[2], alpha=0.5)

        ax6.plot(dates, fair_base_tech2_3.T_aviation, color=colors[4], label=label1, linewidth=l_width)
        ax6.plot(dates, fair_EWF_tech2_3.T_aviation, color=colors[0], label=label2, linewidth=l_width)
        ax6.plot(dates, fair_GWP100_tech2_3.T_aviation, color=colors[1], label=label3, linewidth=l_width)
        ax6.plot(dates, fair_GWPstar_tech2_3.T_aviation, color=colors[2], label=label4, linewidth=l_width)

        ylabel = "$\Delta$ Temperature (°C)"


    ax1.text(0.05, 0.9, 'Fossil jet fuels', transform=ax1.transAxes, va="top", ha="left")
    ax2.text(0.05, 0.9, TECH_1, transform=ax2.transAxes, va="top", ha="left")
    ax3.text(0.05, 0.9, TECH_2, transform=ax3.transAxes, va="top", ha="left")
    ax4.text(0.05, 0.9, 'Fossil jet fuels', transform=ax4.transAxes, va="top", ha="left")
    ax5.text(0.05, 0.9, TECH_1, transform=ax5.transAxes, va="top", ha="left")
    ax6.text(0.05, 0.9, TECH_2, transform=ax6.transAxes, va="top", ha="left")

    ax1.text(0.5, 1.1, scenario1, transform=ax1.transAxes, va="top", ha="center")
    ax4.text(0.5, 1.1, scenario3, transform=ax4.transAxes, va="top", ha="center")

    ax1.set_ylabel(ylabel)
    ax2.set_ylabel(ylabel)
    ax3.set_ylabel(ylabel)
    ax1.set_xlim(start_year, end_year)
    ax2.set_xlim(start_year, end_year)
    ax3.set_xlim(start_year, end_year)
    ax4.set_xlim(start_year, end_year)
    ax5.set_xlim(start_year, end_year)
    ax6.set_xlim(start_year, end_year)

    ax1.set_ylim(low_lim1, up_lim1)
    ax2.set_ylim(low_lim1, up_lim2)
    ax3.set_ylim(low_lim3, up_lim3)
    ax4.set_ylim(low_lim1, up_lim1)
    ax5.set_ylim(low_lim2, up_lim2)
    ax6.set_ylim(low_lim3, up_lim3)

    ax4.set_yticks([])
    ax5.set_yticks([])
    ax6.set_yticks([])

    ax4.legend(bbox_to_anchor=(0.5,-0.01), loc="lower center",
                bbox_transform=fig.transFigure, ncol=4, frameon = False)
    plt.tight_layout()
    plt.savefig("Figures/comparison_fair_allmetrics_" + what + "_" + CDR_type + "_scenarios.png", dpi=850, bbox_inches="tight")


def compare_Toutcomes_deltat(ERF, ERF_gold, ERF_silver, ERF_bronze, df, df_gold, df_silver, df_bronze, scenario, palette,
                     start_year, end_year = 2100):
    # Plot concentrations, forcing, and temperatures under CDR policy and references

    dt = np.concatenate(([1], np.arange(5, 26, 5)))
    T_gold  = pd.DataFrame(columns = dt)
    T_silver = pd.DataFrame(columns = dt)
    T_bronze = pd.DataFrame(columns = dt)
    for i in dt:
        CDR_gold = make_CDR_metric(ERF - ERF_gold, df - df_gold, i, start_year)
        CDR_silver = make_CDR_metric(ERF - ERF_silver, df - df_silver, i, 2019)
        CDR_bronze = make_CDR_metric(ERF - ERF_bronze, df - df_bronze, i, start_year)
        fair_gold = test_CO2_Fair(df, CDR_gold, ERF, df_gold, ERF_gold,
                                      start_year, baseline='Gold')
        fair_silver = test_CO2_Fair(df, CDR_silver, ERF, df_silver, ERF_silver, 2019, baseline = 'SSP1_19')
        fair_bronze = test_CO2_Fair(df, CDR_bronze, ERF, df_bronze, ERF_bronze, start_year, baseline = 'Gold')
        T_gold[i] = fair_gold.T_avCDR
        T_silver[i] = fair_silver.T_avCDR
        T_bronze[i] = fair_bronze.T_avCDR
        T_gold['goal'] = fair_gold.T_baseline
        T_silver['goal'] = fair_silver.T_baseline
        T_bronze['goal'] = fair_bronze.T_baseline
        T_gold['BAU'] = fair_gold.T_aviation
        T_silver['BAU'] = fair_silver.T_aviation
        T_bronze['BAU'] = fair_bronze.T_aviation

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(9.5, 3))
    ax1, ax2, ax3= (
        ax.flatten()[0],
        ax.flatten()[1],
        ax.flatten()[2]
    )
    dates = np.arange(1940, end_year+1)
    colors = sns.color_palette(palette, 26)

    # Plot 1: Bronte
    ax1.plot(dates, T_bronze['BAU'], color='k', label=scenario)
    ax1.plot(dates, T_bronze['goal'], color='k', linestyle = 'dashed', label ='Bronze')
    for i in dt:
        ax1.plot(dates, T_bronze[i], color = colors[i], label = '$\Delta$t = '+str(i))

    ax2.plot(dates, T_silver['BAU'], color='k', label=scenario)
    ax2.plot(dates, T_silver['goal'], color='k', linestyle='dashed', label='Silver')
    for i in dt:
        ax2.plot(dates, T_silver[i], color=colors[i], label='$\Delta$t = ' + str(i))

    ax3.plot(dates, T_gold['BAU'], color='k', label=scenario)
    ax3.plot(dates, T_gold['goal'], color='k', linestyle='dashed', label='Gold')
    for i in dt:
        ax3.plot(dates, T_gold[i], color=colors[i], label='$\Delta$t = ' + str(i))

    ax1.set_ylabel("$\Delta$ Temperature (°C)")
    ax2.set_ylabel("$\Delta$ Temperature (°C)")
    ax3.set_ylabel("$\Delta$ Temperature (°C)")

    ax1.text(0.95, 0.95, 'Bronze', transform=ax1.transAxes, va="top", ha="right")
    ax2.text(0.95, 0.95, 'Silver', transform=ax2.transAxes, va="top", ha="right")
    ax3.text(0.95, 0.95, 'Gold', transform=ax3.transAxes, va="top", ha="right")

    ax1.set_xlim(2000, end_year)
    ax2.set_xlim(2000, end_year)
    ax3.set_xlim(2000, end_year)

    if scenario == scenario1:
        ax1.set_ylim(-0.015, 0.15)
        ax2.set_ylim(-0.015, 0.15)
        ax3.set_ylim(-0.015, 0.15)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()


    ax1.legend(handles = handles1[:2], labels = labels1[:2], loc = 'upper left', frameon = False)
    ax2.legend(handles = handles2[:2], labels = labels2[:2], loc = 'upper left', frameon = False)
    ax3.legend(handles = handles3[:2], labels = labels3[:2], loc = 'upper left', frameon = False)

    fig.legend(handles = handles1[2:], labels=  labels1[2:], bbox_to_anchor=(0.5, -0.1), loc="lower center",
               ncol=6, frameon=False)

    plt.rcParams["axes.labelsize"] = 11
    plt.tight_layout(pad=0.7)
    plt.savefig("Figures/deltat_temperature_"+scenario+".png", dpi=850, bbox_inches="tight")



def plot_climateneutrality_definitions(ERF, ERF_gold, ERF_silver, ERF_bronze, df, df_gold, df_silver, df_bronze,
                                       scenario, start_year, end_year = 2100, what = 'definitions', nonCO2 = 'Contrail',
                                       ylabel_nonCO2 = None):

    E_CO2eq_df = make_CDR_metric(ERF, df, 20, start_date = 2010, CO2only = False, metric ='GWP*', EF = None)
    CDR_bronze = make_CDR_metric(ERF-ERF_bronze, df-df_bronze, 20, start_date= 2019, CO2only = False, metric ='GWP*', EF = None )
    CDR_silver = make_CDR_metric(ERF - ERF_silver, df - df_silver, 20, start_date=2019, CO2only=False, metric='GWP*',
                                 EF=None)
    CDR_gold = make_CDR_metric(ERF - ERF_gold, df - df_gold, 20, start_date=2019, CO2only=False, metric='GWP*',
                                 EF=None)

    E_CO2eq_gold = make_CDR_metric(ERF_gold, df_gold, 20, start_date = 2010, CO2only = False, metric ='GWP*', EF = None)
    E_CO2eq_silver = make_CDR_metric(ERF_silver, df_silver, 20, start_date = 2010, CO2only = False, metric ='GWP*', EF = None)
    E_CO2eq_bronze = make_CDR_metric(ERF_bronze, df_bronze, 20, start_date = 2010, CO2only = False, metric ='GWP*', EF = None)

    # Plot concentrations, forcing, and temperatures under CDR policy and references
    fair_gold = test_CO2_Fair(df, CDR_gold, ERF, df_gold, ERF_gold,
                                  start_year, baseline='Gold')
    fair_silver = test_CO2_Fair(df, CDR_silver, ERF, df_silver, ERF_silver, 2019, baseline = 'SSP1_19')
    fair_bronze = test_CO2_Fair(df, CDR_bronze, ERF, df_bronze, ERF_bronze, start_year, baseline = 'Gold')
    T_gold = fair_gold.T_avCDR
    T_silver = fair_silver.T_avCDR
    T_bronze = fair_bronze.T_avCDR
    T_gold_goal = fair_gold.T_baseline
    T_silver_goal = fair_silver.T_baseline
    T_bronze_goal = fair_bronze.T_baseline
    T_gold_BAU = fair_gold.T_aviation
    T_silver_BAU = fair_silver.T_aviation
    T_bronze_BAU = fair_bronze.T_aviation

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(7.2, 2.5))
    ax1, ax2, ax3 = (
        ax.flatten()[0],
        ax.flatten()[1],
        ax.flatten()[2]
    )
    dates_short = np.arange(1990, end_year+1)
    dates = np.arange(1940, end_year+1)


    if what == 'definitions':
        # Plot 1: Emissions (CO2)
        ax1.plot(dates_short, unumpy.nominal_values(E_CO2eq_df['CO2']) / 1000, color='k', linestyle='dashed',
                 label=scenario)
        ax1.plot(dates_short, unumpy.nominal_values(E_CO2eq_gold['CO2']) / 1000, color=gold, label='Gold')
        ax1.plot(dates_short, unumpy.nominal_values(E_CO2eq_silver['CO2']) / 1000, color=silver, label='Silver')
        ax1.plot(dates_short, unumpy.nominal_values(E_CO2eq_bronze['CO2']) / 1000, color=bronze, label='Bronze')

        # Plot 2: Emissions (CO2-eq)
        if nonCO2 == 'Contrail':
            ax2.plot(dates_short, unumpy.nominal_values(df[nonCO2]) / 10**10, color='k', linestyle='dashed',
                     label=scenario)
            ax2.plot(dates_short, unumpy.nominal_values(df_gold[nonCO2]) / 10 ** 10, color=gold, label='Gold')
            ax2.plot(dates_short, unumpy.nominal_values(df_silver[nonCO2]) / 10 ** 10, color=silver, label='Silver')
            ax2.plot(dates_short, unumpy.nominal_values(df_bronze[nonCO2]) / 10 ** 10, color=bronze, label='Bronze')
            ax2.set_ylabel("Contrail cirrus (10$^{10}$ km/yr)")
        else:
            ax2.plot(dates_short, unumpy.nominal_values(df[nonCO2]), color='k', linestyle='dashed',
                     label=scenario)
            ax2.plot(dates_short, unumpy.nominal_values(df_gold[nonCO2]), color=gold, label='Gold')
            ax2.plot(dates_short, unumpy.nominal_values(df_silver[nonCO2]), color=silver, label='Silver')
            ax2.plot(dates_short, unumpy.nominal_values(df_bronze[nonCO2]), color=bronze, label='Bronze')
            ax2.set_ylabel(ylabel_nonCO2)

        # Plot 3: T outcomes
        ax3.plot(dates, T_silver_BAU, color='k', linestyle='dashed', label=scenario3)
        ax3.plot(dates, T_gold_goal, color=gold, label='Gold')
        ax3.plot(dates, T_silver_goal, color=silver, label='Silver')
        ax3.plot(dates, T_bronze_goal, color=bronze, label='Bronze')

        handles1, labels1 = ax1.get_legend_handles_labels()
        fig.legend(handles=handles1, labels=labels1, bbox_to_anchor=(0.5, -0.1), loc="lower center",
                   ncol=4, frameon=False)

        ax1.set_ylabel("CO$_2$ emissions (GtCO$_2$/yr)")
        ax3.set_ylabel("$\Delta$ Temperature (°C)")

        ax1.text(0.05, 0.95, 'CO$_2$', transform=ax1.transAxes, va="top", ha="left")
        ax2.text(0.05, 0.95, 'non-CO$_2$', transform=ax2.transAxes, va="top", ha="left")
        ax3.text(0.05, 0.95, 'Temperature', transform=ax3.transAxes, va="top", ha="left")

        ax1.set_xlim(2010, end_year)
        ax2.set_xlim(2010, end_year)
        ax3.set_xlim(2010, end_year)

    else:
        # Plot 1: Emissions (CO2)
        ax1.plot(dates_short, unumpy.nominal_values(E_CO2eq_df['CO2']) / 1000, color='k', linestyle='dashed',
                 label=scenario)
        ax1.plot(dates_short, unumpy.nominal_values(E_CO2eq_gold['CO2']) / 1000, color=gold, label='Gold')
        ax1.plot(dates_short, unumpy.nominal_values(E_CO2eq_silver['CO2']) / 1000, color=silver, label='Silver')
        ax1.plot(dates_short, unumpy.nominal_values(E_CO2eq_bronze['CO2']) / 1000, color=bronze, label='Bronze')
        # Plot 2: Emissions (CDR)
        ax2.plot(dates_short, - unumpy.nominal_values(CDR_gold['Tot']) / 1000,  color=gold, label='Gold')
        ax2.plot(dates_short, - unumpy.nominal_values(CDR_silver['Tot']) / 1000, color=silver, label='Silver')
        ax2.plot(dates_short, - unumpy.nominal_values(CDR_bronze['Tot']) / 1000, color=bronze, label='Bronze')
        # Plot 3: Emissions (tot CO2)
        ax3.plot(dates_short, unumpy.nominal_values(E_CO2eq_df['CO2']) / 1000 - unumpy.nominal_values(CDR_gold['Tot']) / 1000,  color=gold, label='Gold')
        ax3.plot(dates_short, unumpy.nominal_values(E_CO2eq_df['CO2']) / 1000 - unumpy.nominal_values(CDR_silver['Tot']) / 1000, color=silver, label='Silver')
        ax3.plot(dates_short, unumpy.nominal_values(E_CO2eq_df['CO2']) / 1000 - unumpy.nominal_values(CDR_bronze['Tot']) / 1000, color=bronze, label='Bronze')

        handles1, labels1 = ax1.get_legend_handles_labels()
        fig.legend(handles = handles1, labels=  labels1, bbox_to_anchor=(0.5, -0.1), loc="lower center",
                   ncol=4, frameon=False)

        ax1.set_ylabel("CO$_2$ emissions (GtCO$_2$/yr)")
        ax2.set_ylabel("Carbon removal rates (GtCO$_2$/yr)")
        ax3.set_ylabel("CO$_2$ + carbon removal (GtCO$_2$/yr)")

        ax1.text(0.05, 0.95, 'CO$_2$', transform=ax1.transAxes, va="top", ha="left")
        ax2.text(0.05, 0.95, 'Carbon removal', transform=ax2.transAxes, va="top", ha="left")
        ax3.text(0.05, 0.95, 'CO$_2$ + carbon removal', transform=ax3.transAxes, va="top", ha="left")

        ax1.set_xlim(2010, end_year)
        ax2.set_xlim(2010, end_year)
        ax3.set_xlim(2010, end_year)

    plt.rcParams["axes.labelsize"] = 11
    plt.tight_layout(h_pad=0.1, w_pad=0.5)
    plt.savefig("Figures/"+what+"_climate_neutrality_"+scenario+".pdf", dpi=650, bbox_inches="tight")

def plt_species_subplots(data, start_date, txt, low_lim, up_lim, ylabel, ax=None, palette=None):
    if palette is not None:
        colors = sns.color_palette(palette, 9)
    ax = ax or plt.gca()
    years = data.loc[str(start_date):, 'CO2'].index
    dates = np.arange(start=start_date, stop=start_date + len(years))
    netNOx = unumpy.nominal_values(data.loc[str(start_date):, ['O3 short', 'CH4', 'O3 long', 'SWV']].sum(axis=1))
    O3short = unumpy.nominal_values(data.loc[str(start_date):, 'O3 short'])
    CH4 = unumpy.nominal_values(data.loc[str(start_date):, 'CH4'])
    O3long = unumpy.nominal_values(data.loc[str(start_date):, 'O3 long'])
    SWV = unumpy.nominal_values(data.loc[str(start_date):, 'SWV'])
    BC = unumpy.nominal_values(data.loc[str(start_date):, 'BC'])
    SO4 = unumpy.nominal_values(data.loc[str(start_date):, 'SO4'])
    H2O = unumpy.nominal_values(data.loc[str(start_date):, 'H2O'])
    contrails = unumpy.nominal_values(data.loc[str(start_date):, 'Contrails and C-C'])
    CO2 = unumpy.nominal_values(data.loc[str(start_date):, 'CO2'])
    tot = unumpy.nominal_values(data.loc[str(start_date):, 'Tot'])
    tot_err = unumpy.std_devs(data.loc[str(start_date):, 'Tot'])
    ax.hlines(0, start_date, start_date + len(years), linestyles="dashed", color='k')
    ax.fill_between(dates, SO4, label='Sulfur aerosol', color=colors[3])
    ax.fill_between(dates, CH4 + SO4, SO4, label = 'CH$_4$ decrease', color = colors[2] )
    ax.fill_between(dates, CH4 + SO4 + O3long, CH4 + SO4, label='Long-term O$_3$ decrease', color=colors[1])
    ax.fill_between(dates, CH4 + SO4 + O3long + SWV, CH4 + SO4 + O3long , label='Stratospheric water vapour decrease', color=colors[0])
    ax.fill_between(dates, O3short, label='Short-term O$_3$ increase', color=colors[4], alpha = 0.9)
    ax.fill_between(dates, O3short + H2O, O3short , label='H$_2$O', color=colors[5], alpha = 0.9)
    ax.fill_between(dates, BC + O3short + H2O, O3short + H2O, label='Soot aerosol',
                    color=colors[6], alpha = 0.9)
    ax.fill_between(dates, CO2 + BC + O3short + H2O, BC + O3short + H2O, label='CO$_2$', color=colors[7],alpha= 0.85)
    ax.fill_between(dates, contrails + CO2 + BC + O3short + H2O, CO2 + BC + O3short + H2O, label='Contrail cirrus', color=colors[8], alpha= 0.85)
    ax.errorbar(dates[0::3], tot[0::3], yerr=tot_err[0::3], fmt='none', color = const, zorder = 1, label = 'Uncertainty')
    ax.scatter(dates, tot, label='Total aviation', marker='o', color='k', s=2)
    #ax.text(0.9, 0.95, txt, transform=ax.transAxes, va="top", ha="right")
    ax.set_ylim(low_lim, up_lim)
    ax.set_ylabel(ylabel)
    #ax.legend(loc = 'upper left', ncol = 2)
    return

def plot_species_subplots(data, start_date, txt, low_lim, up_lim, ylabel, ax=None):
    ax = ax or plt.gca()
    years = data.loc[str(start_date):, 'CO2'].index
    dates = np.arange(start=start_date, stop=start_date + len(years))
    netNOx = unumpy.nominal_values(data.loc[str(start_date):, ['O3 short', 'CH4', 'O3 long']].sum(axis=1))
    BC = unumpy.nominal_values(data.loc[str(start_date):, 'BC'])
    SO4 = unumpy.nominal_values(data.loc[str(start_date):, 'SO4'])
    H2O = unumpy.nominal_values(data.loc[str(start_date):, 'H2O'])
    contrails = unumpy.nominal_values(data.loc[str(start_date):, 'Contrails and C-C'])
    CO2 = unumpy.nominal_values(data.loc[str(start_date):, 'CO2'])
    tot = unumpy.nominal_values(data.loc[str(start_date):, 'Tot'])
    tot_err = unumpy.std_devs(data.loc[str(start_date):, 'Tot'])
    width = 0.7
    ax.bar(dates, SO4, width,  label='Sulfur aerosol', color = '#3E7194')
    ax.bar(dates, CO2, width, label='CO$_2$', color = '#b15301')
    ax.bar(dates, contrails, width, label='Contrail cirrus', bottom = CO2, color = '#FD7905')
    ax.bar(dates, netNOx, width, label='net NOx', bottom= contrails+CO2, color = '#f89b01')
    ax.bar(dates, H2O, width, label='H$_2$O', bottom= contrails+CO2+netNOx, color = '#feac24')
    ax.bar(dates, BC, width, label='Soot aerosol', bottom=contrails+CO2+H2O+netNOx, color = '#fed28a')
    ax.errorbar(dates, tot, yerr= tot_err, label='Tot', fmt='kx')
    #if txt == 'B1 care':
    #    ax.text(0.95, 0.95, txt, transform=ax.transAxes, va="top", ha="right")
    #else:
    #ax.text(0.9, 0.95, txt, transform=ax.transAxes, va="top", ha="right")
    ax.set_ylim(low_lim, up_lim)
    ax.set_ylabel(ylabel)
    #ax.legend(loc = 'upper left', ncol = 2)
    return

def plot_ERForCDR_scenarios(df1, df2, df3, scenario1, scenario2, scenario3, ref_scenario,
                            df_ref = None, what = 'ERF', CDR_type = 'all', low_lim = None, up_lim = None, start_date = 2018, type ='plot',palette = 'icefire'):
    if what == 'ERF':
        ylabel = 'ERF (mW/m$^2$)'
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 9))
        ax4, ax1, ax2, ax3= (
            ax.flatten()[0],
            ax.flatten()[1],
            ax.flatten()[2],
            ax.flatten()[3]
        )
    elif what == 'CDR':
        ylabel = 'CDR (MtCO$_2$/yr)'
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))
        ax1, ax2, ax3 = (
            ax.flatten()[0],
            ax.flatten()[1],
            ax.flatten()[2]
        )
    if type == 'bar':
        plot_species_subplots(data=df1, start_date=start_date, txt=scenario1, low_lim=low_lim, up_lim=up_lim,
                              ylabel=ylabel, ax=ax1)
    else:
        plt_species_subplots(data=df1, start_date=start_date, txt=scenario1, low_lim=low_lim, up_lim=up_lim,
                             ylabel=ylabel, ax=ax1, palette = palette)
    if type == 'bar':
        plot_species_subplots(data=df2, start_date=start_date, txt=scenario2, low_lim=low_lim, up_lim=up_lim,
                              ylabel=ylabel, ax=ax2)
    else:
        plt_species_subplots(data=df2, start_date=start_date, txt=scenario2, low_lim=low_lim, up_lim=up_lim,
                             ylabel=ylabel, ax=ax2, palette = palette)
    if type == 'bar':
        plot_species_subplots(data=df3, start_date=start_date, txt=scenario3, low_lim=low_lim, up_lim=up_lim,
                              ylabel=ylabel, ax=ax3)
    else:
        plt_species_subplots(data=df3, start_date=start_date, txt=scenario3, low_lim=low_lim, up_lim=up_lim,
                             ylabel=ylabel, ax=ax3, palette = palette)
    if what == 'ERF':
        if type == 'bar':
            plot_species_subplots(data=df_ref, start_date=start_date, txt=ref_scenario, low_lim=low_lim, up_lim=up_lim,
                                  ylabel=ylabel, ax=ax4)
        else:
            plt_species_subplots(data=df_ref, start_date=start_date, txt=ref_scenario, low_lim=low_lim, up_lim=up_lim,
                                 ylabel=ylabel, ax=ax4)
    handles1, labels1 = ax1.get_legend_handles_labels()

    fig.legend(handles = handles1, labels=  labels1, bbox_to_anchor=(0.5, -0.1), loc="lower center",
               ncol=4, frameon=False)
    if what == 'ERF':
        fig.tight_layout(pad=0.4)
        plt.rcParams["axes.labelsize"] = 11
    ax1.text(0.05, 0.95, 'SSP1-2.6', transform=ax1.transAxes, color=col_scenario1, fontweight='bold', va="top",
             ha="left")
    ax2.text(0.05, 0.95, 'SSP2-4.5', transform=ax2.transAxes, color=col_scenario2, fontweight='bold', va="top",
             ha="left")
    ax3.text(0.05, 0.95, 'SSP5-8.5', transform=ax3.transAxes, color=col_scenario3, fontweight='bold', va="top",
             ha="left")
    ax4.text(0.05, 0.95, 'SSP1-1.9', transform=ax4.transAxes, color=col_scenarioref, fontweight='bold', va="top",
             ha="left")

    fig.tight_layout(pad=0.7)
    fig.savefig("Figures/"+ what +"_scenarios_"+CDR_type+".png", dpi=850, bbox_inches="tight")


def plot_ERF_alltechs_scenarios(df1, df2, df3, df1_tech1, df2_tech1, df3_tech1, df1_tech2, df2_tech2, df3_tech2, scenario1, scenario2, scenario3, ref_scenario,
                            df_ref = None, low_lim1 = None, up_lim1 = None, low_lim2 = None, up_lim2 = None,
                                low_lim3 = None, up_lim3 = None, start_date = 2018, type ='plot', palette= 'icefire', what = 'new'):


    ylabel = 'ERF (mW/m$^2$)'
    if what == 'new':
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(7.2, 4.5))
        fig.tight_layout(pad=0.7)
        ax1, ax2, ax3,  ax7, ax8, ax9 = (
            ax.flatten()[0],
            ax.flatten()[1],
            ax.flatten()[2],
            ax.flatten()[3],
            ax.flatten()[4],
            ax.flatten()[5],
        )
    else:
        fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(10, 9))
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = (
            ax.flatten()[0],
            ax.flatten()[1],
            ax.flatten()[2],
            ax.flatten()[3],
            ax.flatten()[4],
            ax.flatten()[5],
            ax.flatten()[6],
            ax.flatten()[7],
            ax.flatten()[8],
        )

    plt_species_subplots(data=df1, start_date=start_date, txt=scenario1, low_lim=low_lim1, up_lim=up_lim1,
                         ylabel=ylabel, ax=ax1, palette = palette)
    plt_species_subplots(data=df1_tech1, start_date=start_date, txt=scenario1, low_lim=low_lim1, up_lim=up_lim1,
                         ylabel=None, ax=ax2, palette = palette)
    plt_species_subplots(data=df1_tech2, start_date=start_date, txt=scenario1, low_lim=low_lim1, up_lim=up_lim1,
                         ylabel=None, ax=ax3, palette = palette)
    if what != 'new':
        plt_species_subplots(data=df2, start_date=start_date, txt=scenario2, low_lim=low_lim2, up_lim=up_lim2,
                             ylabel=ylabel, ax=ax4, palette = palette)
        plt_species_subplots(data=df2_tech1, start_date=start_date, txt=scenario2, low_lim=low_lim2, up_lim=up_lim2,
                             ylabel=ylabel, ax=ax5, palette = palette)
        plt_species_subplots(data=df2_tech2, start_date=start_date, txt=scenario2, low_lim=low_lim2, up_lim=up_lim2,
                             ylabel=ylabel, ax=ax6, palette = palette)
    plt_species_subplots(data=df3, start_date=start_date, txt=scenario3, low_lim=low_lim3, up_lim=up_lim3,
                         ylabel=ylabel, ax=ax7, palette = palette)
    plt_species_subplots(data=df3_tech1, start_date=start_date, txt=scenario3, low_lim=low_lim3, up_lim=up_lim3,
                         ylabel=None, ax=ax8, palette = palette)
    plt_species_subplots(data=df3_tech2, start_date=start_date, txt=scenario3, low_lim=low_lim3, up_lim=up_lim3,
                         ylabel=None, ax=ax9, palette = palette)

    handles1, labels1 = ax1.get_legend_handles_labels()

    if what == 'new':
        fig.legend(handles=handles1, labels=labels1, bbox_to_anchor=(0.5, -0.25), loc="lower center",
                   ncol=3, frameon=False)
    else:
        fig.legend(handles = handles1, labels=  labels1, bbox_to_anchor=(0.5, -0.1), loc="lower center",
                   ncol=3, frameon=False)

    ax1.text(0.5, 1.1, 'Fossil jet fuels',  transform=ax1.transAxes, va="top", ha="center" )
    ax2.text(0.5, 1.1, TECH_1, transform=ax2.transAxes, va="top", ha="center")
    ax3.text(0.5, 1.1, TECH_2, transform=ax3.transAxes, va="top", ha="center")
    ax1.text(0.05, 0.95, 'SSP1-2.6', transform=ax1.transAxes, color=col_scenario1, fontweight='bold', va="top",
             ha="left")
    ax2.text(0.05, 0.95, 'SSP1-2.6', transform=ax2.transAxes, color=col_scenario1, fontweight='bold', va="top",
             ha="left")
    ax3.text(0.05, 0.95, 'SSP1-2.6', transform=ax3.transAxes, color=col_scenario1, fontweight='bold', va="top",
             ha="left")
    if what != 'new':
        ax4.text(0.05, 0.95, 'SSP2-4.5', transform=ax4.transAxes, color=col_scenario2, fontweight='bold', va="top",
                 ha="left")
        ax5.text(0.05, 0.95, 'SSP2-4.5', transform=ax5.transAxes, color=col_scenario2, fontweight='bold', va="top",
                 ha="left")
        ax6.text(0.05, 0.95, 'SSP2-4.5', transform=ax6.transAxes, color=col_scenario2, fontweight='bold', va="top",
                 ha="left")
    ax7.text(0.05, 0.95, 'SSP5-8.5', transform=ax7.transAxes, color=col_scenario3, fontweight='bold', va="top",
             ha="left")
    ax8.text(0.05, 0.95, 'SSP5-8.5', transform=ax8.transAxes, color=col_scenario3, fontweight='bold', va="top",
             ha="left")
    ax9.text(0.05, 0.95, 'SSP5-8.5', transform=ax9.transAxes, color=col_scenario3, fontweight='bold', va="top",
             ha="left")

    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 8.5
    plt.rcParams["xtick.labelsize"] = 8.5
    fig.tight_layout(pad=0.3)
    fig.savefig("Figures/Alltech_ERF_scenarios.pdf", dpi=600, bbox_inches="tight")

def plot_CDR_rates_multi(mean_df_original, cum_df_original, positive_df_original, negative_df_original,
                             scenario1='SSP1_26', scenario3='SSP5_85', size_p = 7,
                             tech1=TECH_1, tech2=TECH_2,  scenario = 'A', plot_type = 'bar',  plots = 'new'):
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(7.2, 7.1), tight_layout=True)
    ax1, ax2, ax3, ax4 = (
        ax.flatten()[0],
        ax.flatten()[1],
        ax.flatten()[2],
        ax.flatten()[3]
    )
    ylabel_mean = 'mean CO$_2$ removal (GtCO$_2$/yr)'
    txt_mean = 'mean CDR'
    x = 'Climate neutrality'
    hue = 'Technology'
    ylabel_cum = 'cumulative CO$_2$ removal (GtCO$_2$)'
    txt_cum = 'cumulative CDR'

    x_ticks_names = np.array([baseline_1, baseline_2, baseline_3])
    x_ticks = np.array([0, 1, 2])


    colors1 = [[sns.dark_palette(col_scenario1_shade, reverse=True)[0],
                sns.dark_palette(const_shadier, reverse=True)[0]],
               [sns.dark_palette(col_scenario1_shade, reverse=True)[1],
                sns.dark_palette(const_shadier, reverse=True)[1]],
               [sns.dark_palette(col_scenario1_shade, reverse=True)[2],
                sns.dark_palette(const_shadier, reverse=True)[2]]]
    colors3 = [[sns.dark_palette(col_scenario3_shade, reverse=True)[0],
                sns.dark_palette(const_shadier, reverse=True)[0]],
               [sns.dark_palette(col_scenario3_shade, reverse=True)[1],
                sns.dark_palette(const_shadier, reverse=True)[1]],
               [sns.dark_palette(col_scenario3_shade, reverse=True)[2],
                sns.dark_palette(const_shadier, reverse=True)[2]]]
    categories = ['Fossil jet fuels (+/-)',
                  TECH_1 + ' (+/-)',
                  TECH_2 + ' (+/-)']

    mean_df = mean_df_original.copy()
    cum_df = cum_df_original.copy()
    positive_df = positive_df_original.copy()
    negative_df = negative_df_original.copy()


    mean_df['summary_Tot_CDR'] = mean_df['summary_Tot_CDR'].copy() / 1000
    mean_df['summary_CO2_CDR'] = mean_df['summary_CO2_CDR'].copy() / 1000
    mean_df['summary_Tot_CDR_std'] = mean_df['summary_Tot_CDR_std'].copy() / 1000
    cum_df['summary_Tot_CDR'] = cum_df['summary_Tot_CDR'].copy() / 1000
    cum_df['summary_CO2_CDR'] = cum_df['summary_CO2_CDR'].copy() / 1000
    cum_df['summary_Tot_CDR_std'] = cum_df['summary_Tot_CDR_std'].copy() / 1000
    positive_df['summary_Tot_CDR'] = positive_df['summary_Tot_CDR'].copy() / 1000
    positive_df['summary_CO2_CDR'] = positive_df['summary_CO2_CDR'].copy() / 1000
    positive_df['summary_Tot_CDR_std'] = positive_df['summary_Tot_CDR_std'].copy() / 1000
    negative_df['summary_Tot_CDR'] = negative_df['summary_Tot_CDR'].copy() / 1000
    negative_df['summary_CO2_CDR'] = negative_df['summary_CO2_CDR'].copy() / 1000
    negative_df['summary_Tot_CDR_std'] = negative_df['summary_Tot_CDR_std'].copy() / 1000

    add = 0.27
    x_std = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]) + np.array([-add, -add, -add, 0, 0, 0, add, add, add])

    ax1.hlines(0, -0.4, 2.4, linestyles='dashed', colors=const_shade)
    ax2.hlines(0, -0.4, 2.4, linestyles='dashed', colors=const_shade)
    sns.swarmplot(ax=ax1, x=x, y='summary_Tot_CDR', hue=hue, data=mean_df[mean_df["Scenario"] == scenario1],
                  # color = col_scenario1)
                  palette=sns.dark_palette(col_scenario1_shade, reverse=True),
                  dodge=True, size=size_p)
    sns.swarmplot(ax=ax2, x=x, y='summary_Tot_CDR', hue=hue,
                  data=mean_df[mean_df["Scenario"] == scenario3],
                  dodge=True, size=size_p,
                  palette=sns.dark_palette(col_scenario3_shade, reverse=True))
    ax1.errorbar(x=x_std, y=mean_df[mean_df["Scenario"] == scenario1]['summary_Tot_CDR'],
                 yerr=mean_df[mean_df["Scenario"] == scenario1]['summary_Tot_CDR_std'],
                 c='black', fmt='none',
                 capsize=2)
    ax2.errorbar(x=x_std, y=mean_df[mean_df["Scenario"] == scenario3]['summary_Tot_CDR'],
                 yerr=mean_df[mean_df["Scenario"] == scenario3]['summary_Tot_CDR_std'],
                 c='black', fmt='none',
                 capsize=2)


    legend_elements_1 = [
        Line2D([0], [0], marker='o', color='w', label='Fossil jet fuels',
               markerfacecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[0], markersize=size_p),
        Line2D([0], [0], marker='o', color='w', label=TECH_1,
               markerfacecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[1], markersize=size_p),
        Line2D([0], [0], marker='o', color='w', label=TECH_2,
               markerfacecolor=sns.dark_palette(col_scenario1_shade, reverse=True)[2], markersize=size_p),
    ]
    legend_elements_3 = [
        Line2D([0], [0], marker='o', color='w', label='Fossil jet fuels',
               markerfacecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[0], markersize=size_p),
        Line2D([0], [0], marker='o', color='w', label=TECH_1,
               markerfacecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[1], markersize=size_p),
        Line2D([0], [0], marker='o', color='w', label=TECH_2,
               markerfacecolor=sns.dark_palette(col_scenario3_shade, reverse=True)[2], markersize=size_p),
    ]

    ax1.legend(handles=legend_elements_1, bbox_to_anchor=(0.5, -0.52), loc="lower center", ncol=1, frameon=False)
    ax2.legend(handles=legend_elements_3, bbox_to_anchor=(0.5, -0.52), loc="lower center", ncol=1, frameon=False)

    sns.barplot(ax=ax3, x=x, y='summary_Tot_CDR', hue=hue, data=positive_df[positive_df["Scenario"] == scenario1],
                # color = col_scenario1)
                palette=sns.dark_palette(col_scenario1_shade, reverse=True))
    sns.barplot(ax=ax3, x=x, y='summary_Tot_CDR', hue=hue,
                data=negative_df[negative_df["Scenario"] == scenario1],
                # color = col_scenario1)
                palette=sns.dark_palette(const_shadier, reverse=True))
    ax3.errorbar(x=x_std, y=cum_df[cum_df["Scenario"] == scenario1]['summary_Tot_CDR'],
                 yerr=cum_df[cum_df["Scenario"] == scenario1]['summary_Tot_CDR_std'],
                 fmt='o', c='black', label='cumulative CDR',
                 capsize=2)
    sns.barplot(ax=ax4, x=x, y='summary_Tot_CDR', hue=hue,
                data=positive_df[positive_df["Scenario"] == scenario3],
                # color = col_scenario2)
                palette=sns.dark_palette(col_scenario3_shade, reverse=True))
    sns.barplot(ax=ax4, x=x, y='summary_Tot_CDR', hue=hue,
                data=negative_df[negative_df["Scenario"] == scenario3],
                # color = col_scenario2)
                palette=sns.dark_palette(const_shadier, reverse=True))
    ax4.errorbar(x=x_std, y=cum_df[cum_df["Scenario"] == scenario3]['summary_Tot_CDR'],
                 yerr=cum_df[cum_df["Scenario"] == scenario3]['summary_Tot_CDR_std'],
                 fmt='o', c='black', label='cumulative CDR',
                 capsize=2)

    legend_dict3 = dict(zip(categories, colors1))
    legend_dict4 = dict(zip(categories, colors3))
    legend_elements_3 = []
    legend_elements_4 = []
    for cat, col in legend_dict3.items():
        legend_elements_3.append([mpatches.Patch(facecolor=c, label=cat) for c in col])
    for cat, col in legend_dict4.items():
        legend_elements_4.append([mpatches.Patch(facecolor=c, label=cat) for c in col])
    legend_element_4 = [Line2D([0], [0], marker='o', ls='none', markeredgecolor='none',
                               label='Net CDR',
                               markerfacecolor='k',
                               markersize=size_p)]
    legend3 = ax3.legend(handles=legend_element_4, loc="upper right", frameon=False)
    legend4 = ax4.legend(handles=legend_element_4, loc="upper right", frameon=False)

    ax3.legend(handles=legend_elements_3, labels=categories, bbox_to_anchor=(0.5, -0.5), loc="lower center",
               borderaxespad=0,
               ncol=1, frameon=False, handler_map={list: HandlerTuple(None)})
    ax4.legend(handles=legend_elements_4, labels=categories, bbox_to_anchor=(0.5, -0.5), loc="lower center",
               borderaxespad=0,
               ncol=1, frameon=False, handler_map={list: HandlerTuple(None)})
    ax3.add_artist(legend3)
    ax4.add_artist(legend4)

    ax1.set_ylabel(ylabel=ylabel_mean)
    ax2.set(ylabel=None)
    ax3.set_ylabel(ylabel=ylabel_cum)
    ax4.set(ylabel=None)
    ax1.set(xlabel=None)
    ax3.set(xlabel=None)
    ax2.set(xlabel=None)
    ax4.set(xlabel=None)
    ax1.set_xticklabels(x_ticks_names)
    ax3.set_xticklabels(x_ticks_names)
    ax2.set_xticklabels(x_ticks_names)
    ax4.set_xticklabels(x_ticks_names)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    box = ax4.get_position()
    ax4.set_position([box.x0, box.y0, box.width * 0.95, box.height])

    ax1.text(0.05, 0.05, 'SSP1-2.6', transform=ax1.transAxes, color=col_scenario1, fontweight='bold', va="bottom",
             ha="left")
    ax2.text(0.05, 0.05, 'SSP5-8.5', transform=ax2.transAxes, color=col_scenario3, fontweight='bold', va="bottom",
             ha="left")
    ax3.text(0.05, 0.05, 'SSP1-2.6', transform=ax3.transAxes, color=col_scenario1, fontweight='bold', va="bottom",
             ha="left")
    ax4.text(0.05, 0.05, 'SSP5-8.5', transform=ax4.transAxes, color=col_scenario3, fontweight='bold', va="bottom",
             ha="left")

    # fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')

    plt.rcParams['font.size'] = 11

    #fig.tight_layout(pad = 0.3)
    fig.savefig("Figures/compare_CDR_alltechs_" + scenario + "_" + plot_type + "_" + plots + ".pdf",
                dpi=600, bbox_inches="tight")
