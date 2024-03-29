import matplotlib.pyplot as plt
import seaborn as sns


def pretty_plot_settings():
    sns.set(rc={
                'axes.axisbelow': False,
                'axes.edgecolor': 'lightgrey',
                'axes.facecolor': 'None',
                'axes.grid': False,
                'axes.labelcolor': 'dimgrey',
                'axes.spines.right': False,
                'axes.spines.top': False,
                'figure.facecolor': 'white',
                'lines.solid_capstyle': 'round',
                'patch.edgecolor': 'w',
                'patch.force_edgecolor': True,
                'text.color': 'dimgrey',
                'xtick.bottom': False,
                'xtick.color': 'dimgrey',
                'xtick.direction': 'out',
                'xtick.top': False,
                'ytick.color': 'dimgrey',
                'ytick.direction': 'out',
                'ytick.left': False,
                'ytick.right': False,
                'figure.autolayout': True})s

    sns.set_context("notebook", rc={"font.size": 16,
                                    "axes.titlesize": 20,
                                    "axes.labelsize": 16})

    # define colors
    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
                  CB91_Purple, CB91_Violet]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
