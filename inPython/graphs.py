# -*- coding: utf-8 -*-
#______________________________________________________________________________
#______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 21/03/2019
#______________________________________________________________________________
#______________________________________________________________________________
'''
The functions given on this package allow the user to open data from the 
COMSOL export data.

'''
# ------------------------
# Importing Modules
# ------------------------ 
# Data Managment
import numpy as np
# Graphs
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.colors as colors
import seaborn as sns

# Personal libraries
import utilities as utl

# ------------------------
# Functions
# ------------------------ 
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def Graph2D(DataT,times,Name,PathImg,S=None,vlimits=None,
        verbose=True,TimeF=True,cmap='jet'):
    '''
    DESCRIPTION:
        This function generates the 2D plots
    ____________________________________________________________
    INPUT:
        :param DataT: A dict, Variables to be plotted.
        :param times: A list, list with the times to plot.
        :param Name: A str, Name of the image with extension.
        :param PathImg: A str, Directory where the image would
                           be stored.
        :param S: A list, Variables that are streamlines.
        :param vlimits: A list, limits of the variable.
        :param verbose: A bool, flag to print everything.
        :param TimeF: A bool, flag to include time in the title.
    ____________________________________________________________
    OUTPUT:
        This function return some images.
    '''
    # Creating Folder
    utl.CrFolder(PathImg)
    # Parameters
    den = 35 # For streamlines
    Vars = list(DataT)

    # Creating Figure
    plt.rcParams.update({'font.size': 16,'font.family': 'sans-serif'\
        ,'font.sans-serif': 'Arial'\
        ,'xtick.labelsize': 16,'xtick.major.size': 6,'xtick.minor.size': 4\
        ,'xtick.major.width': 1,'xtick.minor.width': 1\
        ,'ytick.labelsize': 16,'ytick.major.size': 12,'ytick.minor.size': 4\
        ,'ytick.major.width': 1,'ytick.minor.width': 1\
        ,'axes.linewidth':1\
        ,'grid.alpha':0.1,'grid.linestyle':'-'})
    plt.tick_params(axis='x',which='both',bottom=True,top=False,\
        labelbottom=True,direction='out')
    plt.tick_params(axis='y',which='both',left=True,right=False,\
        labelleft=True)
    plt.tick_params(axis='y',which='major',direction='inout') 

    if times[0] == 'all':
        times = DataT[Vars[0]]['time']

    if len(Vars) > 1:
        ValueError('For Subplots use Graph2DSubPlots')

    # Set Limits
    vmaxT = dict()
    vminT = dict()
    v = dict()
    bounds = dict()
    if not(vlimits is None):
        for iV,V in enumerate(Vars):
            Data = DataT[V]
            D = np.zeros((len(times),Data['V'].shape[1],Data['V'].shape[2]))
            # Setting maximum and minimum
            vmaxT[V] = np.nanmax(vlimits[1])
            vminT[V] = np.nanmin(vlimits[0])
            v[V] = np.linspace(vminT[V], vmaxT[V], 10, endpoint=True)
            bounds[V] = np.linspace(vminT[V],vmaxT[V],20)
    else:
        for iV,V in enumerate(Vars):
            Data = DataT[V]
            D = np.zeros((len(times),Data['V'].shape[1],Data['V'].shape[2]))
            for iT,T in enumerate(times):
                TT = np.where(Data['time'] == T)[0][0]
                D[iT] = Data['V'][TT]
            # Setting maximum and minimum
            vmaxT[V] = np.nanmax(D)
            vminT[V] = np.nanmin(D)
            v[V] = np.linspace(vminT[V], vmaxT[V], 10, endpoint=True)
            bounds[V] = np.linspace(vminT[V],vmaxT[V],20)

    if not(S is None):
        E = 3
    else:
        E = 1
    for iT,T in enumerate(times):
        if verbose:
            print(' Plotting time {}'.format(T))
        # Figure
        Inch = 0.393701
        fH = 25*Inch
        fV = fH*(1/2)
        F,axs = plt.subplots(len(Vars)-E+1,1,figsize=(fH,fV))
        if E-1 == 0:
            EN = Vars
        else:
            EN = Vars[:-E+1]
        for iV,V in enumerate(EN):
            Data = DataT[V]
            TT = np.where(Data['time'] == T)[0][0]
            # Plot
            plt.pcolormesh(Data['x'],Data['y'],Data['V'][TT,:,:],
                    vmax=vmaxT[V],vmin=vminT[V],cmap=cmap)
            cbar = plt.colorbar(boundaries=bounds[V],ticks=v[V])
            units = Data['VarUnits'].split('^')
            if len(units) == 2:
                units = r'{}$^{}$'.format(units[0],units[1])
            else:
                units = units[0]
            if units == '':
                cbar.set_label('{}'.format(Data['Description']))
            else:
                cbar.set_label('{} [{}]'.format(Data['Description'],
                units))

            if not(S is None):
                if len(S) == 2:
                    # Slope = 0.025
                    # Amp = 100
                    # n = 5
                    # L = 40000
                    # Lambda = L/(n-0.5)
                    xS = np.linspace(np.min(DataT[S[0]]['x']),np.max(DataT[S[0]]['x']),200)
                    # yS = Slope*xS+Amp+Amp*np.sin((2*np.pi*xS/Lambda)-(np.pi/2))
                    # yS = Slope*xS+Amp*np.sin((2*np.pi*xS/Lambda)-(np.pi/2))
                    yS = np.zeros(xS.shape)
                    seed_points = np.array([xS,yS])
                    # xS = np.linspace(np.min(DataT[S[0]]['x']),np.max(DataT[S[0]]['x']),200)
                    # yS = np.ones(xS.shape)*-1*2000
                    # seed_points = np.hstack((seed_points,np.array([xS,yS])))
                    speed = np.sqrt(DataT[S[0]]['V'][TT,:,:]**2+DataT[S[1]]['V'][TT,:,:]**2)*1000
                    lws = speed/np.nanmax(speed)*10
                    plt.streamplot(DataT[S[0]]['x'],DataT[S[0]]['y'],
                            DataT[S[0]]['V'][TT,:,:],DataT[S[1]]['V'][TT,:,:],
                            color='k',density=den,start_points=seed_points.T)
                            # linewidth=lws)
                    # plt.quiver(DataT[S[0]]['x'],DataT[S[0]]['y'],
                    #     DataT[S[0]]['V'][TT,:,:],DataT[S[1]]['V'][TT,:,:],
                    #     color='k')
                    if iT == 0:
                        Title = Data ['Description'] + ', Streamlines: {}'.format(DataT[S[0]]['Description'])
                else:
                    print(' Must have 2 components')
            else:
                Title = Data['Description']
            plt.xlim([np.min(Data['x']),np.max(Data['x'])])
            plt.ylim([np.min(Data['y']),np.max(Data['y'])])
            if TimeF:
                plt.title(r'{} $t = {}$'.format(Title,T),fontsize=16)
            else:
                plt.title(r'{}'.format(Title),fontsize=16)
            plt.xlabel('Horizontal Distance [m]',fontsize=16)
            plt.ylabel('Elevation [m]',fontsize=16)
            plt.grid()
            # Hide the right and top spines
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # Scale image
        plt.tight_layout()
        # Include minor axis
        axes = plt.gca()
        yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # Minor tick value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        xTL = axes.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # Minor tick value
        minorLocatorx = MultipleLocator(MxL)
        plt.gca().xaxis.set_minor_locator(minorLocatorx)

        # Saving Image
        if len(times) == 1:
            plt.savefig('{}{}.{}'.format(PathImg,
                Name.split('.')[0],Name.split('.')[-1]),
                format=Name.split('.')[-1],dpi=200)
        else:
            plt.savefig('{}{}_{:04d}.{}'.format(PathImg,
                Name.split('.')[0],iT,Name.split('.')[-1]),
                format=Name.split('.')[-1],dpi=200)
        plt.close('all')
    return 

def Graph2DSubPlots(Projects,Vars,Name,PathImg,S=None,SB=None,
        verbose=True,TimeF=True,cmap='jet',Titles=None,vlimits=None):
    '''
    DESCRIPTION:
        This function generates the 2D plots
    ____________________________________________________________
    INPUT:
        :param Projects: A list, List with projects to plot.
                         This would be the columns.
        :param Vars: A list, List with variables to plot
                     This would be the rows.
        :param times: A list, list with the times to plot.
        :param Name: A str, Name of the image with extension.
        :param PathImg: A str, Directory where the image would
                           be stored.
        :param S: A list, Variables that are streamlines.
        :param SB: A list, Flag to include the streamlines.
        :param verbose: A bool, flag to print everything.
        :param TimeF: A bool, flag to include time in the title.
    ____________________________________________________________
    OUTPUT:
        This function return some images.
    '''
    # Creating Folder
    utl.CrFolder(PathImg)
    # Parameters
    den = 35 # For streamlines

    # Creating Figure
    plt.rcParams.update({'font.size': 14,'font.family': 'sans-serif'\
        ,'font.sans-serif': 'Arial'\
        ,'xtick.labelsize': 14,'xtick.major.size': 6,'xtick.minor.size': 4\
        ,'xtick.major.width': 1,'xtick.minor.width': 1\
        ,'ytick.labelsize': 14,'ytick.major.size': 12,'ytick.minor.size': 4\
        ,'ytick.major.width': 1,'ytick.minor.width': 1\
        ,'axes.linewidth':1\
        ,'grid.alpha':0.1,'grid.linestyle':'-'})

    vmaxT = dict()
    vminT = dict()
    xlim = []
    ylim = []
    v = dict()
    bounds = dict()
    if vlimits is None:
        vlimits = dict()
    for iV,V in enumerate(Vars):
        # Columns
        for iC,DataT in enumerate(Projects):
            Data = DataT[V]
            TT = 0
            if iC == 0:
                D = np.zeros((len(Projects),Data['V'].shape[1],
                    Data['V'].shape[2]))
            D[iC] = Data['V'][TT]
            xlim.append(np.max(Data['x']))
            xlim.append(np.min(Data['x']))
            ylim.append(np.max(Data['y']))
            ylim.append(np.min(Data['y']))
        # Setting maximum and minimum

        try:
            vmaxT[V] = np.nanmax(vlimits[V])
            vminT[V] = np.nanmin(vlimits[V])
        except KeyError:
            vmaxT[V] = np.nanmax(D)
            vminT[V] = np.nanmin(D)

        v[V] = np.linspace(vminT[V], vmaxT[V], 5, endpoint=True)
        bounds[V] = np.linspace(vminT[V],vmaxT[V],20)

    xlims = [np.min(xlim),np.max(xlim)]
    ylims = [np.min(ylim),np.max(ylim)]

    if isinstance(cmap,str):
        cmaps = cmap
        cmap = dict()
        for iV,V in enumerate(Vars):
            cmap[V] = cmaps
    else:
        for iV,V in enumerate(Vars):
            try:
                cmap[V] = cmap[V]
            except KeyError:
                cmap[V] = 'jet'
        
    Plus = -15+((len(Vars)-1)*10)
    # Figure
    Inch = 0.393701
    fH = (25+(5*(len(Projects)-1)))*Inch
    fV = (20+Plus)* Inch
    F,axs = plt.subplots(len(Vars),len(Projects),figsize=(fH,fV))
    axs = axs.ravel()
    plot = 0
    # Rows
    for iV,V in enumerate(Vars):
        plt.rcParams.update({
            'xtick.labelsize': 12,'xtick.major.size': 6,
            'ytick.labelsize': 12,'ytick.major.size': 6,
            })
        # Columns
        plots = []
        for iC,DataT in enumerate(Projects):
            Data = DataT[V]
            TT = 0
            if V[:3] == 'Dif':
                Data = DataT[f'{V}']
                TT = 0
                b = axs[plot].pcolormesh(Data['x'],Data['y'],Data['V'][TT,:,:],
                        vmax=vmaxT[f'{V}'],vmin=vminT[f'{V}'],
                        cmap=cmap[f'{V}'],
                        norm=MidpointNormalize(midpoint=0,vmin=vminT[f'{V}'],
                            vmax=vmaxT[f'{V}']))
            else:
                b = axs[plot].pcolormesh(Data['x'],Data['y'],Data['V'][TT,:,:],
                        vmax=vmaxT[V],vmin=vminT[V],cmap=cmap[V])

            plots.append(axs[plot].get_position().get_points().flatten())

            if iC == len(Projects) - 1:
                plt.rcParams.update({
                    'xtick.labelsize': 12,'xtick.major.size': 2,
                    'ytick.labelsize': 12,'ytick.major.size': 2,
                    })
                if len(Projects) == 1:
                    cbar = plt.colorbar(b,boundaries=bounds[V],ticks=v[V],
                            ax=axs[plot],orientation='vertical')
                else:
                    ax_cbar1 = F.add_axes([plots[-1][2]+0.01, plots[-1][1], 
                        0.015, plots[-1][3]-plots[-1][1]])
                    cbar = plt.colorbar(b,boundaries=bounds[V],ticks=v[V],
                            cax=ax_cbar1,orientation='vertical')
                    # cbar.ax.xaxis.set_ticks_position('top')

                if Data['VarUnits'] == '':
                    # F.text(plots[0][2]+0.001,plots[0][3]+0.01,
                    #         '[-]',fontsize=12)
                    cbar.set_label('[-]')
                else:
                    # F.text(plots[0][2]+0.001,plots[0][3]+0.01,
                    #         '[{}]'.format(Data['VarUnits']),fontsize=12)
                    cbar.set_label('[{}]'.format(Data['VarUnits'],))

            if not(S is None):
                if len(S) == 2:
                    if SB[iV]:
                        speed = np.sqrt(DataT[S[0]]['V'][TT,:,:]**2+DataT[S[1]]['V'][TT,:,:]**2)*1000
                        lws = speed/np.nanmax(speed)*4
                        axs[plot].streamplot(DataT[S[0]]['x'],DataT[S[0]]['y'],
                                DataT[S[0]]['V'][TT,:,:],DataT[S[1]]['V'][TT,:,:],color='k',
                                density=den,linewidth=lws)
                        # axs[iV].quiver(DataT[S[0]]['x'],DataT[S[0]]['y'],
                        #     DataT[S[0]]['V'][TT,:,:],DataT[S[1]]['V'][TT,:,:],
                        #     color='k')
                        Title = Data ['Description'] + ', Streamlines: {}'.format(DataT[S[0]]['Description'])
                    else:
                        Title = Data['Description']
                else:
                    print(' Must have 2 components')
            else:
                Title = Data['Description']
            axs[plot].set_xlim(xlims)
            axs[plot].set_ylim(ylims)

            # Title
            if iC == 0:
                F.text(0.02,plots[-1][3]-0.11,f'{Title}',
                        fontsize=14,rotation=90,weight = 'bold',
                        horizontalalignment='center',
                        verticalalignment='center')

            if not(Titles is None):
                if iV == 0:
                    if len(Projects) >= 5:
                        fontsize = 12
                    else:
                        fontsize = 14
                    axs[plot].set_title(Titles[iC],fontsize=fontsize)

            if iV == len(Vars) - 1:
                axs[plot].set_xlabel('Horizontal Distance [km]',fontsize=14)
            if iC == 0 :
                axs[plot].set_ylabel('Elevation [km]',fontsize=14)
            # Hide the right and top spines
            axs[plot].spines['right'].set_visible(False)
            axs[plot].spines['top'].set_visible(False)
            # if iC > 0:
            #     yTL = axs[plot].yaxis.get_ticklocs() # List of Ticks in y
            #     axs[plot].set_yticklabels(['']*len(yTL))
            # if iV < len(Vars)-1:
            #     xTL = axs[plot].xaxis.get_ticklocs() # List of Ticks in y
            #     axs[plot].set_xticklabels(['']*len(xTL))
            plot += 1
    # plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    plt.subplots_adjust(hspace=0.20)
    for i in range(plot):
        axes = axs[i]
        yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # Minor tick value
        minorLocatory = MultipleLocator(MyL)
        axes.yaxis.set_minor_locator(minorLocatory)
        xTL = axes.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # Minor tick value
        minorLocatorx = MultipleLocator(MxL)
        axes.xaxis.set_minor_locator(minorLocatorx)
    plt.savefig('{}{}'.format(PathImg,Name),
        format=Name.split('.')[-1],dpi=200,bbox_inch='tight')
    plt.close('all')
    return 

def Graph2DSubPlotsDiff(Projects,Vars,Name,PathImg,S=None,SB=None,
        verbose=True,TimeF=True,cmap='jet',Titles=None):
    '''
    DESCRIPTION:
        This function generates the 2D plots
    ____________________________________________________________
    INPUT:
        :param Projects: A list, List with projects to plot.
                         This would be the columns.
        :param Vars: A list, List with variables to plot
                     This would be the rows.
        :param times: A list, list with the times to plot.
        :param Name: A str, Name of the image with extension.
        :param PathImg: A str, Directory where the image would
                           be stored.
        :param S: A list, Variables that are streamlines.
        :param SB: A list, Flag to include the streamlines.
        :param verbose: A bool, flag to print everything.
        :param TimeF: A bool, flag to include time in the title.
    ____________________________________________________________
    OUTPUT:
        This function return some images.
    '''
    # Creating Folder
    utl.CrFolder(PathImg)
    # Parameters
    den = 35 # For streamlines

    # Creating Figure
    plt.rcParams.update({'font.size': 14,'font.family': 'sans-serif'\
        ,'font.sans-serif': 'Arial'\
        ,'xtick.labelsize': 14,'xtick.major.size': 6,'xtick.minor.size': 4\
        ,'xtick.major.width': 1,'xtick.minor.width': 1\
        ,'ytick.labelsize': 14,'ytick.major.size': 12,'ytick.minor.size': 4\
        ,'ytick.major.width': 1,'ytick.minor.width': 1\
        ,'axes.linewidth':1\
        ,'grid.alpha':0.1,'grid.linestyle':'-'})

    vmaxT = dict()
    vminT = dict()
    xlim = []
    ylim = []
    v = dict()
    bounds = dict()
    Vars2 = ['','2']
    for iV,V in enumerate(Vars):
        # Columns
        for iC,DataT in enumerate(Projects[:-1]):
            Data = DataT[f'{V}{Vars2[iC]}']
            TT = 0
            if iC == 0:
                D = np.zeros((len(Projects)-1,Data['V'].shape[1],
                    Data['V'].shape[2]))
            D[iC] = Data['V'][TT]
        # Setting maximum and minimum
        vmaxT[V] = np.nanmax(D)
        vminT[V] = np.nanmin(D)
        vmaxT[f'Dif{V}'] = np.nanmax(DataT[f'Dif{V}']['V'][TT])
        vminT[f'Dif{V}'] = np.nanmin(DataT[f'Dif{V}']['V'][TT])
        v[V] = np.linspace(vminT[V], vmaxT[V], 5, endpoint=True)
        bounds[V] = np.linspace(vminT[V],vmaxT[V],20)
        v[f'Dif{V}'] = np.linspace(np.nanmin(DataT[f'Dif{V}']['V'][TT]),
                np.nanmax(DataT[f'Dif{V}']['V'][TT]), 5, endpoint=True)
        bounds[f'Dif{V}'] = np.linspace(np.nanmin(DataT[f'Dif{V}']['V'][TT]),
                np.nanmax(DataT[f'Dif{V}']['V'][TT]), 20, endpoint=True)
        xlim.append(np.max(Data['x']))
        xlim.append(np.min(Data['x']))
        ylim.append(np.max(Data['y']))
        ylim.append(np.min(Data['y']))

    xlims = [np.min(xlim),np.max(xlim)]
    ylims = [np.min(ylim),np.max(ylim)]

    if isinstance(cmap,str):
        cmaps = cmap
        cmap = dict()
        for iV,V in enumerate(Vars):
            cmap[V] = cmaps
    else:
        for iV,V in enumerate(Vars):
            try:
                cmap[V] = cmap[V]
            except KeyError:
                cmap[V] = 'jet'
        
    Plus = -15+((len(Vars)-1)*10)
    # Figure
    Inch = 0.393701
    fH = (25+(5*(len(Projects)-1)))*Inch
    fV = (20+Plus)* Inch
    F,axs = plt.subplots(len(Vars),3,figsize=(fH,fV))
    axs = axs.ravel()
    plot = 0
    # Rows
    for iV,V in enumerate(Vars):
        plt.rcParams.update({
            'xtick.labelsize': 12,'xtick.major.size': 6,
            'ytick.labelsize': 12,'ytick.major.size': 6,
            })
        # Columns
        plots = []
        for iC,DataT in enumerate(Projects):
            if iC <= 1:
                print(f'{V}{Vars2[iC]}')
                Data = DataT[f'{V}{Vars2[iC]}']
                TT = 0
                b = axs[plot].pcolormesh(Data['x'],Data['y'],Data['V'][TT,:,:],
                        vmax=vmaxT[V],vmin=vminT[V],cmap=cmap[V])
            elif iC == 2:
                Data = DataT[f'Dif{V}']
                TT = 0
                b = axs[plot].pcolormesh(Data['x'],Data['y'],Data['V'][TT,:,:],
                        vmax=vmaxT[f'Dif{V}'],vmin=vminT[f'Dif{V}'],
                        cmap=cmap[f'Dif{V}'],
                        norm=MidpointNormalize(midpoint=0,vmin=vminT[f'Dif{V}'],
                            vmax=vmaxT[f'Dif{V}']))

            plots.append(axs[plot].get_position().get_points().flatten())

            if iC == 1:
                plt.rcParams.update({
                    'xtick.labelsize': 12,'xtick.major.size': 2,
                    'ytick.labelsize': 12,'ytick.major.size': 2,
                    })
                ax_cbar1 = F.add_axes([plots[-1][2]-0.02, plots[-1][1], 
                    0.015, plots[-1][3]-plots[-1][1]])
                cbar = plt.colorbar(b,boundaries=bounds[V],ticks=v[V],
                        cax=ax_cbar1,orientation='vertical')
                # cbar = plt.colorbar(b,boundaries=bounds[V],ticks=v[V],
                #         ax=axs[plot],orientation='vertical')
                # if Data['VarUnits'] == '':
                #     cbar.set_label('[-]')
                # else:
                #     cbar.set_label('[{}]'.format(Data['VarUnits'],))

            elif iC == 2:
                plt.rcParams.update({
                    'xtick.labelsize': 12,'xtick.major.size': 2,
                    'ytick.labelsize': 12,'ytick.major.size': 2,
                    })
                ax_cbar1 = F.add_axes([plots[-1][2]+0.01, plots[-1][1], 
                    0.015, plots[-1][3]-plots[-1][1]])
                cbar = plt.colorbar(b,boundaries=bounds[f'Dif{V}'],ticks=v[f'Dif{V}'],
                        cax=ax_cbar1,orientation='vertical')
                # cbar = plt.colorbar(b,boundaries=bounds[f'Dif{V}'],ticks=v[f'Dif{V}'],
                #         ax=axs[plot],orientation='vertical')
                if Data['VarUnits'] == '':
                    cbar.set_label('[-]')
                else:
                    cbar.set_label('[{}]'.format(Data['VarUnits'],))

            if not(S is None):
                if len(S) == 2:
                    if SB[iV]:
                        speed = np.sqrt(DataT[S[0]]['V'][TT,:,:]**2+DataT[S[1]]['V'][TT,:,:]**2)*1000
                        lws = speed/np.nanmax(speed)*4
                        axs[plot].streamplot(DataT[S[0]]['x'],DataT[S[0]]['y'],
                                DataT[S[0]]['V'][TT,:,:],DataT[S[1]]['V'][TT,:,:],color='k',
                                density=den,linewidth=lws)
                        # axs[iV].quiver(DataT[S[0]]['x'],DataT[S[0]]['y'],
                        #     DataT[S[0]]['V'][TT,:,:],DataT[S[1]]['V'][TT,:,:],
                        #     color='k')
                        Title = Data ['Description'] + ', Streamlines: {}'.format(DataT[S[0]]['Description'])
                    else:
                        Title = Data['Description']
                else:
                    print(' Must have 2 components')
            else:
                Title = Data['Description']
            axs[plot].set_xlim(xlims)
            axs[plot].set_ylim(ylims)

            # Title
            if iC == 0:
                F.text(0.02,plots[-1][3]-0.11,f'{Title}',
                        fontsize=14,rotation=90,weight = 'bold',
                        horizontalalignment='center',
                        verticalalignment='center')

            if not(Titles is None):
                if iV == 0:
                    axs[plot].set_title(Titles[iC],fontsize=14)

            if iV == len(Vars) - 1:
                axs[plot].set_xlabel('Horizontal Distance [km]',fontsize=14)
            if iC == 0 :
                axs[plot].set_ylabel('Elevation [km]',fontsize=14)
            # Hide the right and top spines
            axs[plot].spines['right'].set_visible(False)
            axs[plot].spines['top'].set_visible(False)
            if iC > 0:
                yTL = axs[plot].yaxis.get_ticklocs() # List of Ticks in y
                axs[plot].set_yticklabels(['']*len(yTL))
            if iV < len(Vars)-1:
                xTL = axs[plot].xaxis.get_ticklocs() # List of Ticks in y
                axs[plot].set_xticklabels(['']*len(xTL))
            plot += 1
    # plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    plt.subplots_adjust(hspace=0.20)
    plt.subplots_adjust(wspace=0.40)
    for i in range(plot):
        axes = axs[i]
        yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # Minor tick value
        minorLocatory = MultipleLocator(MyL)
        axes.yaxis.set_minor_locator(minorLocatory)
        xTL = axes.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # Minor tick value
        minorLocatorx = MultipleLocator(MxL)
        axes.xaxis.set_minor_locator(minorLocatorx)
    plt.savefig('{}{}'.format(PathImg,Name),
        format=Name.split('.')[-1],dpi=200,bbox_inch='tight')
    plt.close('all')
    return 

def LineGraph(Data,Variables,Title,xlabel,ylabel,PathImg,Name,
        kwargs=None,kwargslegend=None,logx=None,logy=None,
        vlim=None,Invertx=False,Inverty=False):
    '''
    DESCRIPTION:
         This function creates a histogram with the information given.
    ____________________________________________________________
    INPUT:
        :param Data: a ndarray, Data for plot.
        :param xlabel: a str, xlabel.
        :param ylabel: a str, ylabel.
        :param PathImg: a str, Path to save the image.
        :param Name: a str, Name of the image.
        :param kwargs: a dict, dict with values for the plot.
        :param NumbP: an int, Number of plots.
    ____________________________________________________________
    OUTPUT:
        :return: This function returns a plot.
    '''
    # Generate Figurer
    utl.CrFolder(PathImg)
    # Creating Figure
    plt.rcParams.update({'font.size': 18,'font.family': 'sans-serif'\
        ,'font.sans-serif': 'Arial'\
        ,'xtick.labelsize': 18,'xtick.major.size': 6,'xtick.minor.size': 4\
        ,'xtick.major.width': 1,'xtick.minor.width': 1\
        ,'ytick.labelsize': 18,'ytick.major.size': 12,'ytick.minor.size': 4\
        ,'ytick.major.width': 1,'ytick.minor.width': 1\
        ,'axes.linewidth':1\
        ,})
    Inch = 0.393701
    fH = 20*Inch
    fV = 15*Inch
    plt.figure(figsize=(fH,fV))
    for ivar,var in enumerate(Variables):
        if not(kwargs is None):
            kwargsS = kwargs[ivar]
            plt.plot(Data['t{}'.format(var)],Data['{}'.format(var)],
                    **kwargsS)
        else:
            plt.plot(Data['t{}'.format(var)],Data['{}'.format(var)])

    plt.xlabel(xlabel,fontsize=18)
    plt.ylabel(ylabel,fontsize=18)
    plt.title(Title,fontsize=18)
    # axis
    if not(vlim is None):
        plt.xlim(vlim[0])
        plt.ylim(vlim[1])
    
    if not(logx is None):
        plt.xscale(logx)
    if not(logy is None):
        plt.yscale(logy)

    if Invertx:
        plt.gca().invert_xaxis()
    if Inverty:
        plt.gca().invert_yaxis()
    plt.grid(b=True, which='major', linestyle='-')
    plt.grid(b=True, which='minor', linestyle='--')
    # Legend
    if not(kwargs is None) and not(kwargslegend is None):
        plt.legend(**kwargslegend)
    plt.tight_layout()
    plt.savefig('{}{}'.format(PathImg,Name),format=Name.split('.')[-1],dpi=300)
    plt.close('all')
    return

def GeneralGraph(Data,Variables,Title,xlabel,ylabel,PathImg,Name,
        kwargs=None,kwargslegend=None,logx=None,logy=None,
        vlim=None,Invertx=False,Inverty=False,plotfun=None):
    '''
    DESCRIPTION:
         This function creates a plot with the information given.
    ____________________________________________________________
    INPUT:
        :param Data: a dict, 'tVar' for x and 'Var' for y.
        :param Variables: a list, List of variables.
        :param Title: a str, Title.
        :param xlabel: a str, xlabel.
        :param ylabel: a str, ylabel.
        :param PathImg: a str, Path to save the image.
        :param Name: a str, Name of the image with extension.
        :param kwargs: a dict, dict with values for the plot.
        :param kwargsLegend: a dict, dict with values for the 
                             legend.
        :param logx: an str, set the x axis to log, 
                     for log use 'log'.
        :param logy: an str, set the y axis to log.
                     for log use 'log'.
        :param vlim: a list, list with two lists, for x and y
                     min and max.
        :param Invertx: a bool, Boolean to invert x axis.
        :param Inverty: a bool, Boolean to invert y axis.
        :param plotfun: a dict, dictionary with the plot 
                        functions for each variable.
                        if None the default plot would be
                        plt.plot.
    ____________________________________________________________
    OUTPUT:
        :return: This function returns a plot.
    '''
    # Generate Figurer
    utl.CrFolder(PathImg)
    # Creating Figure
    plt.rcParams.update({'font.size': 18,'font.family': 'sans-serif'\
        ,'font.sans-serif': 'Arial'\
        ,'xtick.labelsize': 18,'xtick.major.size': 6,'xtick.minor.size': 4\
        ,'xtick.major.width': 1,'xtick.minor.width': 1\
        ,'ytick.labelsize': 18,'ytick.major.size': 12,'ytick.minor.size': 4\
        ,'ytick.major.width': 1,'ytick.minor.width': 1\
        ,'axes.linewidth':1\
        ,})
    Inch = 0.393701
    fH = 20*Inch
    fV = 15*Inch
    plt.figure(figsize=(fH,fV))
    for ivar,var in enumerate(Variables):

        # Set plotting function
        if plotfun is None:
            plot = plt.plot
        else:
            try:
                plot = plotfun[var]
            except KeyError:
                plot = plt.plot

        if not(kwargs is None):
            try:
                kwargsS = kwargs[var]
            except TypeError:
                kwargsS = kwargs[ivar]

            if plot is plt.axhline or plot is plt.axvline or plot is sns.distplot:
                plot(Data[f'{var}'],**kwargsS)
            else:
                plot(Data[f't{var}'],Data[f'{var}'],
                        **kwargsS)
        else:
            if plot is plt.axhline or plot is plt.axvline or plot is sns.distplot:
                plot(Data[f'{var}'])
            else:
                plot(Data[f't{var}'],Data[f'{var}'])

    plt.xlabel(xlabel,fontsize=18)
    plt.ylabel(ylabel,fontsize=18)
    plt.title(Title,fontsize=18)
    # axis
    if not(vlim is None):
        if not(vlim[0] is None):
            plt.xlim(vlim[0])
        if not(vlim[1] is None):
            plt.ylim(vlim[1])
    
    if not(logx is None):
        plt.xscale(logx)
    if not(logy is None):
        plt.yscale(logy)

    if Invertx:
        plt.gca().invert_xaxis()
    if Inverty:
        plt.gca().invert_yaxis()
    plt.grid(b=True, which='major', linestyle='-')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Legend
    if not(kwargs is None) and not(kwargslegend is None):
        plt.legend(**kwargslegend)
    plt.tight_layout()
    # Include minor axis
    if logx is None and logy is None:
        axes = plt.gca()
        yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
        MyL = (yTL[1]-yTL[0])/5 # Minor tick value
        minorLocatory = MultipleLocator(MyL)
        plt.gca().yaxis.set_minor_locator(minorLocatory)
        xTL = axes.xaxis.get_ticklocs() # List of Ticks in x
        MxL = (xTL[1]-xTL[0])/5 # Minor tick value
        minorLocatorx = MultipleLocator(MxL)
        plt.gca().xaxis.set_minor_locator(minorLocatorx)

    plt.savefig('{}{}'.format(PathImg,Name),format=Name.split('.')[-1],dpi=300)
    plt.close('all')
    return

def LineSubplots(DataL,Variables,Title,xlabel,ylabel,PathImg,Name,Labels=None,logx=None,
        logy=None,vlim=None,Invertx=False,Inverty=False,loc=0,cols=3,fH=40,fV=30):
    '''
    DESCRIPTION:
         This function makes a subplot type plot with a define
         number of columns and len(DataT) rows.
    ____________________________________________________________
    INPUT:
        :param DataL: A list, list with dictionaries of every
                      line that will be plotted.
        :param Variables: A list, key variables for the plot.
        :param Title: A str, title of each plot.
        :param xLabel: A str, x label.
        :param yLabel: A str, y label.
        :param PathImg: A str, Path to save the image.
        :param Name: A str, Name of the image with extension.
        :param Labels: A str, Labels on the graph.
    ____________________________________________________________
    OUTPUT:
        :return Var: Description.
    '''
    
    # Generate Figurer
    utl.CrFolder(PathImg)
    # Creating Figure
    plt.rcParams.update({'font.size': 18,'font.family': 'sans-serif'\
        ,'font.sans-serif': 'Arial'\
        ,'xtick.labelsize': 18,'xtick.major.size': 6,'xtick.minor.size': 4\
        ,'xtick.major.width': 1,'xtick.minor.width': 1\
        ,'ytick.labelsize': 18,'ytick.major.size': 12,'ytick.minor.size': 4\
        ,'ytick.major.width': 1,'ytick.minor.width': 1\
        ,'axes.linewidth':1\
        ,})
    linestyles = ['-','--','-.',':']
    alpha=[0.3,1,1,1]
    Inch = 0.393701
    fH = fH*Inch
    fV = fV*Inch
    Dim = (int(np.ceil(len(DataL)/cols)),cols)
    fig, axs = plt.subplots(Dim[0],Dim[1],figsize=(fH,fV))#,sharex=True,sharey=True)
    axs = axs.ravel()

    for iData,Data in enumerate(DataL):
        for ivar,var in enumerate(Variables):
            # plt.plot(Data['t{}'.format(ivar+1)],Data[var],
            if len(Variables) <= 4:
                if not(Labels is None):
                    axs[iData].plot(Data['t{}'.format(var)],Data['{}'.format(var)],
                            label=Labels[ivar],linestyle=linestyles[ivar],color='k',
                            alpha=alpha[ivar])
                else:
                    axs[iData].plot(Data['t{}'.format(var)],Data['{}'.format(var)],
                            linestyle=linestyles[ivar],color='k')
            else:
                axs[iData].plot(Data['t{}'.format(var)],Data['{}'.format(var)],
                        label=Labels[ivar])


        if iData >= Dim[0]*cols - cols:
            axs[iData].set_xlabel(xlabel,fontsize=18)
        if iData % cols == 0:
            axs[iData].set_ylabel(ylabel,fontsize=18)
        axs[iData].set_title(Title[iData],fontsize=18)
        # axis
        if not(vlim is None):
            axs[iData].set_xlim(vlim[0])
            axs[iData].set_ylim(vlim[1])
        
        if not(logx is None):
            axs[iData].set_xscale(logx)
        if not(logy is None):
            axs[iData].set_yscale(logy)

        if Invertx:
            axs[iData].gca().invert_xaxis()
        if Inverty:
            axs[iData].gca().invert_yaxis()
        axs[iData].grid(b=True, which='major', linestyle='-')
        axs[iData].grid(b=True, which='minor', linestyle='--')

        # Legend
        if not(Labels is None):
            if iData == cols-1:
                axs[iData].legend(loc=loc,fontsize=10,ncol=2)
    if iData != len(axs)-1:
        for iD in range(iData+1,len(axs)):
            fig.delaxes(axs[iD])
    plt.tight_layout()
    plt.savefig('{}{}'.format(PathImg,Name),format=Name.split('.')[-1],dpi=300)
    plt.close('all')
    return

def Several(X,Y,xLabel,yLabel,Name,flagT=True,Title='',PathImg='',
        Labels=None,Inverse=False,limits=True,**args):
    '''
    DESCRIPTION:
        This Function produces line graphs

    _________________________________________________________________________

    INPUT:
        :param X: A narray or list, X vector
        :param Y: A narray or list, Y vector.
        :param xLabel: A string, Label of x. 
        :param yLabel: A string, Label of y.
        :param Name: A str, Name of the Figure.
        :param flagT: A boolean, Flag to include title.
        :param Title: A str, Title of the figure
        :param yLabel: A list, List with labels for legend.
        :param PathImg: A str, Path to save the figure.
        :param **args: Arguments to change the plot function.
    _________________________________________________________________________
    
    OUTPUT:
        This function gives a figure back.
    '''
    # Create Folder
    # utl.CrFolder(PathImg)
    Inch = 0.393701
    fH = 20*Inch
    fV = fH * (2/3)

    # Create Figure
    F = plt.figure(figsize=(fH,fV))
    # Figure Parameters
    plt.rcParams.update({'font.size': 14,'font.family': 'sans-serif'\
        ,'font.sans-serif': 'Arial'\
        ,'xtick.labelsize': 14,'xtick.major.size': 6,'xtick.minor.size': 4\
        ,'xtick.major.width': 1,'xtick.minor.width': 1\
        ,'ytick.labelsize': 14,'ytick.major.size': 12,'ytick.minor.size': 4\
        ,'ytick.major.width': 1,'ytick.minor.width': 1\
        ,'axes.linewidth':1\
        ,'grid.alpha':0.1,'grid.linestyle':'-'})
    plt.rcParams['agg.path.chunksize'] = 20000
    plt.tick_params(axis='x',which='both',bottom=True,top=False,\
        labelbottom=True,direction='out')
    plt.tick_params(axis='y',which='both',left=True,right=False,\
        labelleft=True)
    plt.tick_params(axis='y',which='major',direction='inout') 
    plt.grid()

    linestyles=['-','--','-.',':']

    # plots
    if len(Y.shape) > 1:
        if Labels is None:
            for i in range(len(Y)):
                if Inverse:
                    plt.plot(Y[i],X[i])
                else:
                    plt.plot(X[i],Y[i])
        else:
            for i in range(len(Y)):
                if Inverse:
                    plt.plot(Y[i],X[i],label=Labels[i])
                else:
                    plt.plot(X[i],Y[i],label=Labels[i],color='k',linestyle=linestyles[i])
    else:
        plt.plot(X,Y,**args)
    # Axis
    axes = plt.gca()
    # Minor axes ticks
    # yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
    # MyL = (yTL[1]-yTL[0])/5 # Minor tick value
    # minorLocatory = MultipleLocator(MyL)
    # plt.gca().yaxis.set_minor_locator(minorLocatory)
    # xTL = axes.xaxis.get_ticklocs() # List of Ticks in y
    # MxL = (xTL[1]-xTL[0])/5 # Minor tick value
    # minorLocatory = MultipleLocator(MxL)
    # plt.gca().yaxis.set_minor_locator(minorLocatory)

    # Labels
    if flagT:
        plt.title(Title,fontsize=14)
    plt.ylabel(yLabel,fontsize=14)
    plt.xlabel(xLabel,fontsize=14)
    if limits:
        plt.xlim([0,1])
        plt.ylim([0,1])

    if not(Labels is None):
        plt.legend(loc=0,fontsize=12)

    # Maximizing figure
    plt.tight_layout()
    # Saving Figure
    plt.savefig(PathImg + Name,format=Name.split('.')[-1],dpi=200)
    plt.close('all')

def CurveEPS(X,Y,PathImg,Name,figsize=(4,4),axis=False):
    plt.figure(figsize=figsize)
    plt.plot(X,Y,color='k')
    if axis:
        plt.axis('on')
        plt.grid()
    else:
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('{}{}.eps'.format(PathImg,Name),format='eps',dpi=1000)
    plt.close('all')

def HistogramG(Data,xlabel,ylabel,Title='',PathImg='',Name=None,
        NumbP=1,kwargs=None,kwlegend=None):
    '''
    DESCRIPTION:
         This function creates a histogram with the information given.
    ____________________________________________________________
    INPUT:
        :param Data: a ndarray, Data for plot.
        :param xlabel: a str, xlabel.
        :param ylabel: a str, ylabel.
        :param PathImg: a str, Path to save the image.
        :param Name: a str, Name of the image.
        :param kwargs: a dict, dict with values for the plot.
                       the parameters are the same that
                       sns.distplot().
        :param NumbP: an int, Number of plots.
    ____________________________________________________________
    OUTPUT:
        :return: This function returns a plot.
    '''

    Inch = 0.393701
    fH = 20*Inch
    fV = fH*2/3
    plt.rcParams.update({'font.size': 18,'font.family': 'sans-serif'\
        ,'font.sans-serif': 'Arial'\
        ,'xtick.labelsize': 18,'xtick.major.size': 6,'xtick.minor.size': 4\
        ,'xtick.major.width': 1,'xtick.minor.width': 1\
        ,'ytick.labelsize': 18,'ytick.major.size': 12,'ytick.minor.size': 4\
        ,'ytick.major.width': 1,'ytick.minor.width': 1\
        ,'axes.linewidth':1\
        ,'grid.alpha':0.6,'grid.linestyle':'-'})
    plt.rcParams['agg.path.chunksize'] = 20000
    plt.tick_params(axis='x',which='both',bottom=True,top=False,\
        labelbottom=True,direction='out')
    plt.tick_params(axis='y',which='both',left=True,right=False,\
        labelleft=True)
    plt.tick_params(axis='y',which='major',direction='inout') 
    plt.grid()

    plt.figure(figsize=(fH,fV))
    if NumbP == 1:
        q = ~(np.isnan(Data))
        Data = Data[q]
        if not(kwargs is None):
            sns.distplot(Data,**kwargs)
        else:
            sns.distplot(Data)
    else:
        if len(Data) != NumbP:
            raise Exception('len(Data) and NumbP must have the same number \nlen(Data)={}, NumbP={}'.format(len(Data),NumbP))
        for p in range(NumbP):
            if isinstance(kwargs,list):
                kwargsp = kwargs[p]
            else:
                kwargsp = kwargs
            Datap = Data[p]
            q = ~(np.isnan(Datap))
            Datap = Datap[q]
            if not(kwargs is None):
                sns.distplot(Datap,**kwargsp)
            else:
                sns.distplot(Datap)
            
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(Title)
    if not(kwlegend is None):
        plt.legend(**kwlegend)
    
    plt.tight_layout()
    if not(Name is None):
        # Creating Folder
        utl.CrFolder(PathImg)
        plt.savefig(PathImg + Name,format=Name.split('.')[-1],dpi=200)
        plt.close('all')
    
    return

def BarPlot(Data,Variables,Title,xlabel,ylabel,PathImg,Name,
        kwargs=None,logx=None,logy=None,
        vlim=None,Invertx=False,Inverty=False):
    '''
    DESCRIPTION:
         This function creates a histogram with the information given.
    ____________________________________________________________
    INPUT:
        :param Data: a ndarray, Data for plot.
        :param Variables: a list, Variables of x axis.
        :param Title: a str, Title.
        :param xlabel: a str, xlabel.
        :param ylabel: a str, ylabel.
        :param PathImg: a str, Path to save the image.
        :param Name: a str, Name of the image.
        :param kwargs: a dict, dict with values for the plot.
        :param NumbP: an int, Number of plots.
    ____________________________________________________________
    OUTPUT:
        :return: This function returns a plot.
    '''
    # Generate Figurer
    utl.CrFolder(PathImg)
    # Creating Figure
    plt.rcParams.update({'font.size': 18,'font.family': 'sans-serif'\
        ,'font.sans-serif': 'Arial'\
        ,'xtick.labelsize': 18,'xtick.major.size': 6,'xtick.minor.size': 4\
        ,'xtick.major.width': 1,'xtick.minor.width': 1\
        ,'ytick.labelsize': 18,'ytick.major.size': 12,'ytick.minor.size': 4\
        ,'ytick.major.width': 1,'ytick.minor.width': 1\
        ,'axes.linewidth':1\
        ,})
    Inch = 0.393701
    fH = 20*Inch
    fV = 15*Inch
    plt.figure(figsize=(fH,fV))
    if not(kwargs is None):
        kwargsS = kwargs
        plt.bar(np.arange(len(Data)),Data,**kwargsS)
    else:
        plt.bar(np.arange(len(Data)),Data)

    plt.xticks(np.arange(len(Data)),Variables)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(Title)
    # axis
    if not(vlim is None):
        plt.xlim(vlim[0])
        plt.ylim(vlim[1])
    
    if not(logx is None):
        plt.xscale(logx)
    if not(logy is None):
        plt.yscale(logy)

    if Invertx:
        plt.gca().invert_xaxis()
    if Inverty:
        plt.gca().invert_yaxis()

    plt.grid(b=True, which='major', linestyle='-')
    # plt.grid(b=True, which='minor', linestyle='--')
    # Legend
    # if not(kwargs is None) and not(kwargslegend is None):
    #     plt.legend(**kwargslegend)
    plt.tight_layout()
    axes = plt.gca()
    yTL = axes.yaxis.get_ticklocs() # List of Ticks in y
    MyL = (yTL[1]-yTL[0])/5 # Minor tick value
    minorLocatory = MultipleLocator(MyL)
    plt.gca().yaxis.set_minor_locator(minorLocatory)

    plt.savefig('{}{}'.format(PathImg,Name),format=Name.split('.')[-1],dpi=300)
    plt.close('all')
    return

