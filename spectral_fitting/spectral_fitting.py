"""
BSD-3

Supplementary code for the IEEE Access Publication "Advancements In Spectral 
Power Distribution Modeling Of Light-Emitting Diodes" 
DOI: 10.1109/ACCESS.2022.3197280 
by Simon Benkner, 
Laboratory for Adaptive Lighting Systems and Visual Processing, 
Department of Electrcical Engineering, 
Technical University of Darmstadt, Germany.

"""

import numpy as np
from lmfit import models, Model, Parameters
import models_distributions as dist_mod
from scipy.integrate import simps
from colour import SpectralDistribution, sd_to_XYZ, wavelength_to_XYZ
import pandas as pd
import os
from functools import partial
import concurrent.futures as cf
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker



np.seterr(all='ignore')

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H-%M-%S')
logger = logging.getLogger(__name__)

warnings.filterwarnings("error")

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 250)
pd.options.mode.chained_assignment = None

plt.rcParams["figure.figsize"] = (3.5,1.75)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "tab:gray"
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams['xtick.direction'] =  'in'
plt.rcParams['ytick.direction'] =  'in'
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6
plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('text', usetex=False)

plt.rcParams['legend.loc'] = "best"



class OOMFormatter(ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

def calc_uv1976(XYZ: list) -> tuple:
    """
    Calculate the CIE 1976 UCS chromaticity coordinates u' and v'.
    :param XYZ: List of Tristimulus Values X,Y,Z.
    :type XYZ: list
    :return: Tuple of u', v' CIE 1976 UCS chromaticity coordinates.
    :rtype: tuple

    """
    try:
        if 0 in XYZ or np.nan in XYZ:
            u, v = np.nan, np.nan
        else:
            denom_true = (XYZ[0] + 15*XYZ[1] + 3*XYZ[2])
            u = 4*XYZ[0]/denom_true
            v = 9*XYZ[1]/denom_true
        return u,v
    except Exception as e:
        logger.exception(e)
        return [np.nan, np.nan, np.nan]


class FunctionFitter:
    """
    The FunctionFitter class provides the necassary fitting methods for SPD
    approximation with commonly used probabiloity density functions (pdf).
    :param output_path: Path to output the fit results to as csv file.
    :type output_path: str
    :return: No return.
    :rtype: None

    """ 
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path
        
        # parameters for multiprocessing
        self.max_workers = 39 # number of workers for parallel fitting
        self.chunk_size = 19 # size of batch to pass to the workers
        
        # General fit parameters used by all functions.
        self.tiny = 1.0e-15 # small, but not zero to avoid ZeroDivisionExceptions
        self.s2pi = np.sqrt(2*np.pi)
        self.s2 = np.sqrt(2.0)
        
        # standard deviation
        self.sigma_val = 10
        self.sigma_min = 1 
        self.sigma_max = 300
        
        # mean / peak wavelength
        self.center = 450
        self.center_min = 400
        self.center_max = 800
        
        # amplitude / magnitude
        self.amplitude = 0.001
        self.amplitude_min = self.tiny
        self.amplitude_max = 100
        
        self.fail_return = 0
        
        # availible pdfs for fitting
        # internal: LMFIT function / external: own declared function
        
        self.mod_dict = {
                    'gaussian_split' :  ['external',
                                         dist_mod.gaussian_split, 
                                         [('sigmar', 
                                           self.sigma_val, 
                                           self.sigma_min, 
                                           self.sigma_max)]
                                         ],
                    'lorentzian_sec_ord' :  ['external',
                                             dist_mod.lorentzian_sec_ord, 
                                             [],
                                            ],
                    'logistic_power_peak' : ['external',
                                             dist_mod.logistic_power_peak, 
                                             [('s', 1, 0.001, 100)],
                                             ],
                    'logistic_asymm_peak' : ['external',
                                             dist_mod.logistic_asymm_peak, 
                                             [('s', 1, self.tiny, 100)]
                                             ],
                    'pearson_vii' : ['external',
                                     dist_mod.pearson_vii, 
                                     [('s', 1, self.tiny, 100)]
                                     ],
                    'pearson_vii_split' : ['external',
                                           dist_mod.pearson_vii_split, 
                                           [('sigmar', 
                                             self.sigma_val, 
                                             self.sigma_min, 
                                             self.sigma_max),
                                            ('s', 1, self.tiny, 100),
                                            ('sr', 1, self.tiny, 100),
                                            ],
                                           ],
                    'sigmoid_double_asymm' : ['external',
                                              dist_mod.sigmoid_double_asymm, 
                                              [('s1', 10, self.tiny, 100),
                                               ('s2', 10, self.tiny, 100),
                                               ]
                                              ],
                    'Gaussian' : ['internal',
                                  models.GaussianModel,
                                  [],
                                  ],
                    'Lorentzian' : ['internal',
                                    models.LorentzianModel,
                                    [],
                                    ],
                    'SplitLorentzian' : ['internal',
                                         models.SplitLorentzianModel,
                                         [('sigma_r', 
                                           self.sigma_val, 
                                           self.sigma_min, 
                                           self.sigma_max)],
                                         ],
                    'Voigt' : ['internal',
                          models.VoigtModel,
                          [],
                          ],
                    'PseudoVoigt' : ['internal',
                                     models.PseudoVoigtModel,
                                     [('fraction', 1, self.tiny, 100)],
                                     ],
                    'Moffat' : ['internal',
                                models.MoffatModel,
                                [('beta', 1, self.tiny, 100)],
                                ],
                    'Pearson7' : ['internal',
                                  models.Pearson7Model,
                                  [('expon', 1, self.tiny, 100)],
                                  ],
                    'StudentsT' : ['internal',
                                   models.StudentsTModel,
                                   [],
                                   ],
                    'Lognormal' : ['internal',
                                   models.LognormalModel,
                                   [],
                                   ],
                    'ExponentialGaussian' : ['internal',
                                             models.ExponentialGaussianModel,
                                             [('gamma', 0, -100, 100)],
                                             ],
                    'SkewedGaussian' : ['internal',
                                        models.SkewedGaussianModel,
                                        [('gamma', 0, -100, 100)],
                                        ],
                    'SkewedVoigt' : ['internal',
                                     models.SkewedVoigtModel,
                                     [('gamma', 1, self.tiny, 10),
                                      ('skew', 1, self.tiny, 100)],
                                     ],
                    
                    # Following functions resulted repeated fitting errors
                    
                    # 'FermiDirac' : ['internal',
                    #       models.ThermalDistributionModel,
                    #       [('kt', 0.05, self.tiny, 0.99)],
                    #       ],
                    # 'BoseEnstein' : ['internal',
                    #       models.ThermalDistributionModel,
                    #       [('kt', 1, -1000, 5000)],
                    #       ],
                    # 'MaxwellBoltzmann' : ['internal',
                    #       models.ThermalDistributionModel,
                    #       [('kt', 1, -1000, 5000)],
                          # ],
                    # 'Constant' : ['internal',
                    #       models.ConstantModel,
                    #       [],
                    #       ],
                    # 'Exponential' : ['internal',
                    #       models.ExponentialModel,
                    #       [('decay', 0, -1000, 1000)],
                    #       ],
                    # 'PowerLaw' : ['internal',
                    #       models.PowerLawModel,
                    #       [('exponent', 0, -1000, 1000)],
                    #       ],
                    
                    }


    def fitter(self, subset: tuple, x_true: np.array, y_true: np.array, 
               u_true: float, v_true: float) -> dict:
        """
        Fit a given set of pdfs onto a spectral power distribution.
        :param subset: Tuple of superimposed pdfs to be fitted. 
        :type subset: tuple
        :param x_true: True x values / wavelengths.
        :type x_true: np.array
        :param y_true: True y values / spectral power.
        :type y_true: np.array
        :param u_true: True u' CIE 1976 UCS chromaticity coordinate.
        :type u_true: float
        :param v_true: True v' CIE 1976 UCS chromaticity coordinate.
        :type v_true: float
        :return: Dictionary with fit results: Name, Number of fitted functions,
                    r2, AIC, BIC, power deviation, chromaticity difference,
                    model parameters.
        :rtype: dict

        """
        try:
            params = Parameters()
            for i, name in enumerate(subset):
                val = self.mod_dict.get(name)
                if val[0] == 'internal':
                    if name == 'FermiDirac':
                        m = val[1](form='fermi', nan_policy='propagate') 
                    elif name == 'BoseEnstein':
                        m = val[1](form='bose', nan_policy='propagate') 
                    elif name == 'MaxwellBoltzmann':
                        m = val[1](form='maxwell', nan_policy='propagate') 
                    else:
                        m = val[1](nan_policy='propagate')                
                else:
                    m = Model(val[1], nan_policy='propagate')
                m.prefix = f'm{i}_'
                params.update(m.make_params())
                params[f'm{i}_amplitude'].set(value=self.amplitude,
                                              min=self.amplitude_min, 
                                              max=self.amplitude_max
                                              )
                if not 'thermal' in str(m):                
                    if not any(fc in str(m ) for fc in ['exponential', 
                                                        'constant', 
                                                        'powerlaw']):
                        params[f'm{i}_center'].set(value=self.center, 
                                                    min=self.center_min, 
                                                    max=self.center_max
                                                    )
                        params[f'm{i}_sigma'].set(value=self.sigma_val, 
                                                  min=self.sigma_min, 
                                                  max=self.sigma_max
                                                  ) 
                if len(val[2]) > 0:
                    for par, value, minimum, maximum in val[2]:
                        params[f'm{i}_{par}'].set(value=value, 
                                                  min=minimum, 
                                                  max=maximum)
                if i == 0:
                    model = m
                else:
                    model = model + m
        except Exception as e:
            logger.exception(e)
            pass  
        try:
            out = model.fit(y_true, params, x=x_true)
            r2 = 1 - out.redchi / np.var(y_true, ddof=2)
            aic = out.aic
            bic = out.bic
            power_dev = round(
                    (simps(out.best_fit,x_true) / simps(y_true,x_true)) - 1, 
                    3)
            spd_fit = pd.Series(data=out.best_fit, index=x_true)
            XYZ_fit = sd_to_XYZ(SpectralDistribution(spd_fit))
            if 0 in XYZ_fit or np.nan in XYZ_fit:
                duv = np.nan
            else:
                u_fit, v_fit = calc_uv1976(XYZ_fit)
                duv = 1000*np.sqrt( (u_fit-u_true)**2 + (v_fit-v_true)**2) # in 10^-3 
            return {'name': '+'.join(subset), 
                            'number_functions': len(subset),
                            'r2' : r2,
                            # 'chi2': chi2,
                            'aic': aic,
                            'bic': bic,
                            'power_dev' : power_dev,
                            'duv' : duv,
                            'best_fit':out.best_fit,
                            'model' : out.params
                            }
        except (ValueError,Exception,RuntimeWarning,OverflowError) as e:
            logger.exception(e)
            return {}
        
    
    def _chunk_list(self, lst: list, n: int) -> list:
        """
        Genertaor that yields successive n-sized chunks from lst.
        :param lst: List to chunk.
        :type lst: list
        :param n: Number of chunks.
        :type n: int
        :yield: Chunk of lst.
        :rtype: list

        """
        for i in range(0, len(lst), n):            
            print(f'Chunk {int(1+i/n)} / {int(1+len(lst)/n)}')
            yield lst[i:i + n], int(1+i/n)
    
        
    def fit_model(self, spec: pd.DataFrame, model:tuple) -> dict:
        """
        Fits a specific model to a given spd. 
        :param spec: Spectral power distribution.
        :type spec: pd.DataFrame
        :param model: Tuple of models.
        :type model: tuple
        :return: Fit results.
        :rtype: dict

        """
        spec['wl'] = spec.wl.round(0)
        spec = spec.groupby('wl').mean().reset_index()
        spec = spec[spec.wl >=360]
        spec = spec[spec.wl <= 780]
        x_true = spec.to_numpy().T[0]
        y_true = spec.to_numpy().T[1]
        spd_true = pd.Series(data=y_true, index=x_true)
        XYZ_true = sd_to_XYZ(SpectralDistribution(spd_true))
        u_true, v_true = calc_uv1976(XYZ_true)
        
        return self.fitter(subset=model, 
                           x_true=x_true, 
                           y_true=y_true, 
                           u_true=u_true, 
                           v_true=v_true
                           )
    
    
    def fit_spd(self, spec: pd.DataFrame, 
                max_nmbr_funcs:int = 1, 
                min_nmbr_funcs:int = 1,
                file: str = None
                ) -> None:
        """
        Fits all proposed pdfs to a given SPD in parallel. 
        :param spec: Pandas DataFrame containing the SPD.
        :type spec: pd.DataFrame
        :param max_nmbr_funcs: Maximum number of superimposed functions to fit 
        to a SPD, defaults to 1
        :type max_nmbr_funcs: int, optional
        :param min_nmbr_funcs: Minimum number of superimposed functions to fit 
        to a SPD, defaults to 1
        :type min_nmbr_funcs: int, optional
        :param file: Output file to store results, defaults to None
        :type file: str, optional
        :return: No return.
        :rtype: None

        """
        spec['wl'] = spec.wl.round(0)
        spec = spec.groupby('wl').mean().reset_index()
        spec = spec[spec.wl >=360]
        spec = spec[spec.wl <= 780]
        x_true = spec.to_numpy().T[0]
        y_true = spec.to_numpy().T[1]
        spd_true = pd.Series(data=y_true, index=x_true)
        XYZ_true = sd_to_XYZ(SpectralDistribution(spd_true))
        u_true, v_true = calc_uv1976(XYZ_true)
        for n in range(min_nmbr_funcs, max_nmbr_funcs+1):
            subset_list = [((subset,)*n) for subset in self.mod_dict.keys()]
            try:
                for chunk, chunk_number in self._chunk_list(subset_list, 
                                                            self.chunk_size):
                    print(chunk_number)
                    r = {}
                    res = []
                    with cf.ProcessPoolExecutor(max_workers=self.max_workers) as ex:
                        futures = [ex.submit(partial(self.fitter,
                                                       x_true=x_true,
                                                       y_true=y_true,
                                                       u_true=u_true,
                                                       v_true=v_true), 
                                            i) for i in chunk]
                        for fut in cf.as_completed(futures):
                            try:
                                res.append(fut.result())
                                r = list(filter(lambda c: c !=None, res))
                            except Exception as e:
                                logger.exception(e)
                                pass
                    if os.path.isfile(self.output_path):
                        header = False
                        mode = 'a'                           
                    else:
                        header = True
                        mode= 'w'
                    try:  
                        pd.DataFrame(r).to_csv(self.output_path,
                                                 mode=mode,
                                                 header=header,
                                                 chunksize=100)
                    except Exception as e:
                        logger.exception(e)
                        pass        
            except (Exception, AttributeError, OverflowError) as e:
                logger.exception(e)
                pass
  
    
def fit_everything(input_path: str, output_path: str) -> None:
    """
    Method for fitting all given sample SPDs.
    :param input_path: Path to SPD files.
    :type input_path: str
    :param output_path: Path to stopre results.
    :type output_path: str
    :return: No return.
    :rtype: None

    """
    for root, dirs, files in os.walk(input_path, topdown=False):
        for file in sorted(files):
            print(file)
            if file.split('_')[0] in ['blue', 'green', 'red']:
                n_min = 1
                n_max = 4
            else:
                n_min = 2
                n_max = 6
            try:
                spec = pd.read_csv(os.path.join(root, file), 
                                   delimiter='\t')
                spec_fitter = FunctionFitter(output_path 
                                             + f'{file.split(".")[0]}.csv')
                spec_fitter.fit_spd(spec,
                                    max_nmbr_funcs=n_max, 
                                    min_nmbr_funcs=n_min, 
                                    file=file)
            except Exception as e:
                logger.exception(e)
                pass
         
            
def plot_test_spectra() -> None:
    """
    Plots the test spectra.
    :return: No return.
    :rtype: None

    """
    semi_list = ['blue_1', 'blue_2', 'blue_3', 'green_1', 
                 'red_1', 'red_2', 'red_3']
    pc_list_white = ['white_1', 'white_2', 'white_2700K', 
                     'white_3000K', 'white_4000K', 'white_6500K']
    pc_list_mono = ['lime_1', 'purple_1']
        
    fig, ax = plt.subplots(3,1, figsize=(3.5,5), sharex=True, sharey=True)
    linesstyles = ['solid', 'dashed', 'dashdot', 'solid', 'solid', 
                   'dashed', 'dashdot']
    colors = ['#1E90FF', '#1E90FF', '#1E90FF', '#6B8E23', '#FF6347', 
              '#FF6347', '#FF6347']
    for i, led in enumerate(semi_list):
        spec = pd.read_csv(rf'test_spectra\different\{led}.ISD', 
                           delimiter='\t')
        spec['pwr'] = spec['pwr']/spec['pwr'].max()
        ax[0].plot(spec.wl, 
                   spec.pwr, 
                   linestyle=linesstyles[i], 
                   color=colors[i], 
                   linewidth=1, 
                   label=led.split('_')[0]+' '+led.split('_')[1]
                   )

    linesstyles = ['solid', 'solid']
    colors = ['#90EE90', '#EE82EE']
    for i, led in enumerate(pc_list_mono):
        if led.split('_')[0] in ['purple', 'lime']:
            name = led.split('_')[0]
        else:
            name = led.split('_')[0]+' '+led.split('_')[1]          
        spec = pd.read_csv(rf'test_spectra\different\{led}.ISD', delimiter='\t')
        spec['pwr'] = spec['pwr']/spec['pwr'].max()
        ax[1].plot(spec.wl, 
                   spec.pwr, 
                   linestyle=linesstyles[i], 
                   color=colors[i], 
                   linewidth=1, 
                   label=name
                   )

    linesstyles = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed']
    colors = ['#B0C4DE', '#FFDAB9', '#F4A460', '#B0C4DE', '#FFDAB9', '#F4A460']
    for i, led in enumerate(pc_list_white):
        if led.split('_')[0] in ['purple', 'lime']:
            name = led.split('_')[0]
        else:
            name = led.split('_')[0]+' '+led.split('_')[1]            
        spec = pd.read_csv(rf'test_spectra\different\{led}.ISD', delimiter='\t')
        spec['pwr'] = spec['pwr']/spec['pwr'].max()
        ax[2].plot(spec.wl, 
                   spec.pwr, 
                   linestyle=linesstyles[i], 
                   color=colors[i], 
                   linewidth=1, 
                   label=name
                   )

    for i in range(3):
        ax[i].set_xlim([400, 780])
        ax[i].set_ylim(0,1)
        ax[i].tick_params(length=2, labelsize=6)  
        ax[i].legend(ncol=1, 
                    loc='upper right',
                    title_fontsize=6,
                    fontsize=6,)
    ax[1].set_ylabel(r'Relative Spectral Power, (a.u.)', fontsize=8)
    ax[2].set_xlabel(r'Wavelength $\lambda$, (nm)', fontsize=8)
    plt.savefig(r'pub\plots\test_spectra.pdf', bbox_inches='tight', pad_inches = 0, dpi=300)
    plt.show()   
  

def concat_results(path: str) -> pd.DataFrame:
    """
    Concats results.
    :param path: Path to fit results.
    :type path: str
    :return: Returns DataFrame of all results. 
    :rtype: pd.DataFrame

    """
    res = pd.DataFrame()
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files: 
            res = pd.concat([res, 
                             pd.read_csv(os.path.join(root, file), 
                                         index_col=0)])
    return res

def evaluate_data(path: str) -> pd.DataFrame:
    """
    Evaluates the results and returns a summary.
    :param path: Input path of fit results
    :type path: str
    :return: Returns summary of evaluated models as DataFrame.
    :rtype: pd.DataFrame

    """
    res = pd.DataFrame()
    for root, dirs, files in os.walk(path, topdown=False):
        fig1, ax1 = plt.subplots(len(files),1, sharex=True, figsize=(10,10))
        fig1.subplots_adjust(hspace=0.3)
        for i, file in enumerate(files): 
            print(f'evaluating {file}')
            results = pd.read_csv(os.path.join(root, file), index_col=0)
            results['led'] = file.split('.')[0]  
            res = pd.concat([res, results])
            styles = ['solid', 'dashed', 'dotted', 'dashdot', 
                      (0,(3,5,1,5,1,5)), (0, (3, 1, 1, 1, 1, 1))]
            n_funcs = []
            if 'mono' in path:
                n_min = 1
                n_max = 3
            else:
                n_min = 2
                n_max = 6        
            for n in range(n_min,n_max+1):
                df_one = results[results.number_functions == n]              
                print(f'---> top ten {file}, {n=}')
                print(df_one.sort_values(by='duv', 
                                         ascending=True)[['name', 
                                                          'power_dev', 
                                                          'duv']
                                                          ].head(10))
                print('\n')
                if n == 1:
                    gaus = 'Gaussian'
                else:
                    gaus = 'Gaussian+'*n
                    gaus = gaus[:-1]
                print(gaus)
                print(df_one[df_one.name == gaus][['name', 
                                                   'power_dev', 
                                                   'duv']])
                print('\n\n\n')
                input("Press Enter to continue...")               
                df_one = df_one[(df_one.duv < 4)
                                & (abs(df_one.power_dev) < 0.25)]
                n_funcs.append(df_one.shape[0])
                if not df_one.empty:
                    ax1[i].hist(df_one.duv, 
                                bins=200, 
                                histtype='step', 
                                density=True,
                                # range=(0,100),
                                color = 'b',
                                linestyle=styles[n-1],
                                linewidth=1,
                                cumulative=True,
                                label=f'n={n}')
                    ax1[i].set_xlim(0,4)
                    ax1[i].tick_params(labelsize=10) 
                    ax1[i].axvline(x=df_one.duv.min(), color='r', linewidth=0.3)
            name_led = (file.split('_')[0]).capitalize()+' '+file.split('_')[1].split('.')[0]
            print(n_funcs)
            if 'mono' in path:
                name = f'{name_led} - N(1)={n_funcs[0]}, \
                    N(2)={n_funcs[1]}, N(3)={n_funcs[2]}'  
            else:
                name = f'{name_led} - N(2)={n_funcs[0]}, N(3)={n_funcs[1]}, \
                    N(4)={n_funcs[2]}, N(5)={n_funcs[3]}, N(6)={n_funcs[4]}'  
            ax1[i].text(0, 1.08, name, fontsize=8, color='tab:gray') 
        ax1[len(files)-1].legend(ncol=int(n_max/2), 
                                 loc='lower right', 
                                 fontsize=10)
        ax1[len(files)-1].set_xlabel(r"Chromaticity Distance $\Delta_{u'v'}$, \
                                     ($\times 10^{-3}$)")
        ax1[3].set_ylabel(r"Cumulative percentage of model functions \
                          $N(n)\leq\Delta_{u'v'}$, (a.u.)")
        plt.show() 
        if 'mono' in path:
            save_path = r'pub\plots\mono_cumulative.pdf'
        else:
            save_path = r'pub\plots\pc_cumulative.pdf'
        fig1.savefig(save_path, bbox_inches='tight', pad_inches = 0, dpi=300)
    return res
        

def application_example_current(model: tuple=None, 
                                params: tuple=None) -> None:
    """
    Function for the publications application example
    :param model: Tuple of function models, defaults to None
    :type model: tuple, optional
    :param params: Tuple of model parameters, defaults to None
    :type params: tuple, optional
    :return: No return.
    :rtype: None

    """
    path = r'test_spectra\current'
    model = ('pearson_vii_split','pearson_vii_split')
    params = ('amplitude', 'center', 'sigma', 'sigmar', 's', 'sr')
    fig, ax = plt.subplots(1,1, figsize=(3.5,3.5))
    mod = model
    results = []
    
    for root, dirs, files in os.walk(path, topdown=False):
        colors = sns.color_palette("gist_yarg", len(files))
        alphas = [0.1, 0.3, 0.5, 0.8]
        for file, color, alpha in zip(files, colors, alphas):
            color = 'b'
            print(f'fitting model: {mod} to {file}')
            if mod == None: raise ValueError('incorrect model')
            if params == None: raise ValueError('incorrect params')
            spec = pd.read_csv(os.path.join(root, file), delimiter='\t')

            name = int(file.split('_')[1].split('m')[0])
            ax.plot(spec.wl, 
                    spec.pwr, 
                    linestyle="None", 
                    marker='o', 
                    markerfacecolor="None",
                    markeredgecolor=color,
                    markersize=3,
                    alpha=alpha, 
                    label=f'{name} mA - original'
                    )           
            spec_fitter = FunctionFitter(rf'results/current_{file}')
            res = spec_fitter.fit_model(spec=spec, model=mod)
            spec['wl'] = spec.wl.round(0)
            spec = spec.groupby('wl').mean().reset_index()
            spec = spec[spec.wl >=360]
            spec = spec[spec.wl <= 780]
            ax.plot(spec.wl, 
                    res.get('best_fit'), 
                    linestyle='solid', 
                    color=color, 
                    linewidth=1,
                    alpha=alpha,
                    label=f'{name} mA - fit'
                    )
           
            res['current'] = name
            results.append(res)
    ax.set_xlim([400, 500])
    ax.legend(ncol=1, 
              loc='best',
              title_fontsize=6,
              fontsize=6,)
    ax.set_ylim(0,0.008)
    ax.tick_params(length=2, labelsize=6)  
    ax.set_xlabel(r'Wavelength $\lambda$, (nm)', fontsize=8)
    ax.set_ylabel(r'Spectral Power Density $\Phi$, ($W\cdot m^{-2}nm^{-1}$)', fontsize=8)
    plt.savefig(r'pub\plots\current_spectra.pdf', bbox_inches='tight', pad_inches = 0, dpi=300)
    plt.show()         

    # df = pd.DataFrame(results).drop(['best_fit'], axis=1)
    # df = df.sort_values(by='current')
    # df = df.reset_index(drop=True)
    # for i, _ in enumerate(mod):
    #     for param in params:
    #         p = f'm{i}_{param}'  
    #         df[p] = np.nan
    
    # for idx in df.index:
    #     for i, _ in enumerate(mod):
    #         for param in params:
    #             p = f'm{i}_{param}'
    #             df.loc[idx,p] = df.model.values[idx][p].value
    
    # df = df.drop('model', axis=1)
 
    # p0 = ['m0_amplitude', 'm0_center', 'm0_sigma', 'm0_sigmar', 'm0_s', 'm0_sr']
    # p1 = ['m1_amplitude', 'm1_center', 'm1_sigma', 'm1_sigmar', 'm1_s', 'm1_sr']
    # for j in [0,1]:
    #     df.loc[j,p0], df.loc[j,p1] = df.loc[j,p1].values, df.loc[j,p0].values
    
    # for col in df.columns[8:]:
    #     df[col] = df[col] / df.loc[0,col]
    
    # fig, ax = plt.subplots(6, figsize=(10,14), sharex=True)
    
    # x = df.current
    
    # cols = ['amplitude', 'center', 'sigma', 'sigmar', 's', 'sr']
    # colors = sns.color_palette("gist_yarg", len(cols))
    # markers = ['x', 'o']
    # linestyles = ['solid', 'dotted']
    # pretty_legend = {'amplitude' : 'A',
    #                  'center' : r'$\lambda_{p,}$',
    #                  'sigma' : r'$\sigma_{1,}$',
    #                  'sigmar' : r'$\sigma_{2,}$',
    #                  's' : r'$S_{1,}$',
    #                  'sr' : r'$S_{2,}$',                 
    #                  }
    
    # # print(df[df.columns])
    # correlations = df[df.columns[7:]].corr()['current']
    # # print(correlations)
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True) 
    # formatter.set_powerlimits((0,0)) 
    # for model, marker, linestyle in zip(range(2), markers, linestyles):
    #     for axes, col in enumerate(cols):
    #         col_name = f'm{model}_{col}'
    #         y = df[col_name]
    #         ax[axes].plot(x,y,
    #                       linestyle="None", 
    #                       marker=marker, 
    #                       markerfacecolor="None",
    #                       markeredgecolor='k',
    #                       markersize=7,
    #                       label=f'M{model+1} - $R^{2}$: {round(correlations[col_name],4)}'
    #                       )

    #         coef = np.polyfit(x, y, 1)
    #         poly1d_fn = np.poly1d(coef)  
    #         ax[axes].plot(x,poly1d_fn(x),
    #                       linestyle=linestyle,
    #                       color='k',
    #                       linewidth=1
    #                       )
    #         if axes == 0:
    #             loc = 'lower right'
    #         elif axes == 1:
    #             loc = 'upper right'
    #         else:
    #             loc = 'center right'
    #         # ax[axes].set_xlim([0, 11])
    #         ax[axes].set_ylabel(f'{pretty_legend.get(col)}/{pretty_legend.get(col)}$_{0}$', fontsize=14)
    # for axe in ax:
    #     axe.yaxis.set_major_formatter(OOMFormatter(0, "%0.3f"))
    #     axe.ticklabel_format(axis='y', style='sci', scilimits=(0,0))         
    # ax[0].legend(ncol=2, loc='lower right')
    # ax[1].legend(ncol=2, loc='upper right')
    # ax[2].legend(ncol=2, loc='center right')
    # ax[3].legend(ncol=2, loc='lower right')
    # ax[4].legend(ncol=2, loc='center right')
    # ax[5].legend(ncol=2, loc='lower right')
    # ax[5].set_xlabel(r'Normalized forward current $I_{F}/I_{F,0}$, (a.u.)', fontsize=14)
    # # ax[0].set_ylim([0, 10])
    # # ax[1].set_ylim([0.99, 1.002])
    # # ax[2].set_ylim([0.99, 1.15])
    # # ax[3].set_ylim([0.99, 1.1])
    # # ax[4].set_ylim([0.9, 2.5])
    # # ax[5].set_ylim([0.3, 1.2])
    # plt.savefig(r'pub\plots\app_ex_correlations.pdf', bbox_inches='tight', pad_inches = 0, dpi=300)
    # plt.show()


def plot_cie_1976() -> None:
    """
    Plot the CIE 1976 u'v' chromaticity coordinates deviations for fig 2.
    :return: No return.
    :rtype: None

    """
    def wl_to_u(wl):
        XYZ = wavelength_to_XYZ(wl)   
        u = 4*XYZ[0] / (XYZ[0] + 15*XYZ[1] + 3*XYZ[2])
        return u


    def wl_to_v(wl):
        XYZ = wavelength_to_XYZ(wl)
        v = 9*XYZ[1] / (XYZ[0] + 15*XYZ[1] + 3*XYZ[2])
        return v
    WL_MIN = 380
    WL_MAX = 780
    df_locus = pd.DataFrame(np.arange(WL_MIN,WL_MAX+1,1), columns=['wl'])
    df_locus['u'] = df_locus['wl'].apply(wl_to_u)
    df_locus['v'] = df_locus['wl'].apply(wl_to_v)

    
    spec_dir = r'test_spectra/different/'
    fit_data_dir = r'results/'
    def get_true_u_v(file: str) -> tuple:
        spec = pd.read_csv(file, delimiter='\t')
        spec['wl'] = spec.wl.round(0)
        spec = spec.groupby('wl').mean().reset_index()
        spec = spec[spec.wl >=360]
        spec = spec[spec.wl <= 780]
        x_true = spec.to_numpy().T[0]
        y_true = spec.to_numpy().T[1]
        spd_true = pd.Series(data=y_true, index=x_true)
        XYZ_true = sd_to_XYZ(SpectralDistribution(spd_true))
        return calc_uv1976(XYZ_true)
    
    fig, ax = plt.subplots(5,3,figsize=(3.5, 5.5), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    markers = ['X', 'P', 'v', 'o', 's', 'p']
    row = 0
    col = 0
    ax[4][1].set_xlabel(r"Chromaticity Coordinate $u'$, ($\times 10^{-3}$)", fontsize=8)
    ax[2][0].set_ylabel(r"Chromaticity Coordinate $v'$, ($\times 10^{-3}$)", fontsize=8)
    
    for root, dirs, files in os.walk(fit_data_dir, topdown=False):
        for n, file in enumerate(files):
            print(file)
            spec_file = os.path.join(spec_dir, file.replace('csv', 'ISD'))
            u_true, v_true = get_true_u_v(spec_file)
            

            fit_data = pd.read_csv(os.path.join(root, file), 
                                    index_col=0
                                    ).reset_index(drop=True)
            
            fit_data['u_fit'], fit_data['v_fit'] = np.nan, np.nan
            fit_data['du'], fit_data['dv'] = np.nan, np.nan
            fit_data = fit_data[fit_data.number_functions.notna()]
            fit_data = fit_data[fit_data.r2.notna()]
            fit_data['number_functions'] = fit_data.number_functions.astype('int')
            for idx in fit_data.index:
                model_name = fit_data.loc[idx, 'name']
                try:
                    for i, m in enumerate(model_name.split('+')):
                        
                        val = FunctionFitter('').mod_dict.get(m)
                        if val[0] == 'internal':
                            mod = val[1](nan_policy='propagate')                
                        else:
                            mod = Model(val[1], nan_policy='propagate')
                        mod.prefix = f'm{i}_'
                        if i == 0: 
                            model = mod
                        else:
                            model = model + mod
                    params_str = fit_data.loc[idx,'model'][12:-2]
                    rep = {'(': '{',
                            '<Parameter ': '{',
                            '>)': '}}'}
                    for k,v in rep.items():
                        params_str = params_str.replace(k,v)                    
                    params = Parameters()
                    for s in params_str.split(', {'):                
                        if 'value' in s and not 'fixed' in s:
                            par = s.split(' +')[0].split(',')[0][1:-1]
                            val = float(s.split(' +')[0].split(',')[1].split('=')[1])
                            params.add(par, value=val, vary=False)
                    results = model.eval(params, x=np.arange(360,781,1))
                    spd_fit = pd.Series(data=results, index=np.arange(360,781,1))
                    XYZ_fit = sd_to_XYZ(SpectralDistribution(spd_fit))
                    u_fit, v_fit = calc_uv1976(XYZ_fit)
                    fit_data.loc[idx, 'u_fit'] = u_fit
                    fit_data.loc[idx, 'v_fit'] = v_fit
                    fit_data.loc[idx, 'du'] = 1000*(u_fit - u_true)
                    fit_data.loc[idx, 'dv'] = 1000*(v_fit - v_true)
                    

                except OverflowError:
                    pass
            fit_data = fit_data[fit_data.u_fit.notna()]
            fit_data = fit_data[fit_data.v_fit.notna()]
            name = file.split('.')[0].split('_')[0]+' '+file.split('.')[0].split('_')[1]
            if 'purple' in name:
                name = 'purple'
            elif 'lime' in name:
                name = 'lime'    
            else:
                pass
        
            ax[row][col].tick_params(length=2, labelsize=6)  
            ax[row][col].text(-23, -23, name, color='k', alpha=0.7, fontsize=6, ha='left')
            ax[row][col].plot(1000*(df_locus.u - u_true),
                              1000*(df_locus.v - v_true),
                              color='k',
                              linestyle='-',
                              alpha=0.5,
                              linewidth=0.5)
            
            for n_func in range(1,7):    
                marker = markers[n_func-1]
                du = fit_data[fit_data.number_functions==n_func]['du']
                dv = fit_data[fit_data.number_functions==n_func]['dv']
                ax[row][col].scatter(du, 
                                    dv, 
                                    linestyle="None", 
                                    marker=marker, 
                                    facecolor='b',
                                    edgecolor="None",
                                    s=7,
                                    alpha=0.4, 
                                    label=n_func
                                    )
            ax[row][col].plot(0,
                              0, 
                              color='r',
                              markersize=2,
                              marker='x')
            for radius in [4,7]:
                circle = plt.Circle((0, 0), 
                                    radius, 
                                    color='r', 
                                    alpha=0.2,
                                    linestyle='-',
                                    fill=False
                                )
                ax[row][col].add_patch(circle)

            ax[row][col].xaxis.set_ticks(range(-50, 51, 25))
            ax[row][col].yaxis.set_ticks(range(-50, 51, 25))
            ax[row][col].set_xlim([-30, 30])
            ax[row][col].set_ylim([-30, 30])

            col = col + 1
            if col > 2:
                col = 0
                row = row + 1
    ax[4][1].legend(loc='lower center',
                    title='#Functions', 
                    title_fontsize=6,
                    fontsize=6,
                    ncol=6, 
                    bbox_to_anchor=(0.5, -1.0)
                    )
    plt.savefig(r'pub\plots\color_coordinates.pdf', bbox_inches='tight', pad_inches = 0.1, dpi=300)
    plt.show()
    
    





