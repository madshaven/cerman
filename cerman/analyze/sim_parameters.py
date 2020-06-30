#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Define configurations of simulation parameters:
    info, get-function, format, abbreviation, label, unit, scale, and symbol.

    Define which parameters that are dependent on others.
    That is, a variation in one gives a variation in the other.

    Define `ParamRepr` telling how to format e.g. value strings and labels.

    Define `get_key_abbr_info` giving info on all parameters/configurations.
'''

# General imports
import logging
from collections import OrderedDict

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# define how simulation parameters should be represented
configurations = OrderedDict()

# todo: consider a closer link between this file and `simulation_input`.

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       EXPERIMENTAL PROPERTIES                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

configurations['gap_size'] = {
    'info' : 'The distance between the needle electrode and the ground electrode.',
    'get'  : lambda sparams: sparams['gap_size'],
    'fmt'  : '4.1f',
    'abbr' : 'gs',
    'label': r'Gap size',
    'unit' : r'mm',
    'scale': 1e3,
    'symbol': r'$d_g$',
    }
configurations['needle_radius'] = {
    'info' : 'The tip curvature radius of the needle.',
    'get'  : lambda sparams: sparams['needle_radius'],
    'fmt'  : '4.1f',
    'abbr' : 'rp',
    'label': r'Needle radius',
    'unit' : '\u00B5m',
    'scale': 1e6,
    'symbol': r'$r_n$',
    }
configurations['needle_voltage'] = {
    'info' : 'The initial voltage of the needle.',
    'get'  : lambda sparams: sparams['needle_voltage'],
    'fmt'  : '5.1f',
    'abbr' : 'v',
    'label': r'Needle potential',
    'unit' : r'kV',
    'scale': 1e-3,
    'symbol': r'$V_n$',
    }
configurations['liquid_name'] = {
    'info' : 'The name of the base insulating liquid.',
    'get'  : lambda sparams: sparams['liquid_name'],
    'fmt'  : 's',
    'abbr' : 'liq',
    'label': r'Liquid name',
    'unit' : r'',
    'scale': 1,
    'symbol': None,
    }
configurations['additive_name'] = {
    'info' : 'The name of the additive to insulating liquid.',
    'get'  : lambda sparams: sparams['additive_name'],
    'fmt'  : 's',
    'abbr' : 'add',
    'label': r'Additive name',
    'unit' : r'',
    'scale': 1,
    'symbol': None,
    }
configurations['additive_cw'] = {
    'info' : 'The additive concentration in weight percent.',
    'get'  : lambda sparams: sparams['additive_cw'],
    'fmt'  : '#0.4g',
    'abbr' : 'cw',
    'label': r'Additive concentration',
    'unit' : r'wt %',
    'scale': 1,
    'symbol': r'$c_w$',
    }
configurations['Q_crit'] = {
    'info' : 'The Meek constant, defining when an avalanche is critical.',
    'get'  : lambda sparams: sparams['Q_crit'],
    'fmt'  : '4.1f',
    'abbr' : 'qc',
    'label': r'Critical avalanche size',
    'unit' : r'',
    'scale': 1,
    'symbol': r'$Q_c$',
    }
configurations['alphakind'] = {
    'info' : 'The method for calculating the electron multiplication.',
    'get'  : lambda sparams: sparams['alphakind'],
    'fmt'  : 's',
    'abbr' : 'ak',
    'label': r'Alpha model',
    'unit' : r'',
    'scale': 1,
    'symbol': None,
    }
configurations['additive_cn'] = {
    'info' : 'The additive concentration in mole fraction.',
    'get'  : lambda sparams: sparams['additive_cn'],
    'fmt'  : '4.3f',
    'abbr' : 'cn',
    'label': r'Additive mole fraction',
    'unit' : r'%',
    'scale': 1e2,
    'symbol': r'$c_n$',
    }
configurations['seeds_cn'] = {
    'info' : 'The seed anion concentration in number density.',
    'get'  : lambda sparams: sparams['seeds_cn'],
    'fmt'  : '0.2e',
    'abbr' : 'sc',
    'label': r'Seed concentration',
    'unit' : r'm$^{-3}$',
    'scale': 1,
    'symbol': r'$n_{ion}$',
    }
configurations['liquid_Ec_ava'] = {
    'info' : 'The electrical field threshold for electron multiplication.',
    'get'  : lambda sparams: sparams['liquid_Ec_ava'],
    'fmt'  : '5.1f',
    'abbr' : 'ec',
    'label': r'Electron multiplication threshold field',
    'unit' : r'MV/m',
    'scale': 1e-6,
    'symbol': r'$E_a$',
    }

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       STREAMER PHYSICS                                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

configurations['streamer_head_radius'] = {
    'info' : 'The tip curvature radius of a streamer head.',
    'get'  : lambda sparams: sparams['streamer_head_radius'],
    'fmt'  : '4.1f',
    'abbr' : 'srp',
    'label': r'Streamer head radius',
    'unit' : '\u00B5m',
    'scale': 1e6,
    'symbol': r'$r_s$',
    }
configurations['streamer_U_grad'] = {
    'info' : 'The minimum electric field in the streamer channel.',
    'get'  : lambda sparams: sparams['streamer_U_grad'],
    'fmt'  : '4.1f',
    'abbr' : 'sug',
    'label': r'Streamer electric field',
    'unit' : r'kV/mm',
    'scale': 1e-6,
    'symbol': r'$E_s$',
    }
configurations['streamer_d_merge'] = {
    'info' : 'Streamer head merge distance.',
    'get'  : lambda sparams: sparams['streamer_d_merge'],
    'fmt'  : '4.1f',
    'abbr' : 'dm',
    'label': r'Streamer merge distance',
    'unit' : '\u00B5m',
    'scale': 1e6,
    'symbol': r'$d_m$',
    }

configurations['streamer_scale_tvl'] = {
    'info' : 'Any streamer head scaled below this threshold is removed.',
    'get'  : lambda sparams: sparams['streamer_scale_tvl'],
    'fmt'  : '4.1f',
    'abbr' : 'sst',
    'label': r'Streamer scale threshold',
    'unit' : r'%',
    'scale': 1e2,
    'symbol': r'$k_c$',
    }
configurations['streamer_photo_enabled'] = {
    'info' : 'Enables fast forward propagation by photoionization.',
    'get'  : lambda sparams: 'Enabled' if sparams['streamer_photo_enabled'] else 'Disabled',
    'fmt'  : 's',
    'abbr' : 'spe',
    'label': r'Photoionization',
    'unit' : r'',
    'scale': 1,
    'symbol': None,
    }
configurations['streamer_photo_efield'] = {
    'info' : 'The electric field required at a streamer head for photoionization.',
    'get'  : lambda sparams: sparams['streamer_photo_efield'],
    'fmt'  : '4.1f',
    'abbr' : 'e4',
    'label': r'Fast streamer threshold field',
    'unit' : r'GV/m',
    'scale': 1e-9,
    'symbol': r'$E_p$',
    }
configurations['streamer_photo_speed'] = {
    'info' : 'The speed added to when photoionization is active.',
    'get'  : lambda sparams: sparams['streamer_photo_speed'],
    'fmt'  : '4.1f',
    'abbr' : 'v4',
    'label': r'Fast streamer propagation speed',
    'unit' : r'km/s',
    'scale': 1e-3,
    'symbol': r'$v_{pi}$',
    }
configurations['streamer_repulsion_mobility'] = {
    'info' : 'The mobility of a streamer head.',
    'get'  : lambda sparams: sparams['streamer_repulsion_mobility'],
    'fmt'  : '#0.4g',
    'abbr' : 'srm',
    'label': r'Streamer repulsion mobility',
    'unit' : r'mm$^2$/(Vs)',
    'scale': 1e6,
    'symbol': r'$\mu_{s}$',
    }
configurations['streamer_new_head_offset'] = {
    'info' : 'The position of a new streamer head, relative to the avalanche position, in the z-direction.',
    'get'  : lambda sparams: sparams['streamer_new_head_offset'],
    'fmt'  : '#0.4g',
    'abbr' : 'snho',
    'label': r'New head offset',
    'unit' : '\u00B5m',
    'scale': 1e6,
    'symbol': r'',
    }

configurations['rc_tau0'] = {
    'info' : 'The base RC time constant.',
    'get'  : lambda sparams: sparams['rc_tau0'],
    'fmt'  : '#0.2e',
    'abbr' : 'rct',
    'label': r'RC relaxation time',
    'unit' : r's',
    'scale': 1,
    'symbol': r'$\tau_0$',
    }
configurations['rc_resistance'] = {
    'info' : 'The model used for resistance in the RC model.',
    'get'  : lambda sparams: sparams['rc_resistance'],
    'fmt'  : 's',
    'abbr' : 'rcr',
    'label': r'RC resistance model',
    'unit' : r'',
    'scale': 1,
    'symbol': None,
    }
configurations['rc_capacitance'] = {
    'info' : 'The model used for capacitance in the RC model.',
    'get'  : lambda sparams: sparams['rc_capacitance'],
    'fmt'  : 's',
    'abbr' : 'rcc',
    'label': r'RC capacitance model',
    'unit' : r'',
    'scale': 1,
    'symbol': None,
    }
configurations['rc_breakdown'] = {
    'info' : 'The electric field threshold for breakdown in the RC model.',
    'get'  : lambda sparams: sparams['rc_breakdown'],
    'fmt'  : '4.1f',
    'abbr' : 'rcb',
    'label': r'RC breakdown',
    'unit' : r'kV/mm',
    'scale': 1e-6,
    'symbol': r'$E_{b}$',
    }
configurations['rc_breakdown_factor'] = {
    'info' : 'The factor modifying the time constant upon breakdown in the RC model.',
    'get'  : lambda sparams: sparams['rc_breakdown_factor'],
    'fmt'  : '.1e',
    'abbr' : 'rcbf',
    'label': r'RC breakdown factor',
    'unit' : r'',
    'scale': 1,
    'symbol': None,
    }
configurations['rc_onsager'] = {
    'info' : 'Enabling the increased conductivity in a streamer channel according to Onsager dissociation.',
    'get'  : lambda sparams: 'Enabled' if sparams['rc_onsager'] else 'Disabled',
    'fmt'  : 's',
    'abbr' : 'rco',
    'label': r'RC Onsager model',
    'unit' : r'',
    'scale': 1,
    'symbol': None,
    }
configurations['rc_potential_merged'] = {
    'info' : 'The model used to determine the potential of merged heads.',
    'get'  : lambda sparams: sparams['rc_potential_merged'],
    'fmt'  : 's',
    'abbr' : 'rcpm',
    'label': r'RC Potential Merged',
    'unit' : r'',
    'scale': 1,
    'symbol': None,
    }
configurations['rc_potential_branched'] = {
    'info' : 'The model used to determine the potential of branched heads.',
    'get'  : lambda sparams: sparams['rc_potential_branched'],
    'fmt'  : 's',
    'abbr' : 'rcpb',
    'label': r'RC Potential Branched',
    'unit' : r'',
    'scale': 1,
    'symbol': None,
    }

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       SIMULATION PROPERTIES                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

configurations['time_step'] = {
    'info' : 'The minimum time step of the simulation.',
    'get'  : lambda sparams: sparams['time_step'],
    'fmt'  : '3.0f',
    'abbr' : 'ts',
    'label': r'Time step',
    'unit' : r'ps',
    'scale': 1e12,
    'symbol': r'$\Delta t$',
    }
configurations['micro_step_no'] = {
    'info' : 'The maximum amount of time steps performed on electrons alone in a single loop of the simulation.',
    'get'  : lambda sparams: sparams['micro_step_no'],
    'fmt'  : '.0f',
    'abbr' : 'msn',
    'label': r'Micro step number',
    'unit' : r'',
    'scale': 1,
    'symbol': r'$N_s$',
    }
configurations['random_seed'] = {
    'info' : 'The number used to initiate the random number generator.',
    'get'  : lambda sparams: sparams['random_seed'],
    'fmt'  : '.0f',
    'abbr' : 'rs',
    'label': r'Random seed',
    'unit' : r'',
    'scale': 1,
    'symbol': r'$N_{random}$',
    }
configurations['roi_replacement'] = {
    'info' : 'The method used to replace seeds out of ROI.',
    'get'  : lambda sparams: sparams['roi_replacement'],
    'fmt'  : 's',
    'abbr' : 'roi_rep',
    'label': r'ROI Replacement',
    'unit' : r'',
    'scale': 1,
    'symbol': None,
    }

configurations['stop_iteration'] = {
    'info' : 'The maximum number of iterations before termination.',
    'get'  : lambda sparams: sparams['stop_iteration'],
    'fmt'  : '.0f',
    'abbr' : 'st_i',
    'label': r'Stop iteration',
    'unit' : r'',
    'scale': 1,
    'symbol': r'$N_{stop}$',
    }
configurations['stop_z_min'] = {
    'info' : 'The minimum allowable z-value before termination.',
    'get'  : lambda sparams: sparams['stop_z_min'],
    'fmt'  : '5.1f',
    'abbr' : 'st_z',
    'label': r'Stop z-min',
    'unit' : '\u00B5m',
    'scale': 1e6,
    'symbol': r'$z_{min}$',
    }
configurations['stop_heads_max'] = {
    'info' : 'The maximum allowable number of streamer heads.',
    'get'  : lambda sparams: sparams['stop_heads_max'],
    'fmt'  : '.0f',
    'abbr' : 'st_t',
    'label': r'Stop head max',
    'unit' : r'',
    'scale': 1,
    'symbol': r'$N_{head max}$',
    }
configurations['stop_seeds_no_electron'] = {
    'info' : 'The minimum number of electron seeds required for continuation.',
    'get'  : lambda sparams: sparams['stop_seeds_no_electron'],
    'fmt'  : '.0f',
    'abbr' : 'st_e',
    'label': r'Stop electron number',
    'unit' : r'',
    'scale': 1,
    'symbol': r'$N_{electron min}$',
    }

configurations['stop_sim_time'] = {
    'info' : 'The maximum simulation time before termination.',
    'get'  : lambda sparams: sparams['stop_sim_time'],
    'fmt'  : '#0.4g',
    'abbr' : 'st_st',
    'label': r'Stop simulation time',
    'unit' : r'ns',
    'scale': 1e9,
    'symbol': r'$t_{sim max}$',
    }
configurations['stop_cpu_time'] = {
    'info' : 'The maximum CPU time used before termination.',
    'get'  : lambda sparams: sparams['stop_cpu_time'],
    'fmt'  : '#0.4g',
    'abbr' : 'st_ct',
    'label': r'Stop CPU time',
    'unit' : r'Ms',
    'scale': 1e-6,
    'symbol': r'$t_{cpu max}$',
    }
configurations['stop_time_since_avalanche'] = {
    'info' : 'The maximum time between avalanches before termination.',
    'get'  : lambda sparams: sparams['stop_time_since_avalanche'],
    'fmt'  : '#0.4g',
    'abbr' : 'st_tsa',
    'label': r'Stop time since avalanche',
    'unit' : r'ns',
    'scale': 1e9,
    'symbol': r'$t_{no avalanche max}$',
    }
configurations['stop_speed_avg'] = {
    'info' : 'The minimum speed required for continuation.',
    'get'  : lambda sparams: sparams['stop_speed_avg'],
    'fmt'  : '#0.4g',
    'abbr' : 'st_sa',
    'label': r'Stop speed average',
    'unit' : r'm/s',
    'scale': 1,
    'symbol': r'$v_{speed min}$',
    }


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       LIQUID                                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

configurations['liquid_IP'] = {
    'info' : 'The ionization potential of the insulating liquid.',
    'get'  : lambda sparams: sparams['liquid_IP'],
    'fmt'  : '#0.4g',
    'abbr' : 'ip',
    'label': r'Ionization potential',
    'unit' : r'eV',
    'scale': 1,
    'symbol': r'$I_b$',
    }
configurations['liquid_e1'] = {
    'info' : 'The energy of the first excited state of the liquid.',
    'get'  : lambda sparams: sparams['liquid_e1'],
    'fmt'  : '#0.4g',
    'abbr' : 'e1',
    'label': r'First excited state',
    'unit' : r'eV',
    'scale': 1,
    'symbol': r'$\varepsilon_1$',
    }
configurations['liquid_beta_b'] = {
    'info' : 'The FDIP reduction parameter of the liquid.',
    'get'  : lambda sparams: sparams['liquid_beta_b'],
    'fmt'  : '#0.4g',
    'abbr' : 'bb',
    'label': r'IP reduction parameter',
    'unit' : r'eV',
    'scale': 1,
    'symbol': r'$\beta_b$',
    }
configurations['liquid_mu_e'] = {
    'info' : 'The electron mobility in the insulating liquid.',
    'get'  : lambda sparams: sparams['liquid_mu_e'],
    'fmt'  : '#0.4g',
    'abbr' : 'mue',
    'label': r'Electron mobility',
    'unit' : r'mm$^2$/(Vs)',
    'scale': 1e6,
    'symbol': r'$\mu_e$',
    }
configurations['liquid_mu_ion'] = {
    'info' : 'The ion mobility in the insulating liquid.',
    'get'  : lambda sparams: sparams['liquid_mu_ion'],
    'fmt'  : '#0.4g',
    'abbr' : 'mui',
    'label': r'Ion mobility',
    'unit' : r'mm$^2$/(Vs)',
    'scale': 1e6,
    'symbol': r'$\mu_{ion}$',
    }
configurations['liquid_sigma'] = {
    'info' : 'The low-field conductivity of the insulating liquid.',
    'get'  : lambda sparams: sparams['liquid_sigma'],
    'fmt'  : '#0.4g',
    'abbr' : 'con',
    'label': r'Conductivity',
    'unit' : r'S/pm',
    'scale': 1e12,
    'symbol': r'$\sigma$',
    }
configurations['liquid_Ed_ion'] = {
    'info' : 'The electric field threshold for electron detachment.',
    'get'  : lambda sparams: sparams['liquid_Ed_ion'],
    'fmt'  : '#0.4g',
    'abbr' : 'ed',
    'label': r'Electron detachment threshold',
    'unit' : r'MV/m',
    'scale': 1e-6,
    'symbol': r'$E_d$',
    }
configurations['liquid_alphamax'] = {
    'info' : 'The electron avalanche saturation constant.',
    'get'  : lambda sparams: sparams['liquid_alphamax'],
    'fmt'  : '#0.4g',
    'abbr' : 'am',
    'label': r'Electron avalanche saturation',
    'unit' : '1/\u00B5m',
    'scale': 1e-6,
    'symbol': r'$\alpha_m$',
    }
configurations['liquid_Ealpha'] = {
    'info' : 'The inelastic scattering constant in electron avalanches.',
    'get'  : lambda sparams: sparams['liquid_Ealpha'],
    'fmt'  : '1.2f',
    'abbr' : 'Ea',
    'label': r'Inelastic scattering constant',
    'unit' : r'GV/m',
    'scale': 1e-9,
    'symbol': r'$E_{\alpha}$',
    }
configurations['liquid_k1'] = {
    'info' : 'The additive proportionality factor in electron avalanche growth.',
    'get'  : lambda sparams: sparams['liquid_k1'],
    'fmt'  : '#0.4g',
    'abbr' : 'k1',
    'label': r'Additive proportionality factor',
    'unit' : r'1',
    'scale': 1,
    'symbol': r'$k_1$',
    }
configurations['liquid_alphacrit'] = {
    'info' : 'The electric field threshold for electron multiplication.',
    'get'  : lambda sparams: sparams['liquid_alphacrit'],
    'fmt'  : '#0.4g',
    'abbr' : 'ac',
    'label': r'Alpha threshold for electron multiplication',
    'unit' : '1/\u00B5m',
    'scale': 1e-6,
    'symbol': r'$\alpha_c$',
    }

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       ADDITIVE                                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

configurations['additive_IP'] = {
    'info' : 'The ionization potential of the additive.',
    'get'  : lambda sparams: sparams['additive_IP'],
    'fmt'  : '#0.4g',
    'abbr' : 'aip',
    'label': r'Ionization potential, additive',
    'unit' : r'eV',
    'scale': 1,
    'symbol': r'$I_a$',
    }
configurations['additive_beta_b'] = {
    'info' : 'The FDIP reduction parameter of the additive.',
    'get'  : lambda sparams: sparams['additive_beta_b'],
    'fmt'  : '#0.4g',
    'abbr' : 'abb',
    'label': r'IP reduction parameter, additive',
    'unit' : r'eV',
    'scale': 1,
    'symbol': r'$\beta_a$',
    }


# map abbreviations and keys to keys
map_abbr = {v['abbr']: k for k, v in configurations.items()}
assert len(map_abbr) == len(configurations), 'error, abbr vs config'
map_keys = {k: k for k, v in configurations.items()}
map_keys = dict(map_abbr, **map_keys)

# define keys that are correlated
# (they should not be plotted together)
dependencies = {key: set() for key in configurations}   # initiate
dependencies = dict(dependencies,                       # add derived
    additive_cn = {
        'additive_cw',
        # 'additive_density',
        # 'liquid_density',
        # 'liquid_mass',
        # 'additive_mass',
        },
    seeds_cn = {
        'liquid_sigma',
        'liquid_mu_ion',
        },
    liquid_Ec_ava = {
        'liquid_alphamax',
        'liquid_Ealpha',
        'liquid_k1',
        'liquid_IP',
        'additive_IP',
        'additive_cn',
        },
    streamer_photo_efield = {
        'streamer_photo_enabled',
        'liquid_IP',
        'liquid_e1',
        'liquid_beta_b',
        # 'liquid_R_perm',
        },
    )
# add the reversed dependencies as well
for k0, dep_set in dependencies.items():
    dependencies[k0].add(k0)
    for k1 in dep_set:
        dependencies[k1].add(k0)


class ParamRepr(object):
    ''' Define how to represent a simulation parameter. '''
    # idea: add defaults for axis_label, legend_label

    def __init__(self, key, abbr, label, info, fmt, unit='', scale=1, symbol=None):
        '''Use `configurations` to represent a parameter as a string'''
        self.key   = key
        self.abbr  = abbr
        self.label = label
        self.info  = info
        self.fmt   = fmt
        self.unit  = unit
        self.scale = scale
        self.symbol = symbol

    def _get_val(self, sparams):
        ''' Return raw value. '''
        return sparams[self.key]

    def get_val(self, sparams=None, val=None, scale=None):
        ''' Return scaled value. '''
        scale = self.scale if (scale is None) else scale
        val = self._get_val(sparams) if (sparams is not None) else val
        val = val * scale if (val is not None) else val
        return val

    def get_val_str(self, sparams=None, val=None, fmt=None, scale=None,
                    unit=None, brace=False, symbol=True):
        ''' Return scaled value as formatted string. E.g. for legend.

        Parameters
        ----------

        sparams :   dict
                    simulation parameter data
        val :       float or bool
                    defaults to the value in `data`
        fmt :       str
                    optional, specify how to format the string
        scale :     float
                    optional, specify the scaling of the value
        unit :      str
                    optional, specify the unit of the value
        brace :     bool
                    optional, add braces around the unit
        symbol :    str or bool
                    optional, add the (default) symbol of the value

        returns
        -------
        out :       str
                    scaled value, possibly with unit and symbol
        '''

        # get val
        if (sparams is not None):
            val = self._get_val(sparams)
        # fix special cases
        val = 'Enabled' if (val is True) else val
        val = 'Disabled' if (val is False) else val
        if val is None:
            return str(val)

        # chose input or defaults
        scale = self.scale if scale is None else scale
        unit = self.unit if unit is None else unit
        fmt = self.fmt if fmt is None else fmt

        # format output
        out = f'{val * scale:{fmt}}'

        if brace and unit:
            unit = '[' + unit + ']'
        if unit:
            out = out + ' ' + unit
        symbol = self.symbol if (symbol is True) else symbol
        if symbol:  # false if symbol is True and self.symbol is None
            out = symbol + ' $=$ ' + out

        return out

    def get_label(self, unit=None, brace=True):
        ''' Return label, e.g. for axis. '''
        unit = self.unit if unit is None else unit
        out = self.label
        if brace and unit:
            unit = '[' + unit + ']'
        if unit:
            out = out + ' ' + unit
        return out


# add all configurations as ParamRepr's
paramr = OrderedDict()
for key, config in configurations.items():
    paramr[key] = ParamRepr(
        key   = key,
        abbr  = config['abbr'],
        label = config['label'],
        info  = config['info'],
        fmt   = config['fmt'],
        unit  = config['unit'],
        scale = config['scale'],
        symbol = config['symbol'],
        )


# verify that no abbreviation is used twice
for key, pr in paramr.items():
    if pr.abbr in paramr:
        msg = f'Implementation error, {pr.abbr} defined twice.'
        raise SystemExit(msg)


def get_key_abbr_info(keys=None):
    # return a string containing key, abbreviation and info
    # all available keys are used as default

    logger.debug('Got keys:' + str(keys))
    # use keys if given, else use all keys in dict, in correct order
    if keys is None:
        keys = list(paramr)

    # map_keys contains both keys and abbreviations
    invalid_keys = [k for k in keys if (k not in map_keys)]
    keys = [k for k in keys if (k in map_keys)]
    keys = [map_keys[k] for k in keys]  # change abbreviations to keys

    logger.debug('Using keys:' + str(keys))
    if invalid_keys:
        msg = 'The following keys are invalid:' + str(invalid_keys)
        logger.warning(msg)

    # get the longest needed string length for key and abbreviation
    l_max = max(len(key) + len(paramr[key].abbr) for key in keys)

    msg = '{0:<{1}} ({2:}) - {3:}\n'
    out = ''
    for key in keys:
        abbr = paramr[key].abbr
        info = paramr[key].info
        lk = l_max - len(abbr)
        out += msg.format(key, lk, abbr, info)

    out = out[:-1]  # remove final new line
    return out


if __name__ == '__main__':
    print(paramr['gap_size'].get_val_str(val=10e-3))
    print(paramr['gap_size'].get_val_str(val=1e-3, symbol=True))
    print(paramr['liquid_IP'].get_val_str(val=4.65, brace=True))
    print(paramr['additive_IP'].get_val_str(val=1.65, unit='', brace=True))
    print(paramr['streamer_photo_enabled'].get_val_str(val=False))
    print(paramr['needle_voltage'].get_label())
    print(paramr['needle_radius'].get_label(unit=''))


#
