#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Use gap position on z-axis and plot another variable on the x-axis.
'''

# General imports
import numpy as np
from statsmodels.nonparametric import smoothers_lowess
import logging
from scipy.interpolate import pchip_interpolate

# Import from project files
from .. import plotterXZ
from ..load_data import LoadedData
from .shadow_plotter import get_shadow

# Settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
eps = np.finfo(float).eps  # 2.22e-16 for double

# todo: the get methods should give warning when the data is not present
''' idea: other gap-plots:
    - other speed calculations
    - iter no
    - cpu time
    - head-no
    - head-scale (but why bother?)
    - roi-size
    - seed-nos/info
    - seeds critical no
'''

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE PLOTTER                                                #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class GapPlotter(plotterXZ.PlotterXZ):

    @classmethod
    def _get_opt_bool(cls):
        opts = super()._get_opt_bool()
        opts['cdf'] = 'Show Cumulative Density Function'
        return opts

    def __init__(self, zlabel=None, zscale=None, xmin=None, zmin=None,
                 cdf=None, **kwargs):
        super().__init__(**kwargs)
        self.zlabel = 'Position [mm]' if zlabel is None else zlabel
        self.zscale = 1e3 if zscale is None else zscale

        self.xmin = 0 if (xmin is None and not self.logx_on) else xmin
        self.zmin = 0 if (zmin is None and not self.logz_on) else zmin
        self.cdf_on = cdf or False

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Plot to axis                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # note: the gap-plotter should plot x = x(z)
    #       lowess, interpolation, and cdf need to be given special care

    def plot_xz(self, x, z):
        ''' Actually plot data to the axis, return label.'''
        label = super().plot_xz(x, z)  # super do most of the work
        if self.cdf_on:  # only special addition by gap-plotter
            self.cdf(x, z)
        return label

    def lowess(self, x, z, desc=None):
        ''' Use zx instead of xz for gap lowess. '''
        super().lowess_zx(x, z, desc=desc)

    def interpolate(self, x, z, desc=None):
        ''' Use zx instead of xz for gap interpolate. '''
        super().interpolate_zx(x, z, desc=desc)

    def cdf(self, x, z, desc=None):
        ''' Add cumulative distribution for given variable. '''
        logger.log(5, 'Adding cdf')
        desc = desc or dict(self.desc, **self.interp_desc)
        # x = x[(x > x.max() * 0.01) & (x < x.max() * 0.99)]  # trim edges

        # set up distribution
        zl, step = np.linspace(0, 1, num=x.size, retstep=True, endpoint=False)
        zl = zl + step  # set up zl = [step, 1], ie without 0 as startpoint
        z = zl * max(z)
        x = np.sort(x)

        # calculate average, root-mean-sq-dev, and std dev
        # for P(x) = 1 - exp(- a x)
        # y = ln(1 - P(x)) = - a x
        # find 1/a, by estimating
        # a = - yi xi / xj xj
        # x_ave = 1 / a
        # rmsd = sqrt(sum( (P - Pi)**2 ))
        # rmsd = sqrt(sum( (P(x) - zl)**2 ))
        mask = zl > 0
        mask &= zl < 1              # exclude the last part
        mask &= x < x.max() * 0.99  # exclude the last part (saturation?)
        mask &= x > 0               # exclude values that gives zero
        if mask.sum() > 10:
            xixi = (x[mask]**2).sum()
            yixi = (np.log(1 - zl[mask]) * x[mask]).sum()
            xave = - xixi / yixi  # average
            sd = ((1 - np.exp(- x[mask] / xave)) - zl[mask])**2
            rmsd = np.sqrt(np.average(sd))  # root-mean-sq-deviation
            eps2 = (np.log(1 - zl[mask]) - x[mask] / xave)**2
            x2 = (x[mask] - np.average(x[mask]))**2
            xdev = np.sqrt(eps2.sum() / x2.sum() / (x2.size - 2))
            label = f'xave {xave:.2f}, rmsd {rmsd:.2f}, xdev {xdev:.2f}'
            desc = dict(self.desc, label=label)

        # plot to axis
        self.ax_xz.plot(x, z, **desc)
        self.update_min_max(x, z)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE SPECIAL PLOTTERS                                       #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class StreakPlotter(GapPlotter):
    '''Plot propagation time against gap position.'''

    def __init__(self, title=None, xlabel=None, xscale=None,
                 curve=None, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'streak'
        self.title = 'Streak' if title is True else title
        self.xlabel = 'Time [\u00B5s]' if xlabel is None else xlabel
        self.xscale = 1e6 if xscale is None else xscale
        self.curve_on = True if curve is None else curve

    def get_xzt(self, data):
        # Get data
        z = np.array(data.get('streamer_z_min'))
        x = np.array(data.get('sim_time'))
        t = x
        return x, z, t


class StreakCpuPlotter(GapPlotter):
    '''Plot CPU time against gap position.'''

    def __init__(self, title=None, xlabel=None, xscale=None,
                 curve=None, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'streak_cpu'
        self.title = 'Streak CPU' if title is True else title
        self.xlabel = 'Time [ks]' if xlabel is None else xlabel
        self.xscale = 1e-3 if xscale is None else xscale
        self.curve_on = True if curve is None else curve

    def get_xzt(self, data):
        # Get data
        z = np.array(data.get('streamer_z_min'))
        x = np.array(data.get('cpu_time'))
        t = np.array(data.get('sim_time'))
        return x, z, t


class SpeedPlotter(GapPlotter):
    '''Plot propagation speed against gap position.'''

    @classmethod
    def _get_opt_float(cls):
        opts = super()._get_opt_float()
        opts['smooth'] = 'Fraction of data to use for smoothing.'
        return opts

    @classmethod
    def _get_opt_string(cls):
        opts = super()._get_opt_string()
        opts['method'] = 'How to calculate speed.'
        return opts

    def __init__(self, title=None, xlabel=None, xscale=None,
                 curve=None, smooth=None, method=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.method = 'average_v' if method is None else method
        self.fpath_head = f'speed_{self.method}'
        self.title = 'Streamer propagation speed' if title is True else title
        self.xlabel = 'Speed [km/s]' if xlabel is None else xlabel
        self.xscale = 1e-3 if xscale is None else xscale
        self.curve_on = True if curve is None else curve
        self.smooth = 0.2 if type(smooth) is not float else smooth

    def _smooth_v(self, z, t):
        ''' Return smoothed to instantaneous velocity. '''
        # using unique z-val and min time at each position
        zu, ta, ts, tmin, tmax = self._calc_stat(z, t, sort=True)
        v = -np.diff(zu) / np.diff(tmin)
        v = np.insert(v, 0, 0)  # insert 0 as first speed
        v = smoothers_lowess.lowess(
            v, zu, frac=self.smooth, return_sorted=False)
        return v, zu, tmin

    def _average_v(self, z, t):
        ''' Return average velocity until each point. '''
        # works well, but does not show variations well
        dz = z[0] - z[1:]
        dt = t[0] - t[1:]
        v = dz / -dt
        v = np.insert(v, 0, 0)  # insert 0 as first speed
        return v, z, t

    def _window_v(self, z, t):
        ''' Return average velocity within a window. '''
        z_max = max(z)
        ws = (max(z) - min(z)) * self.smooth
        zu, ta, ts, tmin, tmax = self._calc_stat(z, t, sort=True)
        zl = np.linspace(min(zu), max(zu), 50)
        v = []
        t = []
        for zi in zl:
            wmask = (zu > zi) * (zu <= zi + ws)
            if sum(wmask) < 2:
                # hack: this should be improved
                vi = 0
                ti = 0
            else:
                dz = max(zu[wmask]) - min(zu[wmask])
                dt = max(tmin[wmask]) - min(tmin[wmask])
                vi = dz / dt
                ti = min(tmin[wmask])
            v.append(vi)
            t.append(ti)
        v = np.array(v)
        t = np.array(t)
        return v, zl, t

    def _interpolate_t(self, z, t):
        ''' Interpolate t and return instantaneous velocity. '''
        zu, ta, ts, tmin, tmax = self._calc_stat(z, t, sort=True)
        zl = np.linspace(min(zu), max(zu), 50)
        ta = pchip_interpolate(zu, ta, zl)
        v = -np.diff(zl) / np.diff(ta)
        v = np.insert(v, 0, 0)  # insert 0 as first speed
        return v, zl, ta

    def _interpolate_z(self, z, t):
        ''' Interpolate t and return instantaneous velocity. '''
        tl = np.linspace(min(t), max(t), 50)
        za = pchip_interpolate(t, z, tl)
        v = -np.diff(za) / np.diff(tl)
        v = np.insert(v, 0, 0)  # insert 0 as first speed
        return v, za, tl

    def _smooth_t(self, z, t):
        ''' Smooth t and return instantaneous velocity. '''
        zu, ta, ts, tmin, tmax = self._calc_stat(z, t, sort=True)
        t = smoothers_lowess.lowess(
            ta, zu, frac=self.smooth, return_sorted=False)
        v = -np.diff(zu) / np.diff(t)
        v = np.insert(v, 0, 0)  # insert 0 as first speed
        return v, zu, t

    def _smooth_z(self, z, t):
        ''' Smooth z and return instantaneous velocity. '''
        # does not give a nice result
        z = smoothers_lowess.lowess(
            z, t, frac=self.smooth, return_sorted=False)
        v = -np.diff(z) / np.diff(t)
        v = np.insert(v, 0, 0)  # insert 0 as first speed
        return v, z, t

    def get_xzt(self, data):
        # Get data
        z = np.array(data['streamer_z_min'])
        t = np.array(data['sim_time'])

        if z.size < 2:
            msg = 'Too few data points. Skipping.'
            logger.info(msg)
            return np.array([]), np.array([]), np.array([])

        if False:
            pass
        elif self.method == 'average_v':
            # useful method that makes sense
            v, z, t = self._average_v(z, t)
        elif self.method == 'smooth_v':
            v, z, t = self._smooth_v(z, t)
        elif self.method == 'smooth_t':
            # useful method that makes sense
            v, z, t = self._smooth_t(z, t)
        elif self.method == 'window_v':
            v, z, t = self._window_v(z, t)
        # use with caution, these need to be validated
        # elif self.method == 'interpolate_t':
        #     v, z, t = self._interpolate_t(z, t)
        # elif self.method == 'interpolate_z':
        #     v, z, t = self._interpolate_z(z, t)
        # elif self.method == 'smooth_z':
        #     v, z, t = self._smooth_z(z, t)
        else:
            logger.error(f'Unknown method: {self.method}')
            v, z, t = np.array([]), np.array([]), np.array([])

        return v, z, t


class JumpPlotter(GapPlotter):
    '''Plot streamer jump length against gap position.'''

    def __init__(self, title=None, xlabel=None, xscale=None,
                 scatter=None, lowess=None, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'jump'
        self.title = 'Streamer jumps' if title is True else title
        self.xlabel = 'Distance [\u00B5m]' if xlabel is None else xlabel
        self.xscale = 1e6 if xscale is None else xscale
        self.scatter_on = True if scatter is None else scatter
        self.lowess_on = True if lowess is None else lowess

    def get_xzt(self, data):
        # Get data
        t = np.array(data.get('sim_time'))
        z = np.array(data.get('streamer_z_min'))
        x = z[:-1] - z[1:]
        z = z[1:]
        t = t[1:]
        mask = x == 0  # ignore data for no movement
        x = x[~mask]
        z = z[~mask]
        t = t[~mask]
        return x, z, t


class StepTimePlotter(GapPlotter):
    '''Plot simulation step time against gap position. '''

    def __init__(self, title=None, xlabel=None, xscale=None,
                 scatter=None, lowess=None, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'step_time'
        self.title = 'Step time' if title is True else title
        self.xlabel = 'Time [ps]' if xlabel is None else xlabel
        self.xscale = 1e12 if xscale is None else xscale
        self.scatter_on = True if scatter is None else scatter
        self.lowess_on = True if lowess is None else lowess

    def get_xzt(self, data):
        # Get data
        t = np.array(data.get('sim_time'))
        z = np.array(data.get('streamer_z_min'))
        x = t[1:] - t[:-1]  # diff time
        z = z[1:]
        t = t[1:]
        return x, z, t


class CritTimePlotter(GapPlotter):
    '''Plot time to critical avalanche against gap position. '''

    def __init__(self, title=None, xlabel=None, xscale=None,
                 scatter=None, lowess=None, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'crit_time'
        self.title = 'Critical avalanche time' if title is True else title
        self.xlabel = 'Time [ns]' if xlabel is None else xlabel
        self.xscale = 1e9 if xscale is None else xscale
        self.scatter_on = True if scatter is None else scatter
        self.lowess_on = True if lowess is None else lowess

    def get_xzt(self, data):
        # Get data
        t = np.array(data.get('sim_time'))
        z = np.array(data.get('streamer_z_min'))
        y = np.array(data.get('seeds_no_critical'))
        mask = (y > 0)      # include only data with avalanches
        mask[0] = True      # but keep first point, for now
        t = t[mask]         # keep only relevant positions
        z = z[mask]
        x = t[1:] - t[:-1]  # get the time differences
        z = z[1:]           # remove the first point
        t = t[1:]
        return x, z, t


class NewHeadTimePlotter(GapPlotter):
    '''Plot time to critical avalanche against gap position. '''

    def __init__(self, title=None, xlabel=None, xscale=None,
                 scatter=None, lowess=None, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'new_head_time'
        self.title = 'Time to new head' if title is True else title
        self.xlabel = 'Time [ns]' if xlabel is None else xlabel
        self.xscale = 1e9 if xscale is None else xscale
        self.scatter_on = True if scatter is None else scatter
        self.lowess_on = True if lowess is None else lowess

    def get_xzt(self, data):
        # Get data
        t = np.array(data.get('sim_time'))
        z = np.array(data.get('streamer_z_min'))
        y = data.get('streamer_pos_appended')
        mask = [item.size > 0 for item in y]
        mask = np.array(mask)      # include only data with new heads
        mask[0] = True      # but keep first point, for now
        t = t[mask]         # keep only relevant positions
        z = z[mask]
        x = t[1:] - t[:-1]  # get the time differences
        z = z[1:]           # remove the first point
        t = t[1:]
        return x, z, t


class HeadScalePlotter(GapPlotter):
    '''Plot potential scale against gap position.'''

    @classmethod
    def _get_opt_string(cls):
        opts = super()._get_opt_string()
        opts['rescale'] = (
            'Rescale potential, `nnls`, `1`, [`none`]')
        opts['heads'] = (
            'Heads to use, `previous`, [`current`], `removed`, `appended`')
        opts['what'] = (
            'How to report the values, [`each`], `sum`, `max`, `ave`')
        return opts

    def __init__(self, title=None, xlabel=None, xscale=None,
                 scatter=None, lowess=None, rescale=None, heads=None,
                 what=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'headscale'
        self.title = 'Streamer Head Scale' if title is True else title
        self.xlabel = 'Scale [%]' if xlabel is None else xlabel
        self.xscale = 100 if xscale is None else xscale
        self.scatter_on = True if scatter is None else scatter
        self.lowess_on = True if lowess is None else lowess

        self.rescale = 'nnls' if rescale is None else rescale
        self.heads = 'current' if heads is None else heads
        self.what = 'each' if what is None else what
        self.fpath_head += '_' + self.rescale
        self.fpath_head += '_' + self.heads
        self.fpath_head += '_' + self.what

    def get_xzt(self, data):
        x = []
        z = []
        t = []
        for no, ti in zip(data.get('no'), data.get('sim_time')):
            _heads = LoadedData(data=data).get_heads(
                which=self.heads,
                no=no,
                )
            _heads.set_scale(mode=self.rescale)

            if self.what == 'each':
                # ensure scalars (not numpy arrays first)
                z += [float(head.d) for head in _heads]
                x += [float(head.k) for head in _heads]
                t += [ti for head in _heads]
            elif self.what == 'sum':
                # experimental, sum of k's
                # _heads.U0 = 1
                # k = _heads.calc_scale_nnls()
                x += [sum(_heads.k)]
                z += [min(_heads.d)]
                t += [ti]
            else:
                logger.error(f'Error, unknown value: {self.what}')
                return x, z, t
            # idea: max k?
            # idea: ave k?
        z = np.array(z)
        x = np.array(x)
        t = np.array(t)
        sort = np.argsort(-z)  # indx sort large to small
        z = z[sort]
        x = x[sort]
        t = t[sort]
        logger.debug(f'Returning {len(x)} data points')
        return x, z, t


class HeadU0Plotter(GapPlotter):
    '''Plot potential U0 against gap position.'''

    def __init__(self, title=None, xlabel=None, xscale=None,
                 scatter=None, lowess=None, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'headU0'
        self.title = 'Streamer Head U0' if title is True else title
        self.xlabel = 'Voltage [kV]' if xlabel is None else xlabel
        self.xscale = 1e-3 if xscale is None else xscale
        self.scatter_on = True if scatter is None else scatter
        self.lowess_on = True if lowess is None else lowess

    def get_xzt(self, data):
        # get the data
        heads_lst_lst = data.get('streamer_heads_dict')
        heads = [head
                for heads_lst in heads_lst_lst
                for head in heads_lst]
        x = [head['U0'] for head in heads]
        z = [float(head['pos'][2]) for head in heads]
        # order data
        z = np.array(z)
        x = np.array(x)
        sort = np.argsort(-z)  # indx sort large to small
        z = z[sort]
        x = x[sort]
        t = []  # todo: implement
        return x, z, t


class HeadNoPlotter(GapPlotter):
    '''Plot streamer head number against gap position.'''

    def __init__(self, title=None, xlabel=None, xscale=None,
                 scatter=None, lowess=None, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'headsno'
        self.title = 'Streamer Head Number' if title is True else title
        self.xlabel = 'Number [1]' if xlabel is None else xlabel
        self.xscale = 1 if xscale is None else xscale
        self.scatter_on = True if scatter is None else scatter
        self.lowess_on = True if lowess is None else lowess

    def get_xzt(self, data):
        # get the data
        heads_lst_lst = data.get('streamer_heads_dict')
        x = [len(heads_lst) for heads_lst in heads_lst_lst]
        z = [np.min([head['pos'][2] for head in heads_lst])
             for heads_lst in heads_lst_lst]
        # order data
        z = np.array(z)
        x = np.array(x)
        t = []  # todo: implement
        return x, z, t


class HeadPotentialPlotter(GapPlotter):
    '''Plot potential against gap position. '''

    @classmethod
    def _get_opt_string(cls):
        opts = super()._get_opt_string()
        opts['rescale'] = (
            'Rescale potential, [`nnls`], `1`, `none`, `U0`')
        opts['heads'] = (
            'Heads to use, `previous`, [`current`], `removed`, `appended`')
        opts['pos'] = (
            'Positions to use, `appended`, [`current`], `removed`, `previous`')
        opts['what'] = (
            'How to report the values, [`each`], `leading`, `max`')
        return opts

    def __init__(self, title=None, xlabel=None, xscale=None,
                 scatter=None, lowess=None,
                 rescale=None, heads=None, pos=None,
                 what=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.title = 'Streamer Head Potential' if title is True else title
        self.xlabel = 'Potential [kV]' if xlabel is None else xlabel
        self.xscale = 1e-3 if xscale is None else xscale
        self.scatter_on = True if scatter is None else scatter
        self.lowess_on = True if lowess is None else lowess

        self.rescale = 'nnls' if rescale is None else rescale
        self.heads = 'current' if heads is None else heads
        self.pos = 'current' if pos is None else pos
        self.what = 'each' if what is None else what

        self.fpath_head = 'heads_epot'
        self.fpath_head += '_' + self.rescale
        self.fpath_head += '_' + self.heads
        self.fpath_head += '_' + self.pos
        self.fpath_head += '_' + self.what

    def get_xzt(self, data):
        # initiate output
        x = []
        y = []
        z = []
        t = []
        v = []
        n = []

        # get data
        # idea: this can also be done by knowing only positions
        #       given that rp need is assumed
        # idea: this can also be done by without appended/removed
        #       given that data is collected every step
        for idx, no in enumerate(data['no']):

            time = data['sim_time'][idx]
            heads = LoadedData(data=data).get_heads(which=self.heads, idx=idx)
            pos = LoadedData(data=data).get_heads(which=self.pos, idx=idx).pos

            # scale the heads
            _mode = self.rescale
            _mode = None if _mode =='U0' else _mode  # note: see below
            heads.set_scale(mode=_mode)

            # get electric potential at positions
            val = heads.epot(pos)

            # U0 is used to get potential at heads. no rescale needed
            if self.rescale == 'U0':
                val = heads.U0

            # define how to append
            def _append_i(i):
                x.append(pos[0, i])
                y.append(pos[1, i])
                z.append(pos[2, i])
                v.append(val[i])
                t.append(time)
                n.append(no)

            # append the correct 'what'
            if self.what == 'each':
                for i in range(pos.shape[1]):
                    _append_i(i)
            elif self.what == 'leading':
                if pos.shape[1] > 0:
                    i = np.argmin(pos[2, :])
                    _append_i(i)
            elif self.what == 'max':
                if pos.shape[1] > 0:
                    i = np.argmin(-val)
                    _append_i(i)
            else:
                logger.error(f'Error, unknown value: {self.what}')
                return x, z, t

        # change from list to array
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        t = np.array(t)
        v = np.array(v)
        n = np.array(n)

        # fix output variables
        x = v
        sort = np.argsort(-z)  # index sort large to small
        z = z[sort]
        x = x[sort]
        t = t[sort]
        return x, z, t


class HeadEstrPlotter(GapPlotter):
    '''Plot electric strength against gap position'''

    @classmethod
    def _get_opt_string(cls):
        opts = super()._get_opt_string()
        opts['rescale'] = (
            'Rescale potential, [`nnls`], `1`, `none`, `U0`')
        opts['heads'] = (
            'Heads to use, `previous`, [`current`], `removed`, `appended`')
        opts['pos'] = (
            'Positions to use, `appended`, [`current`], `removed`, `previous`')
        opts['what'] = (
            'How to report the values, [`each`], `leading`, `max`')
        return opts

    def __init__(self, title=None, xlabel=None, xscale=None,
                 scatter=None, lowess=None,
                 rescale=None, heads=None, pos=None,
                 what=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.title = 'Streamer Head Electric Strength' if title is True else title
        self.xlabel = 'Electric Strength [GV/m]' if xlabel is None else xlabel
        self.xscale = 1e-9 if xscale is None else xscale
        self.scatter_on = True if scatter is None else scatter
        self.lowess_on = True if lowess is None else lowess

        self.rescale = 'nnls' if rescale is None else rescale
        self.heads = 'current' if heads is None else heads
        self.pos = 'current' if pos is None else pos
        self.what = 'each' if what is None else what

        self.fpath_head = 'heads_estr'
        self.fpath_head += '_' + self.rescale
        self.fpath_head += '_' + self.heads
        self.fpath_head += '_' + self.pos
        self.fpath_head += '_' + self.what

    def get_xzt(self, data):
        # initiate output
        x = []
        y = []
        z = []
        t = []
        v = []
        n = []

        # get data
        # idea: this can also be done by knowing only positions
        #       given that rp need is assumed
        # idea: this can also be done by without appended/removed
        #       given that data is collected every step
        for idx, no in enumerate(data['no']):
            time = data['sim_time'][idx]
            heads = LoadedData(data=data).get_heads(which=self.heads, idx=idx)
            pos = LoadedData(data=data).get_heads(which=self.pos, idx=idx).pos

            # scale the heads
            _mode = self.rescale
            heads.set_scale(mode=_mode)

            # get electric potential
            val = heads.estr(pos)

            # define how to append
            def _append_i(i):
                x.append(pos[0, i])
                y.append(pos[1, i])
                z.append(pos[2, i])
                v.append(val[i])
                t.append(time)
                n.append(no)

            # append the correct 'what'
            if self.what == 'each':
                for i in range(pos.shape[1]):
                    _append_i(i)
            elif self.what == 'leading':
                if pos.shape[1] > 0:
                    i = np.argmin(pos[2, :])
                    _append_i(i)
            elif self.what == 'max':
                if pos.shape[1] > 0:
                    i = np.argmin(-val)
                    _append_i(i)
            else:
                logger.error(f'Error, unknown value: {self.what}')
                return x, z, t

        # change from list to array
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        t = np.array(t)
        v = np.array(v)
        n = np.array(n)

        # fix output variables
        x = v
        sort = np.argsort(-z)  # index sort large to small
        z = z[sort]
        x = x[sort]
        t = t[sort]
        return x, z, t


class SeedNoPlotter(GapPlotter):
    '''Plot seed number against gap position.'''

    def __init__(self, title=None, xlabel=None, xscale=None,
                 scatter=None, lowess=None, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'seedno'
        self.title = 'Streamer Seed Number' if title is True else title
        self.xlabel = 'Number [k]' if xlabel is None else xlabel
        self.xscale = 1e-3 if xscale is None else xscale
        self.scatter_on = True if scatter is None else scatter
        self.lowess_on = True if lowess is None else lowess

    def get_xzt(self, data):
        z = np.array(data.get('streamer_z_min'))
        x = np.array(data.get('seeds_no'))
        t = np.array(data.get('sim_time'))
        z = np.array(z)
        x = np.array(x)
        t = np.array(t)
        return x, z, t


class SeedNoCriticalPlotter(GapPlotter):
    '''Plot seed number against gap position.'''

    def __init__(self, title=None, xlabel=None, xscale=None,
                 scatter=None, lowess=None, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'seednocrit'
        self.title = 'Seed Number Critical' if title is True else title
        self.xlabel = 'Number [1]' if xlabel is None else xlabel
        self.xscale = 1 if xscale is None else xscale
        self.scatter_on = True if scatter is None else scatter
        self.lowess_on = True if lowess is None else lowess

    def get_xzt(self, data):
        z = np.array(data.get('streamer_z_min'))
        x = np.array(data.get('seeds_no_critical'))
        t = np.array(data.get('sim_time'))
        mask = x == 0  # remove irrelevant data
        z = z[~mask]
        x = x[~mask]
        t = t[~mask]
        return x, z, t


class ShadowRPlotter(GapPlotter):
    '''Plot streamer shadow radial position against gap position.'''

    def __init__(self, title=None, xlabel=None, xscale=None,
                 scatter=None, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'shadowr'
        self.title = 'Shadow Diff R' if title is True else title
        self.xlabel = 'Distance [mm]' if xlabel is None else xlabel
        self.xscale = 1e3 if xscale is None else xscale
        self.scatter_on = True if scatter is None else scatter

    def get_xzt(self, data):
        pos, no, key = get_shadow(data)
        z = pos[2, :]
        x = np.sqrt(pos[0, :]**2 + pos[1, :]**2)
        sort = np.argsort(-z)  # indx sort large to small
        z = z[sort]
        x = x[sort]
        t = []
        return x, z, t


# store a dict of available plotters
plotters = {
    'streak': StreakPlotter,
    'streak_cpu': StreakCpuPlotter,
    'speed': SpeedPlotter,
    'jump': JumpPlotter,
    'step_time': StepTimePlotter,
    'crit_time': CritTimePlotter,
    'new_head_time': NewHeadTimePlotter,
    'headscale': HeadScalePlotter,
    'head_u0': HeadU0Plotter,
    'headsno': HeadNoPlotter,
    'headsepot': HeadPotentialPlotter,
    'headsestr': HeadEstrPlotter,
    'seedno': SeedNoPlotter,
    'seednocrit': SeedNoCriticalPlotter,
    'shadowr': ShadowRPlotter,
}


#
