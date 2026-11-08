#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some fuzzy membership functions.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

import torch

from anfis import AnfisNet


def _mk_param(val, device = 'cpu'):
    '''Make a torch parameter from a scalar value'''
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float, device = device))


class GaussMembFunc(torch.nn.Module):
    '''
        Gaussian membership functions, defined by two parameters:
            mu, the mean (center)
            sigma, the standard deviation.
    '''
    def __init__(self, mu, sigma, interval_size = None, extreme = None, device = 'cpu'):
        super(GaussMembFunc, self).__init__()
        self.device = device
        self.extreme = extreme
        self.register_parameter('mu', _mk_param(mu, device = self.device))
        self.register_parameter('sigma', _mk_param(sigma, device = self.device))
        if (interval_size == None):
            self.interval = False
        else:
            self.register_parameter('interval_size', _mk_param(interval_size, device = self.device))
            self.interval = True

    def set_extreme(self, val):# inferior: 0, superior: 1
        self.extreme = val
   
    def forward(self, x):
        
        
        val = torch.zeros(x.shape, device = self.device)
        
        if (self.extreme == 0):
            val[x < self.mu] = 1.0
            val[x >= self.mu] = torch.exp(-torch.pow(x[x >= self.mu] - self.mu, 2) / (2 * self.sigma**2 + 0.000001)).float()
        elif (self.extreme == 1):
            val[x > self.mu] = 1.0
            val[x <= self.mu] = torch.exp(-torch.pow(x[x <= self.mu] - self.mu, 2) / (2 * self.sigma**2 + 0.000001)).float()
        else:
            val = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma**2 + 0.000001))
        
        if (not self.interval):
            return val
        else:
            return torch.stack((torch.max(val - self.interval_size / 2, torch.zeros(x.shape)), torch.min(val + self.interval_size / 2, torch.ones(x.shape))), axis = 1)

    def pretty(self):
        if (not self.interval):
            return 'GaussMembFunc {} {}'.format(self.mu, self.sigma)
        else:
            return 'GaussMembFunc {} {} {:.8g}'.format(self.mu, self.sigma, self.interval_size)


def make_gauss_mfs(sigma, mu_list, interval_size = None, device = 'cpu'):
    '''Return a list of gaussian mfs, same sigma, list of means'''
    mfdefs = [GaussMembFunc(mu, sigma, interval_size = interval_size, device = device) for mu in mu_list]
    mfdefs[0].set_extreme(0)
    mfdefs[-1].set_extreme(1)
    return mfdefs


class BellMembFunc(torch.nn.Module):
    '''
        Generalised Bell membership function; defined by three parameters:
            a, the half-width (at the crossover point)
            b, controls the slope at the crossover point (which is -b/2a)
            c, the center point
    '''
    def __init__(self, a, b, c, interval_size = None, device = 'cpu'):
        super(BellMembFunc, self).__init__()
        self.device = device
        self.register_parameter('a', _mk_param(a, device = self.device))
        self.register_parameter('b', _mk_param(b, device = self.device))
        self.register_parameter('c', _mk_param(c, device = self.device))
        self.b.register_hook(BellMembFunc.b_log_hook)
        if (interval_size == None):
            self.interval = False
        else:
            self.register_parameter('interval_size', _mk_param(float(interval_size), device = self.device))
            self.interval = True

    @staticmethod
    def b_log_hook(grad):
        '''
            Possibility of a log(0) in the grad for b, giving a nan.
            Fix this by replacing any nan in the grad with ~0.
        '''
        grad[torch.isnan(grad)] = 1e-9
        return grad

    def forward(self, x):
        dist = torch.pow((x - self.c)/self.a, 2)
        val = torch.reciprocal(1 + torch.pow(dist, self.b))
        if (not self.interval):
            return val
        else:
            return torch.stack((torch.max(val - self.interval_size / 2, torch.zeros(x.shape)), torch.min(val + self.interval_size / 2, torch.ones(x.shape))), axis = 1)

    def pretty(self):
        if (not self.interval):
            return 'BellMembFunc {} {} {}'.format(self.a, self.b, self.c)
        else:
            return 'BellMembFunc {} {} {} {:.8g}'.format(self.a, self.b, self.c, self.interval_size)


def make_bell_mfs(a, b, clist, interval_size = None, device = 'cpu'):
    '''Return a list of bell mfs, same (a,b), list of centers'''
    return [BellMembFunc(a, b, c, interval_size = interval_size, device = device) for c in clist]


class TriangularMembFunc(torch.nn.Module):
    '''
        Triangular membership function; defined by three parameters:
            a, left foot, mu(x) = 0
            b, midpoint, mu(x) = 1
            c, right foot, mu(x) = 0
    '''
    def __init__(self, a, b, c, interval_size = None, device = 'cpu'):
        super(TriangularMembFunc, self).__init__()
        self.device = device
        assert a <= b and b <= c,\
            'Triangular parameters: must have a <= b <= c.'
        self.register_parameter('a', _mk_param(a, device = self.device))
        self.register_parameter('b', _mk_param(b, device = self.device))
        self.register_parameter('c', _mk_param(c, device = self.device))
        if (interval_size == None):
            self.interval = False
        else:
            self.register_parameter('interval_size', _mk_param(interval_size, device = self.device))
            self.interval = True

    @staticmethod
    def isosceles(width, center, interval_size = None, device = 'cpu'):
        '''
            Construct a triangle MF with given width-of-base and center
        '''
        return TriangularMembFunc(center-width, center, center+width, interval_size = interval_size, device = device)

    def forward(self, x):
        val = torch.where(
            (self.a < x) & (x <= self.b),
            (x - self.a) / (self.b - self.a),
            # else
            torch.where(
                (self.b < x) & (x <= self.c),
                (self.c - x) / (self.c - self.b),
                torch.zeros_like(x, requires_grad=True)))
        if (not self.interval):
            return val
        else:
            return torch.stack((torch.max(val - self.interval_size / 2, torch.zeros(x.shape)), torch.min(val + self.interval_size / 2, torch.ones(x.shape))), axis = 1)
        
    def pretty(self):
        if (not self.interval):
            return 'TriangularMembFunc {} {} {}'.format(self.a, self.b, self.c)
        else:
            return 'TriangularMembFunc {} {} {} {:.8g}'.format(self.a, self.b, self.c, self.interval_size)
    


def make_tri_mfs(width, clist, interval_size = None, device = 'cpu'):
    '''Return a list of triangular mfs, same width, list of centers'''
    return [TriangularMembFunc(c-width/2, c, c+width/2, interval_size = interval_size, device = device) for c in clist]


class TrapezoidalMembFunc(torch.nn.Module):
    '''
        Trapezoidal membership function; defined by four parameters.
        Membership is defined as:
            to the left of a: always 0
            from a to b: slopes from 0 up to 1
            from b to c: always 1
            from c to d: slopes from 1 down to 0
            to the right of d: always 0
    '''
    def __init__(self, a, b, c, d, interval_size = None, device = 'cpu'):
        super(TrapezoidalMembFunc, self).__init__()
        self.device = device
        assert a <= b and b <= c and c <= d,\
            'Trapezoidal parameters: must have a <= b <= c <= d.'
        self.register_parameter('a', _mk_param(a, device = self.device))
        self.register_parameter('b', _mk_param(b, device = self.device))
        self.register_parameter('c', _mk_param(c, device = self.device))
        self.register_parameter('d', _mk_param(d, device = self.device))
        self.interval_size = interval_size
        if (interval_size == None):
            self.interval = False
        else:
            self.register_parameter('interval_size', _mk_param(interval_size, device = self.device))
            self.interval = True
            
    @staticmethod
    def symmetric(topwidth, slope, midpt, device = 'cpu'):
        '''
            Make a (symmetric) trapezoid mf, given
                topwidth: length of top (when mu == 1)
                slope: extra length at either side for bottom
                midpt: center point of trapezoid
        '''
        b = midpt - topwidth / 2
        c = midpt + topwidth / 2
        return TrapezoidalMembFunc(b - slope, b, c, c + slope, device = device)

    @staticmethod
    def rectangle(left, right, device = 'cpu'):
        '''
            Make a Trapezoidal MF with vertical sides (so a==b and c==d)
        '''
        return TrapezoidalMembFunc(left, left, right, right, device = device)

    @staticmethod
    def triangle(left, midpt, right, device = 'cpu'):
        '''
            Make a triangle-shaped MF as a special case of a Trapezoidal MF.
            Note: this may revert to general trapezoid under learning.
        '''
        return TrapezoidalMembFunc(left, midpt, midpt, right, device = device)

    def forward(self, x):
        yvals = torch.zeros_like(x, device = self.device)
        if self.a < self.b:
            incr = torch.ByteTensor(self.a < x) & torch.ByteTensor(x <= self.b)
            yvals[incr] = (x[incr] - self.a) / (self.b - self.a)
        if self.b < self.c:
            decr = torch.ByteTensor(self.b < x) & torch.ByteTensor(x < self.c)
            yvals[decr] = 1
        if self.c < self.d:
            decr = torch.ByteTensor(self.c <= x) & torch.ByteTensor(x < self.d)
            yvals[decr] = (self.d - x[decr]) / (self.d - self.c)
        
        if (not self.interval):
            return yvals
        else:
            return torch.stack((torch.max(yvals - self.interval_size / 2, torch.zeros(x.shape)), torch.min(yvals + self.interval_size / 2, torch.ones(x.shape))), axis = 1)

    def pretty(self):
        if (not self.interval):
            return 'TrapezoidalMembFunc a={} b={} c={} d={}'.\
            format(self.a, self.b, self.c, self.d)
        else:
            return 'TrapezoidalMembFunc a={} b={} c={} d={} interval_size={:.8g}'.\
            format(self.a, self.b, self.c, self.d, self.interval_size)


def make_trap_mfs(width, slope, clist, interval_size = None, device = 'cpu'):
    '''Return a list of symmetric Trap mfs, same (w,s), list of centers'''
    return [TrapezoidalMembFunc.symmetric(width, slope, c, interval_size = interval_size, device = device) for c in clist]


# Make the classes available via (controlled) reflection:
get_class_for = {n: globals()[n]
                 for n in ['BellMembFunc',
                           'GaussMembFunc',
                           'TriangularMembFunc',
                           'TrapezoidalMembFunc',
                           ]}

# Función que genera el modelo ANFIS en función de los parámetros especificados
# x: Datos de entrenamiento utilizados para generar las funciones de pertenencia
# num_mfs:              Número de funciones de pertenencia por variable
# num_out:              Número de salidas del modelo
# Hybrid:               Habilita otra implementación del consecuente (recomendable dejarlo en False)
# num_rules:            Límite de reglas a generar (-1 significa que no hay límite)
# initial_rules:        Establecer manualmente las reglas que va a tener el modelo. Si no, dejar a None
# random_state:         Semilla utilizada para inicializar las reglas, se puede dejar a None
# feature_importance:   Lista indicando los índices de las variables ordenadas por orden de importancia para la generación de reglas, si está a None, todas las reglas serán aleatorias
# null_values:          Booleano que si se pone a True, en la generación de reglas podrán haber variables que no tengan ningún efecto.
# device:               Indicar el dispositivo que se utilizará para realizar la computación
def make_anfis(x, num_mfs=4, num_out=1, hybrid=False, num_rules = -1, initial_rules = None, random_state = None, feature_importance = None, null_values = True, device = 'cpu'):
    '''
        Make an ANFIS model, auto-calculating the (Gaussian) MFs.
        I need the x-vals to calculate a range and spread for the MFs.
        Variables get named x0, x1, x2,... and y0, y1, y2 etc.
    '''
    num_invars = x.shape[1]
    minvals, _ = torch.min(x, dim=0)
    maxvals, _ = torch.max(x, dim=0)
    ranges = maxvals-minvals
    invars = []
    for i in range(num_invars):
        if(type(num_mfs) == int):
            sigma = ranges[i] / num_mfs
            mulist = torch.linspace(minvals[i], maxvals[i], num_mfs, device = device).tolist()
        else: #(type(num_mfs) == list):
            sigma = ranges[i] / num_mfs[i]
            mulist = torch.linspace(minvals[i], maxvals[i], num_mfs[i], device = device).tolist()
        invars.append(('x{}'.format(i), make_gauss_mfs(sigma, mulist, device = device)))
    outvars = ['y{}'.format(i) for i in range(num_out)]
    model = AnfisNet('Simple classifier', invars, outvars, hybrid=hybrid, num_rules = num_rules, initial_rules = initial_rules, feature_importance = feature_importance,
                     null_values = null_values, random_state = random_state, device = device)
    return model