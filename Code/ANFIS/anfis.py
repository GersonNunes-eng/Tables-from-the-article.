#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    ANFIS in torch: the ANFIS layers
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
    Acknowledgement: twmeggs' implementation of ANFIS in Python was very
    useful in understanding how the ANFIS structures could be interpreted:
        https://github.com/twmeggs/anfis
'''

import itertools
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
import math

dtype = torch.float


class FuzzifyVariable(torch.nn.Module):
    '''
        Represents a single fuzzy variable, holds a list of its MFs.
        Forward pass will then fuzzify the input (value for each MF).
    '''
    def __init__(self, mfdefs, device = 'cpu'):
        super(FuzzifyVariable, self).__init__()
        self.device = device
        if isinstance(mfdefs, list):  # No MF names supplied
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self):
        '''Return the actual number of MFs (ignoring any padding)'''
        return len(self.mfdefs)

    def members(self):
        '''
            Return an iterator over this variables's membership functions.
            Yields tuples of the form (mf-name, MembFunc-object)
        '''
        return self.mfdefs.items()

    def pad_to(self, new_size):
        '''
            Will pad result of forward-pass (with zeros) so it has new_size,
            i.e. as if it had new_size MFs.
        '''
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x):
        '''
            Yield a list of (mf-name, fuzzy values) for these input values.
        '''
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield(mfname, yvals)

    def forward(self, x):
        '''
            Return a tensor giving the membership value for each MF.
            x.shape: n_cases
            y.shape: n_cases * n_mfs
        '''
        y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding, device = self.device)], dim=1)
        return y_pred


class FuzzifyLayer(torch.nn.Module):
    '''
        A list of fuzzy variables, representing the inputs to the FIS.
        Forward pass will fuzzify each variable individually.
        We pad the variables so they all seem to have the same number of MFs,
        as this allows us to put all results in the same tensor.
    '''
    def __init__(self, varmfs, varnames=None, device = 'cpu'):
        self.device = device
        super(FuzzifyLayer, self).__init__()
        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self):
        '''Return the number of input variables'''
        return len(self.varmfs)

    @property
    def max_mfs(self):
        ''' Return the max number of MFs in any variable'''
        return max([var.num_mfs for var in self.varmfs.values()])

    def __repr__(self):
        '''
            Print the variables, MFS and their parameters (for info only)
        '''
        r = ['Input variables']
        for varname, members in self.varmfs.items():
            r.append('Variable {}'.format(varname))
            for mfname, mfdef in members.mfdefs.items():
                r.append('- {}: {}({})'.format(mfname,
                         mfdef.__class__.__name__,
                         ', '.join(['{}={}'.format(n, p.item())
                                   for n, p in mfdef.named_parameters()])))
        return '\n'.join(r)

    def forward(self, x):
        ''' Fuzzyify each variable's value using each of its corresponding mfs.
            x.shape = n_cases * n_in
            y.shape = n_cases * n_in * n_mfs
        '''
        assert x.shape[1] == self.num_in,\
            '{} is wrong no. of input values'.format(self.num_in)
        y_pred = torch.stack([var(x[:, i:i+1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred


class AntecedentLayer(torch.nn.Module):
    '''
        Form the 'rules' by taking all possible combinations of the MFs
        for each variable. Forward pass then calculates the fire-strengths.
    '''
    def __init__(self, varlist, num_rules = -1, random_state = None, initial_rules = None, feature_importance = None, null_values = False, device = 'cpu'):
        super(AntecedentLayer, self).__init__()
        self.device = device
        self.null_values = null_values
        self.option = 0
        # Count the (actual) mfs for each variable:
        if (initial_rules != None):
            self.mf_indices = initial_rules.to(self.device)
        else:
            
            if (self.null_values):
                mf_count = [var.num_mfs + 1 for var in varlist]
            else:
                mf_count = [var.num_mfs for var in varlist]
            
            num_comb_rules = np.prod(mf_count)
            #print(num_comb_rules)
            
            if (num_rules < 0 or num_rules >= num_comb_rules): # Generación de reglas exponencial
                # Now make the MF indices for each rule:
                mf_indices = itertools.product(*[range(n) for n in mf_count])
                self.mf_indices = torch.tensor(list(mf_indices), device = self.device)
                # mf_indices.shape is n_rules * n_in
            elif(feature_importance != None):
                tam_var = len(varlist)
                # Paso 1: Seleccionar el número de variables que se van a utilizar
                count_rules = 1
                size = 0
                while (count_rules < num_rules or size > len(feature_importance)):
                    count_rules = count_rules * mf_count[feature_importance[size]]
                    size = size + 1
                    
                size = size - 1
                
                # Paso 2: Generar de forma determinista las combinaciones de reglas
                mf_indices = torch.zeros(num_rules, tam_var, dtype = torch.int64, device = device)
                index1 = [mf_count[i] for i in feature_importance[:size]]
                #print(torch.tensor(list(itertools.product(*[range(n) for n in index1]))[:num_rules]).shape)
                combination = torch.tensor(list(itertools.product(*[range(n) for n in index1]))[:num_rules], device = device)
                
                # Paso 3: Se duplica la matriz verticalmente y se corta en el límite de reglas establecido
                combinations = combination
                while(combinations.shape[0] < num_rules):
                    combinations = torch.cat([combinations, combination])
                
                
                mf_indices[:,feature_importance[:size]] = combinations[:num_rules]
                
                # Paso 4: Se generan aleatoriamente el resto de reglas
                if (random_state != None):
                    torch.random.seed = random_state
                    torch.manual_seed(random_state)
                
                for i in feature_importance[size:]:
                    num_mfs = varlist[i].num_mfs
                    mf_indices[:, i] = torch.randint(0, num_mfs, (num_rules,), device = self.device)
                # Se cortan las reglas que sobresalen
                
                self.mf_indices = mf_indices
            else:
                # Generar aleatoriamente nuevas reglas para evitar la exponencialidad
                if (random_state != None):
                    torch.random.seed = random_state
                    torch.manual_seed(random_state)
                tam_var = len(varlist)
                
                rules = torch.zeros(num_rules, tam_var, dtype = torch.int64, device = self.device)
                
                for i in range(tam_var):
                    rules[:, i] = torch.randint(0, mf_count[i], (num_rules,), device = self.device)
                
                mask_zero = rules.max(axis = 1).values
                index_zero = torch.where(mask_zero == 0)[0]
                rand_index = torch.randint(0, rules.shape[1], (index_zero.shape[0],)).int()
                for i in range(rand_index.shape[0]):
                    rules[index_zero[i], rand_index[i]] = 1
                
                self.mf_indices = rules
                
            
        # Obtener el número de antecedentes no neutros por regla
        if (self.null_values):
            self.num_ants = (self.mf_indices > 0).sum(axis = 1)
        #    self.mf_indices = self.mf_indices - 1
        else:
            self.num_ants = None
        
        self.truth_rules = torch.nn.Parameter(torch.ones(self.mf_indices.shape[0], device = self.device))
        
        self.convex_weights = torch.nn.Parameter(torch.zeros(self.mf_indices.shape[0], device = self.device) + 0.5)
        self.sugeno_weights = torch.nn.Parameter(torch.ones(self.mf_indices.shape, device = self.device))

        #self.sugeno_weights_2 = torch.nn.Parameter(torch.zeros(self.mf_indices.shape, device = self.device) + 0.01)
        
    def num_rules(self):
        return len(self.mf_indices)

    def extra_repr(self, varlist=None):
        if not varlist:
            return None
        row_ants = []
        mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        for rule_idx in itertools.product(*[range(n) for n in mf_count]):
            thisrule = []
            for (varname, fv), i in zip(varlist.items(), rule_idx):
                thisrule.append('{} is {}'
                                .format(varname, list(fv.mfdefs.keys())[i]))
            row_ants.append(' and '.join(thisrule))
        return '\n'.join(row_ants)
    
    def add_rule(self, rule):
        self.mf_indices = torch.cat((self.mf_indices, rule))

    def delete_rule(self, index):
        self.mf_indices = torch.cat((self.mf_indices[:index], self.mf_indices[index + 1:]))

    def forward(self, x):
        ''' Calculate the fire-strength for (the antecedent of) each rule
            x.shape = n_cases * n_in * n_mfs
            y.shape = n_cases * n_rules
        '''
        
        if (self.null_values):
            null_v = torch.ones((x.shape[0], x.shape[1], 1), device = self.device)
            fuzzy = torch.cat((null_v, x), dim = 2)
        else:
            fuzzy = x
        #print(fuzzy)
        # Expand (repeat) the rule indices to equal the batch size:
        batch_indices = self.mf_indices.expand((fuzzy.shape[0], -1, -1))
        # Then use these indices to populate the rule-antecedents
        
        ants = torch.gather(fuzzy.transpose(1, 2), 1, batch_indices)
        
        if (self.option == 0): # Producto
            rules = torch.prod(ants, dim=2)
        elif (self.option == 1): # Mínimo
            rules = torch.min(ants, dim=2).values
        elif (self.option == 2): # Integral de sugeno generalizada
            rules = generalized_sugeno_integral(ants, self.sugeno_weights, device = self.device)
        elif (self.option == 3): # OWA
            rules = aggregation_owa(ants, self.sugeno_weights, self.mf_indices, device = self.device)
        elif (self.option == 4):
            rules = aggregation_F1(ants, self.mf_indices, alpha = 0.1, num_ants = self.num_ants, device = self.device)
        elif (self.option == 5):
            rules = aggregation_F1(ants, self.mf_indices, alpha = 0.25, num_ants = self.num_ants, device = self.device)
        elif (self.option == 6):
            rules = aggregation_F1(ants, self.mf_indices, alpha = 0.5, num_ants = self.num_ants, device = self.device)
        elif (self.option == 7):
            rules = aggregation_F2(ants, k = 1.1)
        elif (self.option == 8):
            rules = aggregation_F2(ants, k = 1.2)
        elif (self.option == 9):
            rules = aggregation_F3(ants, alpha = 0.5, num_ants = self.num_ants, device = self.device)
        elif (self.option == 10):
            rules = aggregation_F3(ants, alpha = 0.7, num_ants = self.num_ants, device = self.device)
        elif (self.option == 11):
            rules = aggregation_F3(ants, alpha = 0.9, num_ants = self.num_ants, device = self.device)
        elif (self.option == 12): # Media geométrica
            rules = torch.prod(ants, dim = 2)
            rules = torch.where(rules > 1e-10, rules, torch.tensor(0.0, device=rules.device))
            rules = rules.pow(1.0 / self.num_ants)
        elif (self.option == 13): # Overlap 1
            rules = torch.prod(ants, dim = 2)
            rules = torch.where(rules > 1e-10, rules, torch.tensor(0.0, device=rules.device))
            rules = (torch.abs(torch.min(ants, dim=2).values * torch.prod(ants, dim = 2)+1e-10))**0.5 
        elif (self.option == 14): # Overlap 2
            min_ = torch.min(ants, dim = 2).values
            prod_ = torch.prod(ants, dim=2)
            rules = min_ / (min_ + 1 - prod_)
            
        # Para evitar errores por underflow los numeros muy pequeños se convierten a 0
        rules = torch.where(rules > 1e-10, rules, torch.tensor(0.0, device=rules.device))
        
        # Parametro tau de GTSK
        t_rules = rules * self.truth_rules
        return t_rules       

def aggregation_owa(ants, weights, indices, device = 'cpu'):
    weights_copy = weights.clone()
    ants_copy = ants.clone()
    weights_copy[indices == 0] = 0
    ants_copy[indices.repeat(ants.shape[0], 1, 1) == 0] = 0
    
    ordered, _ = torch.sort(ants_copy, descending = True, axis = 2)
    
    n_weights = F.normalize(weights_copy, p=1, dim=1)
    res = ordered * n_weights
    res = res.sum(axis = 2)
        
    return res

def generalized_sugeno_integral(fuzzy_measures, weights, device = 'cpu'):
    res = fuzzy_measures * F.normalize(weights, p=1, dim=1)
    res2 = torch.sum(res, dim = 2)
    return res2

# Media si min > alpha; Producto en otro caso
def aggregation_F1(ants, indices, alpha = 0.1, num_ants = None, device = 'cpu'):
    mask = ants.min(axis = 2).values > alpha
    
    res = torch.zeros((ants.shape[0], ants.shape[1]), device = device)# + epsilon
    
    if (num_ants == None):
        res[mask] = ants[mask].mean(dim = 1)
    else:
        # Se retiran del calculo de la media los antecedentes neutros
        num_ants_rep = num_ants.repeat(ants.shape[0],1)
        neutral_ants = ants[mask].shape[1] - num_ants_rep[mask]
        res[mask] = (ants[mask].sum(dim = 1) - neutral_ants ) / num_ants_rep[mask]# Media
        #res[mask] = (ants[mask].max(dim = 1).values + ants[mask].min(dim = 1).values) / num_ants_rep[mask]# Media
    res[~mask] = ants[~mask].prod(dim = 1)
    
    return res

# Overlap
def aggregation_F2(ants, k = 1.1):
    
    res = ants[:, :, 0]
    
    for i in range(1, ants.shape[2]):
        res = torch.min(res**k * ants[:, :, i], res * ants[:, :, i]**k)
        
    return res

def aggregation_F3(ants, alpha = 0.1, num_ants = None, device = 'cpu'):
    epsilon = 1e-10
    # Agregación LRA
    alphas = torch.zeros((ants.shape[0], ants.shape[1]), device = device) + alpha
    #mask = ants.min(axis = 2).values > alphas
    mask = ants.prod(axis = 2) > alphas
    
    res = torch.zeros((ants.shape[0], ants.shape[1]), device = device)# + epsilon
    
    if (num_ants == None):
        res[mask] = ants[mask].mean(dim = 1)
    else:
        # Se retiran del calculo de la media los antecedentes neutros
        num_ants_rep = num_ants.repeat(ants.shape[0],1)
        neutral_ants = ants[mask].shape[1] - num_ants_rep[mask]
        res[mask] = (ants[mask].sum(dim = 1) - neutral_ants ) / num_ants_rep[mask]# Media
    #res[~mask] = ants[~mask].min(dim = 1).values
    res[~mask] = ants[~mask].prod(dim = 1)
    
    return res

class ConsequentLayer(torch.nn.Module):
    '''
        A simple linear layer to represent the TSK consequents.
        Hybrid learning, so use MSE (not BP) to adjust coefficients.
        Hence, coeffs are no longer parameters for backprop.
    '''
    def __init__(self, d_in, d_rule, d_out, device = 'cpu'):
        super(ConsequentLayer, self).__init__()
        self.device = device
        c_shape = torch.Size([d_rule, d_out, d_in+1])
        #torch.manual_seed(63)
        self._coeff = torch.rand(c_shape, dtype=dtype, requires_grad=True, device = self.device)

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        '''
            Record new coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        '''
        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}'\
            .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def fit_coeff(self, x, weights, y_actual):
        '''
            Use LSE to solve for coeff: y_actual = coeff * (weighted)x
                  x.shape: n_cases * n_in
            weights.shape: n_cases * n_rules
            [ coeff.shape: n_rules * n_out * (n_in+1) ]
                  y.shape: n_cases * n_out
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Shape of weighted_x is n_cases * n_rules * (n_in+1)
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
        # Can't have value 0 for weights, or LSE won't work:
        weighted_x[weighted_x == 0] = 1e-12
        # Squash x and y down to 2D matrices for gels:
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)
        # Use gels to do LSE, then pick out the solution rows:
        try:
            #coeff_2d, _ = torch.lstsq(y_actual_2d, weighted_x_2d)
            coeff_2d = torch.linalg.lstsq(weighted_x_2d, y_actual_2d).solution
        except RuntimeError as e:
            print('Internal error in lstsq', e)
            print('Weights are:', weighted_x)
            raise e
        coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]
        # Reshape to 3D tensor: divide by rules, n_in+1, then swap last 2 dims
        self.coeff = coeff_2d.view(weights.shape[1], x.shape[1]+1, -1)\
            .transpose(1, 2)
        # coeff dim is thus: n_rules * n_out * (n_in+1)

    def forward(self, x):
        '''
            Calculate: y = coeff * x + const   [NB: no weights yet]
                  x.shape: n_cases * n_in
              coeff.shape: n_rules * n_out * (n_in+1)
                  y.shape: n_cases * n_out * n_rules
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1, device = self.device)], dim=1)
        # Need to switch dimansion for the multipy, then switch back:
        y_pred = torch.matmul(self.coeff, x_plus.t())
        return y_pred.transpose(0, 2)  # swaps cases and rules

        
class PlainConsequentLayer(ConsequentLayer):
    '''
        A linear layer to represent the TSK consequents.
        Not hybrid learning, so coefficients are backprop-learnable parameters.
    '''
    def __init__(self, *params, device = 'cpu'):
        super(PlainConsequentLayer, self).__init__(*params, device = device)
        self.device = device
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self.coefficients

    def fit_coeff(self, x, weights, y_actual):
        '''
        '''
        assert False,\
            'Not hybrid learning: I\'m using BP to learn coefficients'

    def add_rule(self):
        new_coeff = torch.zeros(1, self.coefficients.data.shape[1], self.coefficients.data.shape[2], device = self.device)
        new_coeffs = torch.cat((self.coefficients.data, new_coeff))
        self.register_parameter('coefficients',
                                torch.nn.Parameter(new_coeffs))


    def delete_rule(self, index):
        new_coeffs = torch.cat((self.coefficients.data[:index], self.coefficients.data[index + 1:]))
        self.register_parameter('coefficients',
                                torch.nn.Parameter(new_coeffs))


class MLP_ConsequentLayer(torch.nn.Module):#ConsequentLayer
    '''
        A linear layer to represent the TSK consequents.
        Not hybrid learning, so coefficients are backprop-learnable parameters.
    '''
    def __init__(self, d_in, d_rule, d_out, device = 'cpu'):
        super(MLP_ConsequentLayer, self).__init__()
        self.device = device
        
        c_shape = torch.Size([d_rule, d_out, d_in+1])
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True, device = self.device)
        
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))
        
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(d_rule, int(d_rule / 2), device = self.device),#int(d_rule / 2)
             torch.nn.ReLU(),
             torch.nn.Linear(int(d_rule / 2), 1, device = self.device),
             torch.nn.ReLU(),
             ])

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self.coefficients

    def fit_coeff(self, x, weights, y_actual):
        '''
        '''
        assert False,\
            'Not hybrid learning: I\'m using BP to learn coefficients'

    def add_rule(self):
        new_coeff = torch.zeros(1, self.coefficients.data.shape[1], self.coefficients.data.shape[2], device = self.device)
        new_coeffs = torch.cat((self.coefficients.data, new_coeff))
        self.register_parameter('coefficients',
                                torch.nn.Parameter(new_coeffs))


    def delete_rule(self, index):
        new_coeffs = torch.cat((self.coefficients.data[:index], self.coefficients.data[index + 1:]))
        self.register_parameter('coefficients',
                                torch.nn.Parameter(new_coeffs))

    def forward(self, x, tsk):
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1, device = self.device)], dim=1)
        # Need to switch dimansion for the multipy, then switch back:
        c1 = torch.matmul(self.coeff, x_plus.t()).transpose(0, 2)
        
        resi = c1.squeeze(1) * tsk
        
        for layer in self.layers:
            resi = layer(resi)
            
        y_pred = resi
        
        return y_pred
        

class WeightedSumLayer(torch.nn.Module):
    '''
        Sum the TSK for each outvar over rules, weighted by fire strengths.
        This could/should be layer 5 of the Anfis net.
        I don't actually use this class, since it's just one line of code.
    '''
    def __init__(self):
        super(WeightedSumLayer, self).__init__()

    def forward(self, weights, tsk):
        '''
            weights.shape: n_cases * n_rules
                tsk.shape: n_cases * n_out * n_rules
             y_pred.shape: n_cases * n_out
        '''
        # Add a dimension to weights to get the bmm to work:
        y_pred = torch.bmm(tsk, weights.unsqueeze(2))
        return y_pred.squeeze(2)

# HTSK-LN-RELU (Modificación de la capa de normalización)
'''https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
   https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html'''
class HTSK_LN_RELU(torch.nn.Module):
    def __init__(self, num_rules, device = 'cpu'):
        super(HTSK_LN_RELU, self).__init__()
        self.device = device
        self.layer_norm = torch.nn.LayerNorm(num_rules, device = self.device)
        self.relu = torch.nn.ReLU()

    def HTSK(self, x, d):
        eps = 1e-10
        #f1 = (x + eps)**(1.0 / d)
        f1 = torch.zeros(x.shape, device = self.device)
        f1[x > 0.0] = x[x > 0.0]**(1.0 / d)# posible error de gradiente sin epsilon
        out = F.normalize(f1, p=1, dim=1)
        return f1
            

    def forward(self, x, d):
        #out0 = F.normalize(self.x, p=1, dim=1)
        out0 = self.HTSK(x, d)
        out1 = self.layer_norm(out0)#out0
        out2 = self.relu(out1)
        out3 = F.normalize(x, p=1, dim=1)#out2
        return out3


class AnfisNet(torch.nn.Module):
    '''
        This is a container for the 5 layers of the ANFIS net.
        The forward pass maps inputs to outputs based on current settings,
        and then fit_coeff will adjust the TSK coeff using LSE.
    '''
    def __init__(self, description, invardefs, outvarnames, hybrid=False, num_rules = -1, initial_rules = None,
                 random_state = 0, feature_importance = None, null_values = True, device = 'cpu'):
        super(AnfisNet, self).__init__()
        self.device = device
        self.initial_rules = initial_rules
        self.random_state = random_state
        self.description = description
        self.outvarnames = outvarnames
        self.hybrid = hybrid
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs, device = device) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        
        max_rules = math.prod([len(mfs) for _, mfs in invardefs])
        if (self.initial_rules != None):
            self.num_rules = initial_rules.shape[0]
        elif (num_rules < 0 or num_rules >= max_rules):
            self.num_rules = max_rules
        else:
            self.num_rules = num_rules
        if self.hybrid:
            cl = ConsequentLayer(self.num_in, self.num_rules, self.num_out, device = self.device)
        else:
            cl = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out, device = self.device)
            #cl = MLP_ConsequentLayer(self.num_in, self.num_rules, self.num_out, device = self.device)
        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames, device = self.device)),
            ('rules', AntecedentLayer(mfdefs, num_rules = self.num_rules, initial_rules = self.initial_rules,
                                      random_state = self.random_state, feature_importance = feature_importance,
                                      null_values = null_values, device = self.device)),
            # normalisation layer is just implemented as a function.
            ('htsk_ln_relu', HTSK_LN_RELU(self.num_rules, device = self.device)),
            ('consequent', cl)
            # weighted-sum layer is just implemented as a function.
            ]))
        #self.dropout = torch.nn.Dropout(p = 0.1)
    
    @property
    def num_out(self):
        return len(self.outvarnames)

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x, y_actual):
        '''
            Do a forward pass (to get weights), then fit to y_actual.
            Does nothing for a non-hybrid ANFIS, so we have same interface.
        '''
        if self.hybrid:
            self(x)
            self.layer['consequent'].fit_coeff(x, self.weights, y_actual)

    def input_variables(self):
        '''
            Return an iterator over this system's input variables.
            Yields tuples of the form (var-name, FuzzifyVariable-object)
        '''
        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self):
        '''
            Return an list of the names of the system's output variables.
        '''
        return self.outvarnames

    def extra_repr(self):
        rstr = []
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        for i, crow in enumerate(self.layer['consequent'].coeff):
            rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
            rstr.append(' '*9+'THEN {}'.format(crow.tolist()))
        return '\n'.join(rstr)
    
    def get_membership_params(self):
        var_keys = self.layer['fuzzify'].varmfs.keys()
        params = []
        for var in var_keys:
            params_var = []
            members = self.layer['fuzzify'].varmfs[var].mfdefs.keys()
            for memb in members:
                lista = list(self.layer['fuzzify'].varmfs[var].mfdefs[memb].parameters())
                params_var.append(lista)
            params.append(params_var)
        return params

    def get_rules(self):
        rules = self.layer['rules'].mf_indices
        return rules
    
    def add_rule(self, rule):
        self.num_rules = self.num_rules + 1
        # Antecedente
        self.layer['rules'].add_rule(rule)
        # Consecuente
        self.layer['consequent'].add_rule()
        
    def delete_rule(self, index):
        self.num_rules = self.num_rules - 1
        # Antecedente
        self.layer['rules'].delete_rule(index)
        # Consecuente
        self.layer['consequent'].delete_rule(index)

    def clamp_rule_truth(self):
        new = torch.clamp(self.layer['rules'].truth_rules.data, min = 0.0, max = 1.0)
        self.layer['rules'].truth_rules.data = new
        
        new2 = torch.clamp(self.layer['rules'].sugeno_weights.data, min = 0.0, max = 1.0)
        self.layer['rules'].sugeno_weights.data = new2
        
        new = torch.clamp(self.layer['rules'].convex_weights.data, min = 0.0, max = 1.0)
        self.layer['rules'].convex_weights.data = new
        

    def forward(self, x):
        '''
            Forward pass: run x thru the five layers and return the y values.
            I save the outputs from each layer to an instance variable,
            as this might be useful for comprehension/debugging.
        '''
        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        #self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.weights = self.layer['htsk_ln_relu'](self.raw_weights, x.shape[1])
        #self.y_pred = self.layer['consequent'](x, self.weights)#MLP consequent
        self.rule_tsk = self.layer['consequent'](x)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))# multiplica y suma
        self.y_pred = y_pred.squeeze(2)
        return self.y_pred


# These hooks are handy for debugging:

def module_hook(label):
    ''' Use this module hook like this:
        m = AnfisNet()
        m.layer.fuzzify.register_backward_hook(module_hook('fuzzify'))
        m.layer.consequent.register_backward_hook(modul_hook('consequent'))
    '''
    return (lambda module, grad_input, grad_output:
            print('BP for module', label,
                  'with out grad:', grad_output,
                  'and in grad:', grad_input))


def tensor_hook(label):
    '''
        If you want something more fine-graned, attach this to a tensor.
    '''
    return (lambda grad:
            print('BP for', label, 'with grad:', grad))
