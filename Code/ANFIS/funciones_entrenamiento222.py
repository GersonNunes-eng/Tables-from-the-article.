#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some simple functions to supply data and plot results.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""
import numpy
import torch
import time
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

dtype = torch.float

# Función para regularizar los parámetros de las funciones de pertenencia para evitar que excedan 1
# Funciones probadas: Gaussiana
def error_exploding_membership(model, delta = 10):
    vars_list = model.get_membership_params()
    error = 0.0
    for var in vars_list:
        for memb in var:
            for val in memb:
                if (val <= 1 and val >= 0):
                    error = error + val.abs() * 0
                elif (val < 0):
                    error = error + val.abs() * delta
                else: # val > 1
                    error = error + (val.abs() - 1) * delta
    return error

# En principio la idea es reducir el consumo de RAM/VRAM durante la predicción haciendo que la predicción también funcione por baches
def batch_predict(model, data_loader, text = 'Predicting:', device = 'cpu'):
    time_start = time.time()
    i = 0
    printProgressBar(0, len(data_loader), prefix = text, suffix = str(round(time.time() - time_start, 2)) + ' seconds')
    y_pred = torch.tensor([], device = 'cpu')
    for x, y in data_loader:
        with torch.no_grad():
            pred = model(x.to(device)).detach().to('cpu')
        y_pred = torch.cat((y_pred, pred), 0)
        i = i + 1
        printProgressBar(i, len(data_loader), prefix = text, suffix = str(round(time.time() - time_start, 2)) + ' seconds')
    return y_pred

# Funcion que sirve para mostrar una barra de progreso
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 75, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    #if iteration == total: 
    #    print()

''' Funcion utilizada para obtener los parametros utilizados
    para construir las variables difusas del modelo (ANFIS) '''
def get_fuzzy(model, var = 0):
    n_var = 'x' + str(var)
    keys = list(model.layer['fuzzify'].varmfs[n_var].mfdefs.keys())
    res = []
    for i in range(len(keys)):
        mdefs = model.layer['fuzzify'].varmfs[n_var].mfdefs[keys[i]]
        print([float(mdefs.mu), float(mdefs.sigma)])
        res.append(mdefs)
    return res

def print_rules(model, nombre_vars, null_values = True, etiquetas = None):
    comp = int(null_values)
    component_rules = model.get_rules()
    wc = abs(model.layer['consequent']._coeff[:, 0, :])
    wc_m = wc.mean(axis = 1)
    rules = []
    
    for i in range(component_rules.shape[0]):
        aux = 'Rule ' + str(i) + ': If '
        first = True
        for j in range(len(nombre_vars)):
            if (not(null_values) or int(component_rules[i,j]) != 0):
                if (j > 0 and not(first)):
                    aux = aux + ' and '
                first = False
                if (etiquetas != None):
                    aux = aux + nombre_vars[j] + ' is ' + etiquetas[int(component_rules[i,j]) - comp]
                else:
                    aux = aux + nombre_vars[j] + ' is mf' + str(int(component_rules[i,j]) - comp)
        # aux = aux + ' then y(x) wr ~ ' + str(abs(round(float(wc_m[i]), 2)))
        aux = aux + ' then y(x) with w' + str(i) + ' = ' + str(wc[i].detach().to('cpu').numpy().round(2))
    #     print(aux)
        rules.append(aux)
    return(rules, wc, wc_m)

    
# Función creada para utilizar datos de validación y devolver el modelo con el error de validación más bajo
def train_anfis(model, data_tr, data_val, optimizer, criterion, criterion2,
                     epochs = 500, show_plots=False, limite_g = 2, ruta_modelo = './modelo_anfis.pth'):
    '''
        Train the given model using the given (x,y) data.
    '''
    errors = []  # Keep a list of these for plotting afterwards
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print('### Training for {} epochs, training size = {} cases'.
          format(epochs, data_tr.dataset.tensors[0].shape[0]))
    best_epoch = -1
    best_error = -1
    best_10 = [-1, -1]# val mse, val rmse

    x_tr, y_tr = data_tr.dataset.tensors
    x_val, y_val = data_val.dataset.tensors
    
    time_start = time.time()
    last_time = time_start
    
    # Para actualizar el gradiente en caso de no conseguir ninguna mejora
    t_sin_mejora = 0
    limite_g = limite_g
    best_error_train = torch.inf
    
    for t in range(1, epochs + 1):
        printProgressBar(0, len(data_tr), suffix = str(round(time.time() - last_time, 2)) + ' seconds')
        i = 1
        index = 0
        err = 0
        err4 = 0
        # Process each mini-batch in turn:
        for x, y in data_tr:
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred, y)
            loss2 = criterion2(y_pred, y)
            # Regularización de funciones de pertenencia
            loss = loss + error_exploding_membership(model)
            # Zero gradients, perform a backward pass, and update the weights.
            err = err + float(loss)
            err4 = err4 + float(loss2)
            optimizer.zero_grad()
            loss.backward()# retain_graph=True
            optimizer.step()
            model.clamp_rule_truth()
            #limit_parameters_membership(model)
            train_loss = round(err / (i), 5)
            mae_train_loss = round(err4 / (i), 5)
            mape_train_loss = torch.mean(torch.abs((y - y_pred) / y))
            mse_train = criterion(y_pred.to('cpu'), y.to('cpu')).item()
            printProgressBar(i, len(data_tr), prefix = 'Training:', suffix = str(round(time.time() - last_time, 2)) + ' seconds,' + ' loss train: ' + str(train_loss) + '    ')
            index = index + x.shape[0]
            i = i + 1
        # Epoch ending, so now fit the coefficients based on all data:
        with torch.no_grad():
            model.fit_coeff(x_tr, y_tr)
        # Get the error rate for the whole batch:
        #y_tr_pred = batch_predict(model, data_tr, text = 'Pred Train:')
        torch.cuda.empty_cache()
        loss_mae = torch.nn.L1Loss()
        y_val_pred = batch_predict(model, data_val)
        mse_val = criterion(y_val_pred.to('cpu'), y_val.to('cpu'))
        rmse_train2 = pow(mse_train, 0.5)
        rmse_train = pow(train_loss, 0.5)
        rmse_val = pow(mse_val, 0.5)
        mae_val = loss_mae(y_val.to('cpu'), y_val_pred.to('cpu')).item()
        mape_val = torch.mean(torch.abs((y_val.to('cpu') - y_val_pred.to('cpu')) / y_val.to('cpu')))
        
        if (best_error == -1 or mse_val < best_error):
            #print('Actualizacion')
            best_error = mse_val
            best_epoch = t
            torch.save(model, ruta_modelo)
            
        errors.append(mse_val)
        now_time = time.time()
        # Print some progress information as the net is trained:
        
        print('epoch {:4d}: Train: MSE={:.5f}, RMSE={:.5f}, MAE={:.5f}, MAPE={:.5f}, Val: MSE={:.5f}, RMSE={:.5f}, MAE={:.5f}, MAPE={:.5f}, time = {:.2f} seconds                                                             '
              .format(t, train_loss, rmse_train, mae_train_loss, mape_train_loss, mse_val, rmse_val, mae_val, mape_val, now_time-last_time))
        print('MSE_Train= {:.5f}, RMSE= {:.5f}, err= {:.5f}' .format(mse_train, rmse_train2, err))
        last_time = now_time
        
        if (train_loss < best_error_train):
            best_error_train = train_loss
            t_sin_mejora = 0
        elif(t_sin_mejora < limite_g - 1):
            t_sin_mejora = t_sin_mejora + 1
        else:
            #print('Actualizacion del learning rate: lr * 0.5')
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5
            best_error_train = torch.inf
            t_sin_mejora = 0
        
    now_time = time.time()
    print('End of Training, saved best model on ' + ruta_modelo)
    print('Best model: epoch ' + str(best_epoch))
    # End of training, so graph the results: