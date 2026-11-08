import time
import torch
import sys
import numpy as np

# Importa o módulo original para usar as funções matemáticas idênticas
import funciones_entrenamiento222

def train_anfis(model, data_tr, data_val, optimizer, criterion, criterion2,
                epochs = 500, show_plots=False, limite_g = 2, ruta_modelo = './modelo_anfis.pth',
                paciencia = 1000, limiar_tolerancia = 1e-5):
    '''
    Treinamento ANFIS com o laço aberto de forma contínua, usando a matemática 
    exata e as funções de regularização da biblioteca original.
    '''
    errors = []  
    print('### Training for {} epochs, training size = {} cases'.
          format(epochs, data_tr.dataset.tensors[0].shape[0]))
    best_epoch = -1
    best_error = -1

    x_tr, y_tr = data_tr.dataset.tensors
    x_val, y_val = data_val.dataset.tensors
    
    time_start = time.time()
    last_time = time_start
    
    t_sin_mejora = 0
    best_error_train = torch.inf
    
    # --- VARIÁVEIS DE CONTROLE DA PARADA ANTECIPADA ---
    melhor_perda_val = float('inf')
    cont_sem_melhora_val = 0
    
    # O laço roda continuamente aqui fora, mantendo o estado do optimizer intacto!
    for t in range(1, epochs + 1):
        funciones_entrenamiento222.printProgressBar(0, len(data_tr), suffix = str(round(time.time() - last_time, 2)) + ' seconds')
        i = 1
        index = 0
        err = 0
        err4 = 0
        
        # Processa cada mini-batch:
        for x, y in data_tr:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss2 = criterion2(y_pred, y)
            
            # CHAMA A REGULARIZAÇÃO ORIGINAL EXATA DA BIBLIOTECA
            loss = loss + funciones_entrenamiento222.error_exploding_membership(model)
            
            err = err + float(loss)
            err4 = err4 + float(loss2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.clamp_rule_truth()
            
            train_loss = round(err / i, 5)
            mae_train_loss = round(err4 / i, 5)
            mape_train_loss = torch.mean(torch.abs((y - y_pred) / y))
            mse_train = criterion(y_pred.to('cpu'), y.to('cpu')).item()
            
            funciones_entrenamiento222.printProgressBar(i, len(data_tr), prefix = 'Training:', suffix = str(round(time.time() - last_time, 2)) + ' seconds,' + ' loss train: ' + str(train_loss) + '    ')
            index = index + x.shape[0]
            i = i + 1
            
        # Ajusta os coeficientes ao final da época usando a lógica original
        with torch.no_grad():
            model.fit_coeff(x_tr, y_tr)
            
        torch.cuda.empty_cache()
        loss_mae = torch.nn.L1Loss()
        
        # Predição de validação usando a função original
        y_val_pred = funciones_entrenamiento222.batch_predict(model, data_val)
        mse_val = criterion(y_val_pred.to('cpu'), y_val.to('cpu')).item()
        
        rmse_train2 = pow(mse_train, 0.5)
        rmse_train = pow(train_loss, 0.5)
        rmse_val = pow(mse_val, 0.5)
        mae_val = loss_mae(y_val.to('cpu'), y_val_pred.to('cpu')).item()
        mape_val = torch.mean(torch.abs((y_val.to('cpu') - y_val_pred.to('cpu')) / y_val.to('cpu')))
        
        # Salva o melhor modelo se bater o recorde (Lógica mantida do original)
        if (best_error == -1 or mse_val < best_error):
            best_error = mse_val
            best_epoch = t
            torch.save(model, ruta_modelo)
            
        errors.append(mse_val)
        now_time = time.time()
        
        # Print padrão de progresso
        print('\nepoch {:4d}: Train: MSE={:.5f}, RMSE={:.5f}, MAE={:.5f}, MAPE={:.5f}, Val: MSE={:.5f}, RMSE={:.5f}, MAE={:.5f}, MAPE={:.5f}, time = {:.2f} seconds'
              .format(t, train_loss, rmse_train, mae_train_loss, mape_train_loss, mse_val, rmse_val, mae_val, mape_val, now_time-last_time))
        print('MSE_Train= {:.5f}, RMSE= {:.5f}, err= {:.5f}' .format(mse_train, rmse_train2, err))
        last_time = now_time
        
        # --- LÓGICA DO CRITÉRIO DE PARADA ANTECIPADA (EARLY STOPPING) ---
        if mse_val < (melhor_perda_val - limiar_tolerancia):
            melhor_perda_val = mse_val
            cont_sem_melhora_val = 0 
        else:
            cont_sem_melhora_val += 1 
            
        if cont_sem_melhora_val >= paciencia:
            print(f"\n Parada antecipada ativada na época {t}!")
            print(f"O modelo ficou {paciencia} épocas seguidas sem melhora significativa na validação.")
            print(f"Melhor MSE de validação registrado: {melhor_perda_val:.6f}")
            break
        # ----------------------------------------------------------------
        
        if (train_loss < best_error_train):
            best_error_train = train_loss
            t_sin_mejora = 0
        elif(t_sin_mejora < limite_g - 1):
            t_sin_mejora = t_sin_mejora + 1
        else:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5
            best_error_train = torch.inf
            t_sin_mejora = 0
        
    now_time = time.time()
    print('End of Training, saved best model on ' + ruta_modelo)
    print('Best model: epoch ' + str(best_epoch))

def batch_predict(model, data_loader, text='Predicting:'):
    # Repassa diretamente para a função estável original da biblioteca
    return funciones_entrenamiento222.batch_predict(model, data_loader, text=text)

