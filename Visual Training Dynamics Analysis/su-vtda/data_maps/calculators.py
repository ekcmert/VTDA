import torch
import pandas as pd
import numpy as np

def calculate_confidence(prediction_history,targets,df):
    confidence_means = torch.mean(torch.softmax(prediction_history, dim=2), 0)
    df["confidence"] = np.take_along_axis(confidence_means.detach().cpu().numpy(), np.expand_dims(targets, axis=1),axis=1)

def calculate_variability(prediction_history,targets,df):
    std = torch.std(torch.softmax(prediction_history, dim=2), 0)
    df["variability"] = np.take_along_axis(std.detach().cpu().numpy(), np.expand_dims(targets, axis=1), axis=1)

def calculate_correctness(prediction_history,epochs,targets,df):
    epoch_pred = prediction_history.argmax(2).permute(1, 0).cpu().numpy()
    targetxE = np.tile(targets, (epochs, 1)).transpose()  ### label matrix x epoch num
    print(epoch_pred)
    correctness_matrix = (targetxE == epoch_pred).astype(int)
    df["correctness"] = correctness_matrix.sum(axis=1) / (epochs)


