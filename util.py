import pandas as pd
import numpy as np

def writeLayerWeightBias(model_layer, f_mode, path, thd_col_num, weight_name, bias_name):
    f = open(path, f_mode)
    count1 = 0
    countw = 0
    countb = 0
    for layer in model_layer:
        if hasattr(layer, "weight"):
            f.write(weight_name + "["+str(countw)+"]<<\n")
            countw += 1
            s = ""
            for i in range (layer.weight.shape[0]):
                for j in range(layer.weight.shape[1]):
                    if i == layer.weight.shape[0] -1 and j == layer.weight.shape[1] - 1:
                        delimiter = ";"
                    else:
                        delimiter = ","
                    if layer.weight[i][j].item() > 0 :
                        s += "{:<25}".format(" " + str(layer.weight[i][j].item()) + delimiter)
                    else:
                        s += "{:<25}".format(str(layer.weight[i][j].item()) + delimiter)
                    if (i * layer.weight.shape[1] + j) % thd_col_num == thd_col_num - 1:
                        s += "\n"
            f.write(s+"\n")
        if hasattr(layer, "bias"):
            f.write(bias_name + "["+str(countb)+"]<<\n")
            countb += 1
            s = ""
            for i in range (layer.bias.shape[0]):
                if i == layer.weight.shape[0] -1 and j == layer.weight.shape[1] - 1:
                    delimiter = ";"
                else:
                    delimiter = ","
                if layer.bias[i].item() > 0 :
                    s += "{:<25}".format(" " + str(layer.bias[i].item()) + delimiter)
                else:
                    s += "{:<25}".format(str(layer.bias[i].item()) + delimiter)
                if i % thd_col_num == thd_col_num - 1:
                    s += "\n"
            f.write(s+"\n")
        count1 += 1
    f.close()

def loadAnchors (input_file, output_file):
    input_data = pd.read_csv(input_file)
    output_data = pd.read_csv(output_file)

    anchors = np.array(input_data.copy())
    svs=np.array(output_data.copy())
    return anchors, svs

def loadInputOutputData (input_file1, output_file1, input_file2, output_file2):
    input1, out_put1 = loadAnchors (input_file1, output_file1) 
    input2, out_put2 = loadAnchors (input_file2, output_file2) 
    return input1, out_put1, input2, out_put2



