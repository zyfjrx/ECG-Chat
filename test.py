import os

import numpy as np
import pandas as pd

zero_shot_class = {"ptbxl_super_class": ['CD', 'HYP', 'MI', 'NORM', 'STTC'],
                   "ptbxl_sub_class": ['AMI', 'CLBBB', 'CRBBB', 'ILBBB', 'IMI', 'IRBBB', 'ISCA', 'ISCI',
                                       'ISC_', 'IVCD', 'LAFB/LPFB', 'LAO/LAE', 'LMI', 'LVH', 'NORM',
                                       'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'STTC', 'WPW', '_AVB'],
                   "ptbxl_form": ['ABQRS', 'DIG', 'HVOLT', 'INVT', 'LNGQT', 'LOWT', 'LPR', 'LVOLT',
                                  'NDT', 'NST_', 'NT_', 'PAC', 'PRC(S)', 'PVC', 'QWAVE', 'STD_',
                                  'STE_', 'TAB_', 'VCLVH'],
                   "ptbxl_rhythm": ['AFIB', 'AFLT', 'BIGU', 'PACE', 'PSVT', 'SARRH', 'SBRAD', 'SR',
                                    'STACH', 'SVARR', 'SVTAC', 'TRIGU'],
                   "cpsc2018": ["NORM", "AFIB", "1AVB", "LBBB",
                             "RBBB", "PAC", "PVC", "STD", "STE"]}



def load_ptbxl_diagnostics(path, is_train, sampling_rate=500):
    dataset_names = ["super_class", "sub_class", "form", "rhythm"]
    data = {}
    for dataset in dataset_names:
        name2index = {}
        for i, name in enumerate(zero_shot_class["ptbxl_" + dataset]):
            name2index[name] = i
        Y = pd.read_csv(os.path.join(path, f"ptbxl_database_{dataset}.csv"))

        test_fold = 10
        if is_train:
            Y = Y[Y.strat_fold != test_fold]
        else:
            Y = Y[Y.strat_fold == test_fold]

        if sampling_rate == 500:
            X_rel = Y.filename_hr.values
        else:
            X_rel = Y.filename_lr.values
        X = [os.path.join(path, x) for x in X_rel]
        y = Y.labels.values

        labels = [label.split(';') for label in y]

        targets = np.zeros((len(X), len(zero_shot_class["ptbxl_" + dataset])))
        for i in range(len(X)):
            for lbl in labels[i]:
                targets[i][name2index[lbl]] = 1

        data[dataset] = (X, targets)
    return data
if __name__ == '__main__':
    data = load_ptbxl_diagnostics("/Users/zhangyf/PycharmProjects/cfel/plus/ECG-Chat/data/ptb-xl", True)
    print(data)