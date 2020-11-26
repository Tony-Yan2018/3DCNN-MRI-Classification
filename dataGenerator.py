import os
import cv2
import numpy as np
# from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import backend as K
import json
from imageProcess import imPreProcess
import random


CLASSES = ["RMI_OK", "RMI_INF", "RMI_DEG"]
SEQUENCES = ['STIR', 'T1']

Patients = dict()
dataSet = []

def generate_data(input_dir, size):
    PatientsSequences = {}  # patient's id mapped to STIR and T1 number
    with open('PatientsSequences.json', 'rb') as f:
        text = json.load(f)
        for key, value in text.items():
            PatientsSequences[int(key)] = value

    PatientsLocations = {}
    with open('PatientsLocations.json', 'rb') as f:
        text = json.load(f)
        for key, value in text.items():
            PatientsLocations[int(key)] = value

    X = []
    Y = []
    classes_list = os.listdir(input_dir)  # 3 image folders

    for c in classes_list:  # c: the name of each folder
        file_list = os.listdir(os.path.join(input_dir, c).replace('\\', '/'))  # list of files in one folder

        patient_set = set()  # unordered unique set
        for f in file_list:  # f: each picture in one folder
            if '.png' in f:
                pid = int(f[:4])  # patient id
                patient_set.add(pid)
        Patients[c] = patient_set  # stores the name of patients of one direct linked to direct name

    PatientsOK = {}
    min_cuts = 100
    # PatientsErrors = {}
    for c in classes_list:
        for pid in Patients[c]:
            m1 = PatientsSequences[pid]['STIR'][0][1]  # STIR photo number
            m2 = PatientsSequences[pid]['T1'][0][1]  # T1 photo number
            if m1 != m2:
                # l1 = PatientsLocations[pid]['T1'][0]
                # l2 = PatientsLocations[pid]['STIR'][0]
                # PatientsErrors[pid] = {'CL': c, 'T1': m1, 'STIR': m2, 'T1Cuts': l1, 'STIRCuts': l2}
                # print('Different length for', pid, '(', m1, '-', m2, ')', l1, l2)
                PatientsOK[pid] = 0
            else:
                PatientsOK[pid] = 1
            m = min(m1, m2)
            min_cuts = min(min_cuts, m)

    # PatientsErrors: each item is a dictionary which stores class, T1 pic number, STIR pic number, T1 of a patients sample
    # whose T1 pic number and STIR pic number are different(50).
    # PatientsOK: 1 if T1 pic number and STIR pic number are same; 0 if T1 pic number and STIR pic number are different(128).
    # min_cuts: the minimum pic number among all patient samples (11).

    # f = open("PatientsErrors.json", "w")
    # for pid in PatientsErrors.keys():
    #     f.write('{}:{}\n'.format(pid, PatientsErrors[pid]))
    # f.close()
    PatientsMap = dict()
    classMap = dict()
    for c in classes_list:
        y = [0] * len(CLASSES)
        y[CLASSES.index(c)] = 1
        classMap[c] = np.float32(y)
        for pid in Patients[c]:
            if PatientsOK[pid]:
                PatientsMap[pid] = c

    l = list(PatientsMap.items())
    random.shuffle(l)
    PatientsMap = dict(l)

    half_cut = int(min_cuts / 2)
    nb_frames = min_cuts * 2  # T1 + STIR num
    for key, value in PatientsMap.items():
        simage = np.zeros((size, size, nb_frames))
        i = 0
        for s in SEQUENCES:
            half_nb_imgs = int(PatientsSequences[key][s][0][1] / 2)
            first_img_id = PatientsSequences[key][s][0][0] + half_nb_imgs - half_cut

            for img_id in range(min_cuts):
                cut_id = first_img_id + img_id
                file = os.path.join(input_dir, value) + '/' + str(key).zfill(4) + '_' + s + '_' + str(cut_id).zfill(4) + '.png'
                try:
                    image = imPreProcess(file)
                    image = cv2.resize(image, (size, size))
                    simage[:, :, i] = image
                except Exception as e:
                    print(f'Error found at {file} & {cut_id}\n')

                i += 1

        X.append(simage)  # X to be the concatenation of STIR and T1
        Y.append(classMap[value])
        yield [np.asarray(X)[:, :, :,:,np.newaxis], np.asarray(Y)]