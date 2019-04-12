import csv, time, sys, os
import numpy as np
from random import randint

def load_data():
    data = []
    label = []
    jumlah_data = 42000
    jumlah_progress = 25
    os.system("clear")
    print("Loading Data : ")
    with open("input_data/train.csv", "r") as f:
        loaded = 0
        for i in csv.reader(f):
            data.append(i[1:])
            label.append(i[0])
            print("    {} Data Loaded | [".format(loaded), end="")
            for j in range(int((loaded/jumlah_data)*jumlah_progress)):
                print("=", end="")
            for j in range(jumlah_progress - int((loaded/jumlah_data)*jumlah_progress)):
                print(" ", end="")
            print("] {}% \r".format(int((loaded/jumlah_data)*100)), end="")
            sys.stdout.flush()
            loaded += 1
    print("\r")
    
    label.reverse()
    label.pop()
    label.reverse()

    data.reverse()
    data.pop()
    data.reverse()
    return data, label

def generate_test_data(train_data: list, label_data: list, amount: int) -> tuple:
    data_length = len(train_data)
    data = []
    label = []

    i = 0
    while i < data_length-1:
        datai = train_data[i]
        labeli = label_data[i]

        if randint(0, 100) >= 50:
            data.append(datai)
            label.append(labeli)

        i += 1
        if i >= data_length:
            i = 0

        if len(data) >= amount:
            break

    return data, label


def to_matrix(one_dimension_list: list, slices: int) -> list:
    return [one_dimension_list[i:i+slices] for i in range(0, len(one_dimension_list), slices)]