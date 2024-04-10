import pandas as pd
import numpy as np
from train_test_predict_affect_context import Modeller_affect_context
import warnings
import argparse

warnings.filterwarnings("ignore")

np.random.seed(1998) #for the sake of reproducibility


def compute_accuracy(dat1,dat2,prof_skeleton):
    acc_affect = 0
    acc_context = 0
    n_affect_true = 0
    n_context_true = 0

    for i in range(len(prof_skeleton)):
        if prof_skeleton["affect"][i] == True:

            n_affect_true += 1

            if dat1["affect"][i] == dat2["affect"][i]:

                acc_affect+=1


        if prof_skeleton["context"][i] == True:

            n_context_true += 1

            if dat1["context"][i] == dat2["context"][i]:

                acc_context+=1

    acc = (acc_affect/n_affect_true + acc_context/n_context_true)/2

    return acc

parser = argparse.ArgumentParser()
parser.add_argument("--data1", type=str, default='prof_data.csv', help = "path to the prof_data file of the competition")
parser.add_argument("--data2", type=str, default="SessionData-all.csv", help = "path to the SessionData-all file of the competition")
parser.add_argument("--prof_skeleton", type=str, default="prof_skeleton.csv", help = "path to the prof_skeleton file of the competition")

args = parser.parse_args()

Data1 = pd.read_csv(args.data1)
prof_skeleton = pd.read_csv(args.prof_skeleton)
# Data3 = pd.read_csv("test_data_bbdc.csv")
Data2 = pd.read_csv(args.data2)






Modeller = Modeller_affect_context(Data1, Data2, prof_skeleton )
skeleton = Modeller.train_predict_affect()

predicted_skeleton = Modeller.train_predict_context(skeleton)

prof_skeleton = pd.read_csv(args.prof_skeleton)




dat2 = pd.read_csv("prof_solution.csv")
dat1 = pd.read_csv("trial_111.csv")





accuracy1 = compute_accuracy(predicted_skeleton,dat2,prof_skeleton)
accuracy2 = compute_accuracy(dat1,dat2,prof_skeleton)

print(f"New accuracy from bbdc: {accuracy1}")
print(f"Old accuracy from bbdc: {accuracy2}")














