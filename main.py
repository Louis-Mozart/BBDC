import pandas as pd
import numpy as np
from train_test_predict_affect_context import Modeller_affect_context
from score import compute_accuracy
import warnings
import argparse
warnings.filterwarnings("ignore")

np.random.seed(1998) 


parser = argparse.ArgumentParser()
parser.add_argument("--data1", type=str, default='prof_data.csv', help = "path to the prof_data file of the competition")
parser.add_argument("--data2", type=str, default="SessionData-all.csv", help = "path to the SessionData-all file of the competition")
parser.add_argument("--prof_skeleton", type=str, default="prof_skeleton.csv", help = "path to the prof_skeleton file of the competition")
parser.add_argument("--prof_solution", type=str, default="prof_solution.csv", help= "path to the prof_solution file of the competition")
parser.add_argument("--evaluate_score", type=str, default="no", help="if yes, we compute the bbdc score based on the solution file, \
                    hence make sure  you have the prof_solution.csv file in this repo", choices=["yes","no"])
parser.add_argument("--save_our_solution", type =str, default = "no", help = "if yes, we save our predictions file as csv", choices=["yes", "no"])

args = parser.parse_args()


Data1 = pd.read_csv(args.data1)
prof_skeleton = pd.read_csv(args.prof_skeleton)
Data2 = pd.read_csv(args.data2)



Modeller = Modeller_affect_context(Data1, Data2, prof_skeleton )
skeleton = Modeller.train_predict_affect()

predicted_skeleton = Modeller.train_predict_context(skeleton)


if args.save_our_solution == "yes":

    predicted_skeleton.to_csv("KTLM_solution.csv", index=False)


if args.evaluate_score == "yes":

    prof_solution = pd.read_csv(args.prof_solution)
    # dat1 = pd.read_csv("trial_111.csv")
    prof_skeleton = pd.read_csv(args.prof_skeleton)

    accuracy1 = compute_accuracy(predicted_skeleton,prof_solution,prof_skeleton)
    # accuracy2 = compute_accuracy(dat1,prof_solution,prof_skeleton)

    print(f"Accuracy from bbdc: {accuracy1}")
    # print(f"Old accuracy from bbdc: {accuracy2}")














