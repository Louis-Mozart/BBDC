# TEAM KTLM

This repository contains our solution to the BBDC 2024. Team KTLM consists of two members: [Leonie Sieger](https://www.dice-research.org/LeonieSieger) and [Louis Mozart KAMDEM](https://dice-research.org/LouisMozartKAMDEM).

## Installation

To use our solution, clone this repository to your local computer:

```bash
git clone https://github.com/Louis-Mozart/BBDC.git
```

Before running the project, ensure you have all the dependencies installed. You can install them using `pip` and the provided `requirements.txt` file. 

```bash
pip install -r requirements.txt
```

## How to Run?

To run and save our model, ensure you have all the required data files (those provided by BBDC) in this directory, then execute:

```bash
python3 main.py --save_our_solution yes
```
This will then save our solution in the current directory with the name `KTLM_solution.csv`

If the data files are located in another directory, specify the paths using the `--data1`, `--data2`, and `--prof_skeleton` flags:

```bash
python3 main.py --data1 path_to_prof_data.csv --data2 path_to_SessionData-all.csv --prof_skeleton path_to_prof_skeleton.csv --save_our_solution yes
```
Example

```bash 
python3 main.py --data1  /home/dice/Desktop/BBDC/prof_data.csv --data2  /home/dice/Desktop/BBDC/SessionData-all.csv --prof_skeleton /home/dice/Desktop/BBDC/prof_skeleton.csv  --evaluate_score yes
```


To compute the score according to the BBDC solution, provide the `prof_solution.csv` file and execute:

```bash 
python3 main.py --data1 prof_data.csv --data2 SessionData-all.csv --prof_skeleton prof_skeleton.csv --evaluate_score yes --prof_solution prof_solution.csv --save_our_solution yes
```
If the data are not in the same directory, replace `prof_data.csv`, `SessionData-all.csv`, `rof_skeleton.csv` and `prof_solution.csv` with their respective paths.

## Hyperparameters

While we were unable to reproduce the score achieved on the BBDC dashboard, we tuned our model to produce better and reproducible results. The hyperparameters we used for our model are printed during the preprocessing, training and postprocessing so please have a look on the terminal when the model is running.
