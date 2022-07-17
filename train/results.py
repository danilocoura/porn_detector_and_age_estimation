import pandas as pd
import os

def save_log(log,approach,parameter,iteration):
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(os.path.join("results",approach)):
        os.mkdir(os.path.join("results",approach))
    if not os.path.exists(os.path.join("results",approach,"{0}".format(parameter))):
        os.mkdir(os.path.join("results",approach,"{0}".format(parameter)))
    text_file = open(os.path.join("results",approach,"{0}".format(parameter),"{0}_param{1}_it{2}.txt".format(approach, parameter, iteration)), "w")
    n = text_file.write(log)
    text_file.close()

def save_csv(df,approach,parameter):
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(os.path.join("results",approach)):
        os.mkdir(os.path.join("results",approach))
    if not os.path.exists(os.path.join("results",approach,"{0}".format(parameter))):
        os.mkdir(os.path.join("results",approach,"{0}".format(parameter)))
    df.to_csv(os.path.join("results",approach,"{0}".format(parameter),"{0}_param{1}.csv".format(approach, parameter)))