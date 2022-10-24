#python model_validate.py -p total.pkl -lb Total(3).csv
#True Pos, False Pos

from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import *

parser = ArgumentParser()
parser.add_argument("-p", "--pkl", dest="pkl_filename",help="Provide the .pkl from learned model",required=True)
parser.add_argument("-lb", "--analyze", dest="labeled_csv",help="Labeled csv file to analyze",required=True)
args = parser.parse_args()


csv_filename=args.labeled_csv
pkl_filename=args.pkl_filename

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    True_Pos=[]
    False_Pos=[]
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TN += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
           False_Pos.append(i)
        if y_actual[i]==y_hat[i]==-1:
           TP += 1
           True_Pos.append(i)
        if y_hat[i]==-1 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN,True_Pos,False_Pos

#norm_filename=args.normal_filename

clf = pickle.load(open(pkl_filename, 'rb'))
#data = pd.read_csv(norm_filename, sep=',')

lbl_data = pd.read_csv(csv_filename, sep=',')
ground_truth = lbl_data['label'].to_list()
inter_time = lbl_data['Interarrival Time'].to_list()
inter_time_put=[[i] for i in inter_time if i == i]
X=np.array(inter_time_put)

#lbl_inter_time_put=[[i] for i in lbl_inter_time if i == i]
#XL=np.array(lbl_inter_time_put)

y_pred_test = clf.predict(X)


#exit(0)


acc_score=metrics.accuracy_score(ground_truth, y_pred_test)
f1_scores=f1_score(ground_truth, y_pred_test)
recalls=recall_score(ground_truth, y_pred_test)
precisions=precision_score(ground_truth, y_pred_test)

print ("Accuracy:", acc_score)
print ("f1_scores: ", f1_scores)
print ("precisions: ", precisions)
print ("recalls: ", recalls)
#tn, fp, fn, tp = confusion_matrix(ground_truth,y_pred_test, labels=[1, -1]).ravel()


print()
TP, FP, TN, FN, True_Pos, False_Pos = perf_measure(y_pred_test,ground_truth)
print ('True Negative: ',TN)
print ('False Postives: ',FP)
print ('False Negative: ',FN)
print ('True Postives: ',TP)
print()
print ("TP Values: ",[inter_time[i] for i in True_Pos])
print ("FP Values: ",[inter_time[i] for i in False_Pos])
print()
#print ('>> True Negative: ',tn)
#print ('>> False Postives: ',fp)
#print ('>> False Negative: ',fn)
#print ('>> True Postives: ',tp)
#print ()

print(classification_report(ground_truth, y_pred_test))







