#python model_train.py -n normal-traffic.csv -k 30 -o asdf.pkl


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import *
#from sklearn.externals import joblib
from sklearn.neighbors import LocalOutlierFactor
from argparse import ArgumentParser
#import joblib
import pickle

# Save the trained model as a pickle string.

parser = ArgumentParser()
parser.add_argument("-n", "--normal", dest="norm_filename",help="Csv file to read normal data from",required=True)
parser.add_argument("-k", "--kval", dest="k_value",help="Please provide n_neighbors value",required=True)
parser.add_argument("-o", "--savefile", dest="save_file",help="Provide the filename to save the learned model to, eg filename.pkl",required=True)
args = parser.parse_args()


def split_two(lst, ratio=[0.5, 0.5]):
    assert(np.sum(ratio) == 1.0)  # makes sure the splits make sense
    train_ratio = ratio[0]
    # note this function needs only the "middle" index to split, the remaining is the rest of the split
    indices_for_splittin = [int(len(lst) * train_ratio)]
    train, test = np.split(lst, indices_for_splittin)
    return train, test



filename=args.norm_filename
k_value=int(args.k_value)
save_model=args.save_file

#filename="normal-traffic.csv"
data = pd.read_csv(filename, sep=',')
inter_time = data['Interarrival Time'].to_list()
train_data,test_data=split_two(inter_time,[0.7, 0.3])


X=np.array([[i] for i in train_data if i == i])
X2=np.array([[i] for i in test_data if i == i])


clf = LocalOutlierFactor(n_neighbors=k_value, novelty=True, contamination=0.1)
clf.fit(X)



#saved_model = pickle.dumps(clf)
#joblib.dump(clf, save_model)

pickle.dump(clf, open(save_model, 'wb'))
print(">> Trained novelty LOF Model: ",save_model," with k = ",k_value)




