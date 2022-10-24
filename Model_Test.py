#python model_test.py -a Total(3).csv Total(3).csv Total(3).csv Total(3).csv Total(3).csv Total(3).csv  -p asdf.pkl asdf.pkl asdf.pkl asdf.pkl asdf.pkl
#model_test.py -a Fm_connection.csv  -p 1.pkl

#todo
#print outliers number
#print the outlier interval


#circles on outliers
#key , colors and number of outliers in the key
#run with mutliple files and output same window


#print FP
#acc after slided
#number of slided window
#number of -1 after sliding

from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

parser = ArgumentParser()
parser.add_argument("-p", "--pkl", dest="pkl_filename",help="Provide the .pkl from learned model",required=True)
parser.add_argument("-a", "--analyze", dest="analyze_filename",help="Csv file to analyze",required=True)

#parser.add_argument('-p', '--pkl-list', nargs='+', default=[],dest="pkl_filename")
#parser.add_argument('-a', '--analyze-list', nargs='+', default=[],dest="analyze_filename")
args = parser.parse_args()

csv_filename=args.analyze_filename
pkl_filename=args.pkl_filename

object_tuples=[]
names_tuples=[]


'''
if len(pkl_filenames) != len(csv_filenames):
    print('[+] The number csv file names is not equal to pickle ')
    exit(0)
if len(pkl_filenames) > 5 or len(csv_filenames)>5:
    exit(0)

color_list=[]
color_list1=['blue','rosybrown','black']
color_list2=['navy','red','orange']
color_list3=['lightsteelblue','darkred','gold']
color_list4=['mediumpurple','tomato','lawngreen']
color_list5=['indigo','coral','crimson']


def sliding_window():
    pass


#idxv=0
#for csv_filename,pkl_filename in zip(csv_filenames,pkl_filenames):
if idxv==0:
    color_list=color_list1
if idxv==1:
    color_list=color_list2
if idxv==2:
    color_list=color_list3
if idxv==3:
    color_list=color_list4
if idxv==4:
    color_list=color_list5
'''
clf = pickle.load(open(pkl_filename, 'rb'))
data = pd.read_csv(csv_filename, sep=',')
#print (data.columns.values)
inter_time = data['Interarrival Time'].to_list()
inter_range = data['Intervals'].to_list()
label_truths = data['label'].to_list()

inter_time_put=[[i] for i in inter_time if i == i]
inter_range_put=[[i] for i in inter_range if i == i]
label_truths_range_put=[[i] for i in label_truths if i == i]

LBLS=np.array(label_truths_range_put)
X=np.array(inter_time_put)
XR=np.array(inter_range_put)

y_pred_outliers = clf.predict(X)
lofs_index = np.where(y_pred_outliers==-1)
values = X[lofs_index]

unslided=[]
unslided_map=[]
slided=[]

print ("\n[+] FILE ",csv_filename)
print("\nTotal number of outliers: ",len(XR[lofs_index]),'\n')
print("Outlier intervals\n")
for r,x in zip(XR[lofs_index],LBLS[lofs_index]):
    print ('>> ',r[0],x[0])
    unslided.append(r[0])
    unslided_map.append(x[0])
print('\n\n')


for j in range(len(unslided_map)):
    if j==len(unslided_map)-2:
        slided.append(unslided_map[-2])
        slided.append(unslided_map[-1])
        break
    cur_item=unslided_map[j]
    next_item=unslided_map[j+1]
    next_item2=unslided_map[j+2]

    if cur_item==-1 and next_item == -1 and next_item2 == -1:
        slided.append(-1)
    else:
        slided.append(1)

print('unslided: ',unslided_map,len(unslided_map))
print('slided: ',slided,len(slided))

print('\nAfter applying sliding window\n')

for a,b in zip(slided,unslided):
    if a == -1:
        print (b)




idx=[]
val=[]
for i in lofs_index[0]:
    idx.append(i)
    val.append(inter_time_put[i][0])
#print ('>>>>>>>>> ',radius)



plt.title("Novelty Detection with Local Outlier Factor")
blue=plt.scatter([x for x in range(len(inter_time))], inter_time,s=9,color='blue')
red=plt.scatter(idx,val, color='red',s=9)
#black=plt.scatter(idx,val, s=200, facecolors='none', edgecolors=color_list[2])

object_tuples.append(blue)
object_tuples.append(red)
#object_tuples.append(black)
names_tuples.append('Normals')
names_tuples.append('Outliers')
#names_tuples.append(csv_filename+' outlier points')



plt.legend(tuple(object_tuples),tuple(names_tuples),
           title="Number of Outliers: "+str(len(val)),
           #loc='upper left',
           fontsize=8)

plt.xlabel('Time Windows')
plt.ylabel('Number of Packets')
plt.show()
