import numpy as np
import PyQt5
#import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
#import tkinter
from matplotlib.ticker import PercentFormatter
import random



def precision_vs_recall(Rq,Aq):
    pvr = {}
    R_precision = 0
    r,lRq = 0,len(Rq)
    pvr["doc"] = []
    pvr["recall"] = []
    pvr["precision"] = []
    for i in range(len(Aq)):
        item = Aq[i]
        if item in Rq:
            pvr["doc"].append(item)
            r+=1
            pvr["recall"].append(r/lRq)
            pvr["precision"].append(r/(i+1))
        if i+1==lRq:
            R_precision = len(pvr["precision"])/lRq
    return pvr,R_precision

def print_result(data):
    df = pd.DataFrame(data)
    df = df.set_index(df.columns[0])
    print(df)
    return df

def get_precision_at_standard_11_recall_levels(data):
    
    recall = [0]
    precision = [data["precision"][0]]

    for i in np.arange(0.1,1.1,0.1):
        
        i = round(i,1)
        
        recall.append(i)
        if len(data["recall"])==0:
            precision.append(0)
        
        else:
            if data["recall"][0]>i:
                precision.append(data["precision"][0])
                
            elif data["recall"][0]==i:
                precision.append(data["precision"][0])
                data["recall"].pop(0)
                data["precision"].pop(0)
                
            else:
                data["recall"].pop(0)
                data["precision"].pop(0)
                precision.append(data["precision"][0])
           
    return recall,precision
    
    
def plot_graph(recall,precision):
    plt.plot(recall,precision,"bo-")
    plt.xlim(0,1.2)
    plt.ylim(0,1.2)
    plt.xlabel("Recall :")
    plt.ylabel("Precision : ")
    plt.title("Graph : Precision at 11 standard recall levels")
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()

    
print("\n --------------------------- R Precision ------------------------------- \n")

Rq_1=["d3","d5","d9","d25","d39","d44","d56","d71","d89","d123"]
Aq_1=["d123","d84","d56","d6","d8","d9","d511","d129","d187","d25","d38","d48","d250","d113","d3"]

res_pvr,r_precision = precision_vs_recall(Rq_1,Aq_1) 
result = print_result(res_pvr)

recall,precision = get_precision_at_standard_11_recall_levels(res_pvr)      
plot_graph(recall,precision)

print("\nR-Precision Set : ",r_precision)
Rq_2=["d3","d56","d129"]
Aq_2=["d123","d84","d56","d6","d8","d9","d511","d129","d187","d25","d38","d48","d250","d113","d3"]
res_pvr_1,r_precision_1 = precision_vs_recall(Rq_2,Aq_2) 
print("\n")
result_1 = print_result(res_pvr_1)

recall,precision = get_precision_at_standard_11_recall_levels(res_pvr_1)
plot_graph(recall,precision)


print("R-Precision : ",r_precision_1)

def get_r_precision(Rq,Aq):
    r_prec = []
    for rq,aq in zip(Rq,Aq):
        counter = 0
        len_rq = len(rq)
        for i in range(len_rq):
            item = aq[i]
            if item in rq:
                counter+=1
        r_prec.append(counter/len_rq)
    return r_prec

docs = ["d"+str(i) for i in range(1,21)]

Rq_for_10_queries, Aq_for_algo_A, Aq_for_algo_B = [],[],[]

# Generating n relavent documents for a given query
for _ in range(10):
    n = random.randint(6,9)
    li = random.sample(docs,n)
    Rq_for_10_queries.append(li)
    
# Generating n documents retrieved by Algoritm A for a given query
for _ in range(10):
    n = random.randint(13,18)
    li = random.sample(docs,n)
    Aq_for_algo_A.append(li)
    
# Generating n documents retrieved by Algoritm B for a given query
for _ in range(10):
    n = random.randint(13,18)
    li = random.sample(docs,n)
    Aq_for_algo_B.append(li)
    
qno = list(range(1,11))

RP_A = get_r_precision(Rq_for_10_queries,Aq_for_algo_A)
RP_B = get_r_precision(Rq_for_10_queries,Aq_for_algo_B)
RP_AB = [A-B for A,B in zip(RP_A,RP_B)] #interset of algo A and algo B

plt.bar(qno,RP_AB,color="red",width=0.7)
plt.xticks(qno)
plt.axhline(y = 0, color="black",linestyle = '-')
plt.xlabel("Query Number")
plt.ylabel("R Precision Algo A- Algo B")
plt.title("Precision Histogram")
plt.ylim(-1.5,1.5)
plt.show()

def harmonic_mean(data):
    F = 2/(1/data["recall"]+1/data["precision"])
    return F

def e_measure(data,b):
    E = 1-((1+b**2)/(b**2/data["recall"] + 1/data["precision"]))
    return E 


print("\n --------------------------- After copying Prev result ------------------------------- \n")

res_copy = result.copy()
result["harmonic_mean"] = harmonic_mean(res_copy)
result["e_measure(b=1)"] = e_measure(res_copy,1) #b=1
result["e_measure(b=2)"] = e_measure(res_copy,2) #b>1
result["e_measure(b=0.5)"] = e_measure(res_copy,0.5) #b<1

print(result)
print("\n --------------------------- After copying Prev result 1 ------------------------------- \n")

res_copy = result_1.copy()
result_1["harmonic_mean"] = harmonic_mean(res_copy)
result_1["e_measure(b=1)"] = e_measure(res_copy,1)
result_1["e_measure(b=2)"] = e_measure(res_copy,2)
result_1["e_measure(b=0.5)"] = e_measure(res_copy,0.5)

print(result_1)

