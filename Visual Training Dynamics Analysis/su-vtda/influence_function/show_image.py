import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_top_influencers(helpORharm,classes,training_data,influences,numTop=10):
    plt.figure(figsize=(8*numTop,2*numTop))
    selected = []
    for i in range(numTop):
        #print(a)
        a = helpORharm[i]
        plt.subplot(1,numTop,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(training_data.data[a], cmap=plt.cm.binary)
        selected.append(a)
    plt.show()

    labels = []

    fig, axes = plt.subplots(1, numTop, figsize=(40, 5),sharex=True,sharey=True)
    x = 0
    for i in selected:
        labels.append(classes[training_data.targets[i]])
        Label = classes[training_data.targets[i]]
        axes[x].set_title(f"Label: {Label}")
        inf_data = []
        for j in influences:
            inf_data.append(influences[j]["influence"][i])
        sns.scatterplot(ax=axes[x],x=range(1,10+1),y=inf_data,color="Purple",s=80)
        x+=1

    #print(labels)
    plt.show()