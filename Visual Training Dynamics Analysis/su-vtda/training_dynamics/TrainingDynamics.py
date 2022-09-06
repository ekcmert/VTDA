import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from scipy.stats import entropy

import random
from math import ceil
from . import AreaUnderTheMarginRanking
import warnings


class TrainingDynamics:
    def __init__(self, targets, classes=[], device="cpu"):

        self.epochs = 0

        self.device = device

        self.classes = classes
        self.class_num = len(classes)

        self.targets = targets

        self.prediction_history = torch.zeros(0, len(targets),
                                              self.class_num)  ## Epoch Number x Dataset Size x Class number
        self.prediction_history = self.prediction_history.to(self.device)  ## to GPU

        self.epoch_pred = torch.zeros(0, self.class_num)
        self.epoch_pred = self.epoch_pred.to(self.device)

        self.AUM = AreaUnderTheMarginRanking()

    def batch_accumulate(self, preds, batch, y):

        self.epoch_pred = torch.cat((self.epoch_pred, preds))

        img_id = range(batch, batch + len(y))
        img_id = torch.tensor(img_id).to(self.device)
        self.AUM.accumulate(preds, img_id, y)

    def epoch_accumulate(self):
        self.prediction_history = torch.cat((self.prediction_history, torch.unsqueeze(self.epoch_pred, 0)))
        self.AUM.accumulate_epoch()

        del self.epoch_pred

        self.epoch_pred = torch.zeros(0, self.class_num)  ###
        self.epoch_pred = self.epoch_pred.to(self.device)  ###

        self.epochs += 1

    def calculate(self):

        self.pred_hist_prob = torch.softmax(self.prediction_history,
                                            dim=2).detach().cpu().numpy()  ##turning logit values to probability values -> to cpu -> to numpy epoch num * dataset size * class num

        #   confidence_means=np.mean(self.pred_hist_prob,0)     ## getting confidence means from nmpy   !!!!## method changed
        #   std=np.std(self.pred_hist_prob,0)                   ## getting standart deviation from nmpy !!!!## method changed

        ### self.targets=training_data.targets.cpu().numpy()   ## getting targets from dataset

        self.correct_label_pred = np.take_along_axis(self.pred_hist_prob, np.atleast_3d(self.targets),
                                                     axis=2).squeeze().transpose()

        self.df = pd.DataFrame(self.targets, columns=["Label"])  ## creating dataframe for instance features
        self.df["Confidence Mean"] = self.correct_label_pred.mean(axis=1)  ## adding confidence mean
        self.df["Pred Std"] = self.correct_label_pred.std(axis=1)  ## adding std
        self.df["Last Pred"] = self.pred_hist_prob[self.epochs - 1].argmax(axis=1)  ## last prediction of model
        self.df["Last Pred Corr"] = (self.df["Last Pred"] == self.df["Label"])

        self.epoch_prediction = self.pred_hist_prob.argmax(2).transpose()  ## Which label predicted for every epoch
        self.correctness_matrix = (np.tile(self.targets, (self.epochs, 1)).transpose() == self.epoch_prediction).astype(
            int)  ## getting matrix of either prediction correct
        self.df["Correctness"] = self.correctness_matrix.mean(axis=1)

        self.label_pred_df = pd.DataFrame(
            np.concatenate([np.expand_dims(self.targets, axis=1), self.correct_label_pred],
                           axis=1))  ## getting label-prediction df
        self.label_pred_df.rename(columns={0: 'Label'}, inplace=True)

        ### binary correctness matrix -> dataframe with labels
        self.correctness_df = pd.DataFrame(
            np.concatenate([np.expand_dims(self.targets, axis=1), self.correctness_matrix], axis=1))
        self.correctness_df.rename(columns={0: 'Label'}, inplace=True)

        self.label_pred_grouped = self.label_pred_df.groupby("Label").mean().transpose()
        self.correctness_df_grouped = self.correctness_df.groupby("Label").mean().transpose()

        ##AUM
        num_AUM_history = (self.AUM.hist_delta_AUM).cpu().numpy()
        self.label_logits = num_AUM_history.transpose(2, 1, 0)[0]
        self.highest_non_logits = num_AUM_history.transpose(2, 1, 0)[1]
        self.df["AUM"] = self.label_logits.mean(axis=1) - self.highest_non_logits.mean(axis=1)

        ##Forgetting
        ### 1 forgot -1 learned
        action = (self.correctness_matrix[:, :-1] - self.correctness_matrix[:, 1:])

        self.forgots_in_epoch = (action == np.ones(action.shape)).astype(int)
        self.learns_in_epoch = (self.forgots_in_epoch - action).astype(int)

        self.df["Forget Nums"] = self.forgots_in_epoch.sum(axis=1)
        self.df["Learn Nums"] = self.learns_in_epoch.sum(axis=1) + self.correctness_matrix[:, 0]

        self.forgotten_items = self.df["Forget Nums"] != 0

        ##Entropy
        self.entropies = pd.DataFrame(entropy(self.pred_hist_prob, qk=None, base=None,
                                              axis=2).transpose())  ## got entropy of predictions for every instance in epochs
        self.entropies["Label"] = self.df["Label"]
        self.df["Entropy Mean"] = self.entropies.mean(axis=1)
        self.df["Entropy Std"] = self.entropies.std(axis=1)

        self.numerical_cols = ["Confidence Mean", "Pred Std", "Correctness", "AUM", "Entropy Mean", "Entropy Std"]

    ######### VISUALIZATION PART

    def plot_labeled(self, df, x_axis, y_axis, hue_, pal):

        def plotlabel(xvar, yvar, label):
            ax.text(xvar, yvar, label)

        fig = plt.figure(figsize=(14, 7))
        ax = sns.scatterplot(x=x_axis, y=y_axis, data=df, s=500, hue=hue_, palette=pal)

        df.apply(lambda x: plotlabel(x[x_axis], x[y_axis], self.classes[int(x["Label"])]), axis=1)

        if max(df[y_axis]) <= 1:
            ax.set_ylim(0, 1)

        plt.title(f"{y_axis} - {x_axis}", size=20)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)

    def plot_dist(self, forgets=False):
        if forgets:
            draw_df = self.df[self.forgotten_items]
        else:
            draw_df = self.df

        fig, axes = plt.subplots(ceil(len(self.numerical_cols) / 3), 3,
                                 figsize=(30, ceil(len(self.numerical_cols) / 3) * 5), constrained_layout=True)

        for i, col in enumerate(self.numerical_cols):
            axes[i // 3, i % 3].set_title(col)
            sns.histplot(ax=axes[i // 3, i % 3], data=draw_df[col])

    def plot_maps(self, hue_feature="Correctness", pal="RdYlGn", forgets=False):

        if forgets:
            draw_df = self.df[self.forgotten_items]
        else:
            draw_df = self.df

        fig, axes = plt.subplots(1, 2, figsize=(30, 7), sharex=True, sharey=True, constrained_layout=True)

        sns.scatterplot(ax=axes[0], data=draw_df, x="Pred Std", y="Confidence Mean", hue=hue_feature, palette="RdYlGn")
        sns.scatterplot(ax=axes[1], data=draw_df, x="Pred Std", y="Confidence Mean", hue="Label", palette="tab10")
        plt.show()

        fig, axes = plt.subplots(ceil(len(self.classes) / 5), 5, figsize=(30, len(self.classes) // 5 * 3), sharex=True,
                                 sharey=True, constrained_layout=True)
        for i in range(len(self.classes)):
            r, c = i // 5, i % 5
            sns.scatterplot(data=draw_df[draw_df["Label"] == i], ax=axes[r, c], x="Pred Std", y="Confidence Mean",
                            hue=hue_feature, palette=pal, legend=False)
            axes[r, c].set_title(self.classes[i])

    def plot_mean_std(self, forgets=False):
        if forgets:
            draw_df = self.df[self.forgotten_items]
        else:
            draw_df = self.df

        ####Whole dataset
        fig, axes = plt.subplots(1, 2, figsize=(30, 7), constrained_layout=True)

        axes[0].set_title("Dataset - Conf Mean")
        axes[1].set_title("Dataset - Conf Std")

        sns.histplot(ax=axes[0], data=draw_df, x="Confidence Mean")
        sns.histplot(ax=axes[1], data=draw_df, x="Pred Std", color="Red")

        plt.show()

        ####Whole dataset - but classes
        fig, axes = plt.subplots(1, 2, figsize=(30, 7), constrained_layout=True)

        axes[0].set_title("Classes - Conf Mean")
        axes[1].set_title("Classes - Conf Std")

        sns.kdeplot(ax=axes[0], data=draw_df, x="Confidence Mean", hue="Label",
                    palette="tab10")  ### plot 2 kde for confidence and std hu=label
        sns.kdeplot(ax=axes[1], data=draw_df, x="Pred Std", hue="Label",
                    palette="tab10")  ### plot 2 kde for confidence and std hu=label

        plt.show()

        ## Classes individually
        fig, axes = plt.subplots(ceil(len(self.classes) / 4), 8, figsize=(30, (len(self.classes))), sharex=False,
                                 sharey=False, constrained_layout=True)

        for i in range(len(self.classes)):
            r, c = i // 4, (i % 4) * 2
            axes[r, c].set_title(self.classes[i] + " - Conf Mean")
            axes[r, c + 1].set_title(self.classes[i] + " - Conf Std")
            sns.histplot(ax=axes[r, c], data=draw_df[draw_df["Label"] == i], x="Confidence Mean")
            sns.histplot(ax=axes[r, c + 1], data=draw_df[draw_df["Label"] == i], x="Pred Std", color="Red")

        plt.show()

    def plot_accuracy(self, forgets=False):

        if forgets:
            draw_df = self.df[self.forgotten_items]

            label_pred_grouped = self.label_pred_df[self.forgotten_items].groupby("Label").mean().transpose()
            correctness_df_grouped = self.correctness_df[self.forgotten_items].groupby("Label").mean().transpose()
        else:
            draw_df = self.df
            label_pred_grouped = self.label_pred_df.groupby("Label").mean().transpose()
            correctness_df_grouped = self.correctness_df.groupby("Label").mean().transpose()

            #### Binary Accuracy vs Confidence
        ##### first part

        fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharex=True, constrained_layout=True)

        fig.suptitle('History of Classes', size=40)

        legend = []
        for col in label_pred_grouped.columns:
            col = int(col)
            legend.append(self.classes[col])
            sns.lineplot(ax=axes[0], data=label_pred_grouped[col])
            axes[0].set_ylim(0, 1)

        axes[0].legend(legend)
        axes[0].set_title("Confidence of items by labels")

        for col in correctness_df_grouped.columns:
            sns.lineplot(ax=axes[1], data=correctness_df_grouped[col])

        axes[1].legend(legend)
        axes[1].set_title("Binary Accuracy of labels")

        plt.show()

        ##### second part

        fig, axes = plt.subplots(ceil(len(self.classes) / 5), 5, figsize=(24, 4 * ((len(self.classes) + 1) // 4)),
                                 sharex=True, sharey=True, constrained_layout=True)
        for num, col in enumerate(label_pred_grouped.columns):
            r, c = (num) // 5, num % 5
            sns.lineplot(ax=axes[r, c], data=label_pred_grouped[col])
            sns.lineplot(ax=axes[r, c], data=correctness_df_grouped[col])

            axes[r, c].legend(["Confidence Mean", "Binary Accuracy"])
            axes[r, c].set_title(self.classes[int(col)])

        plt.show()

    def plot_aum(self, training_data):
        selected = []
        plt.figure(figsize=(40, 10))
        for i in range(8):
            plt.subplot(1, 8, i + 1)
            num = random.randint(0, len(self.targets))
            selected.append(num)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(training_data.data[num], cmap=plt.cm.binary)
        plt.show()

        fig, axes = plt.subplots(1, 8, figsize=(40, 5), sharex=True, sharey=True, constrained_layout=True)
        for i in range(8):
            num = selected[i]
            plt.xlabel(self.classes[self.targets[num]])
            sns.scatterplot(ax=axes[i], x=range(1, self.epochs + 1), y=self.label_logits[num], color="Green")
            sns.scatterplot(ax=axes[i], x=range(1, self.epochs + 1), y=self.highest_non_logits[num], color="Red")

            Label = self.classes[self.df["Label"][num]]
            pred = self.classes[self.df["Last Pred"][num]]
            axes[i].set_title(f"Label: {Label}\nLast Pred: {pred}", size=20)

            axes[i].fill_between(
                range(1, self.epochs + 1), self.label_logits[num], self.highest_non_logits[num],
                where=(self.label_logits[num] > self.highest_non_logits[num]),
                interpolate=True, color="green", alpha=0.25,
                label="Positive"
            )

            axes[i].fill_between(
                range(1, self.epochs + 1), self.label_logits[num], self.highest_non_logits[num],
                where=(self.label_logits[num] <= self.highest_non_logits[num]),
                interpolate=True, color="red", alpha=0.25,
                label="Negative"
            )
        plt.show()

    def plot_forgets(self):

        forget_nums = list(self.df["Forget Nums"].unique())

        fig, axes = plt.subplots(ncols=2, figsize=(20, 10))
        fig.suptitle('Forgets', size=30)

        sns.lineplot(y=self.forgots_in_epoch.sum(axis=0), x=range(2, self.epochs + 1), ax=axes[0])
        sns.lineplot(y=self.learns_in_epoch.sum(axis=0), x=range(2, self.epochs + 1), ax=axes[1])

        axes[0].set_title("Number of forgets each epoch")
        axes[1].set_title("Number of learns each epoch")

        fig, axes = plt.subplots(4 + ceil(len(forget_nums) / 2), 2, figsize=(14, 20), constrained_layout=True)
        fig.suptitle('Forget Num Dists', size=30)

        ### THESE 3 FOR MAYBE INTO ONE LOOP

        sns.kdeplot(ax=axes[0, 0], data=self.df, x="Confidence Mean", hue="Forget Nums")

        legend = []
        for i in list(self.df["Forget Nums"].unique()):
            sns.kdeplot(ax=axes[0, 1], data=self.df[self.df["Forget Nums"] == i]["Confidence Mean"], shade=True)
            legend.append(i)
        axes[0, 1].legend(legend)
        axes[0, 1].set_title("Number of forgets-Correct Label Confidence Density")

        sns.kdeplot(ax=axes[1, 0], data=self.df, x="Pred Std", hue="Forget Nums")

        for i in list(self.df["Forget Nums"].unique()):
            sns.kdeplot(ax=axes[1, 1], data=self.df[self.df["Forget Nums"] == i]["Pred Std"], shade=True)
            legend.append(i)
        axes[1, 1].legend(legend)
        axes[1, 1].set_title("Number of forgets-Correct Label Pred Std")

        sns.kdeplot(ax=axes[2, 0], data=self.df, x="AUM", hue="Forget Nums")

        for i in list(self.df["Forget Nums"].unique()):
            sns.kdeplot(ax=axes[2, 1], data=self.df[self.df["Forget Nums"] == i]["AUM"], shade=True)
            legend.append(i)
        axes[2, 1].legend(legend)
        axes[2, 1].set_title("Number of forgets-AUM")

        sns.scatterplot(ax=axes[3, 0], data=self.df, x="Pred Std", y="Confidence Mean", hue="Forget Nums",
                        palette="RdYlGn_r")
        sns.scatterplot(ax=axes[3, 1], data=self.df, x="Pred Std", y="Confidence Mean", hue="Learn Nums",
                        palette="RdYlGn")

        for num, i in enumerate(forget_nums):  ######  when len(forget_nums)=2 0=> problem

            r, c = num // 2, num % 2
            sns.scatterplot(ax=axes[r + 4, c], data=self.df[self.df["Forget Nums"] == i], x="Pred Std",
                            y="Confidence Mean", hue="Correctness", palette="RdYlGn", alpha=0.6, )
            axes[r + 4, c].set_title(f"Forget Nums = {i}")
            axes[r + 4, c].set_xlim(left=0, right=0.5)
            axes[r + 4, c].set_ylim(bottom=0, top=1)

        plt.show()

    def plot_entropy(self, forgets=False):

        if forgets:
            draw_df = self.df[forgets]
        else:
            draw_df = self.df

        fig, axes = plt.subplots(4, 2, figsize=(20, 20), sharex=False, sharey=False, constrained_layout=True)
        sns.kdeplot(draw_df["Entropy Mean"], ax=axes[0, 0], shade=True)
        sns.kdeplot(draw_df["Entropy Std"], ax=axes[0, 1], shade=True)

        sns.scatterplot(ax=axes[1, 0], data=draw_df, x="Pred Std", y="Confidence Mean", hue="Entropy Mean",
                        palette="RdYlGn_r", alpha=0.6)
        self.entropies.groupby("Label").mean().transpose().plot.line(title="Entropy means of labels at each epoch",
                                                                     ax=axes[1, 1], )

        sns.scatterplot(ax=axes[2, 0], data=draw_df, x="Pred Std", y="Confidence Mean", hue="Entropy Std",
                        palette="RdYlGn", alpha=0.6)
        sns.scatterplot(ax=axes[2, 1], data=draw_df, x="AUM", y="Entropy Mean", hue="Correctness", palette="RdYlGn",
                        alpha=0.6)
        sns.scatterplot(ax=axes[3, 0], data=draw_df, x="Entropy Std", y="Entropy Mean", hue="Correctness",
                        palette="RdYlGn", alpha=0.6)
        sns.scatterplot(ax=axes[3, 1], data=draw_df, x="Entropy Std", y="Entropy Mean", hue="Label", palette="tab10",
                        alpha=0.6)

    def plot_classmaps(self, forgets=False):
        class_df = self.df.groupby("Label", as_index=False).mean()

        self.plot_labeled(class_df, "Pred Std", "Confidence Mean", "Correctness", "RdYlGn")
        self.plot_labeled(class_df, "Pred Std", "Confidence Mean", "Forget Nums", "RdYlGn_r")
        self.plot_labeled(class_df, "Pred Std", "Confidence Mean", "Entropy Mean", "RdYlGn")
        self.plot_labeled(class_df, "Pred Std", "Confidence Mean", "Entropy Std", "RdYlGn_r")
        self.plot_labeled(class_df, "Entropy Std", "Entropy Mean", "Correctness", "RdYlGn")
        self.plot_labeled(class_df, "Entropy Std", "Entropy Mean", "AUM", "RdYlGn")
        self.plot_labeled(class_df, "AUM", "Entropy Mean", "Correctness", "RdYlGn")
        self.plot_labeled(class_df, "AUM", "Entropy Std", "Forget Nums", "RdYlGn")

    def plot_all(self, training_data):
        self.plot_maps()
        self.plot_dist()
        self.plot_mean_std()
        self.plot_accuracy()
        self.plot_aum(training_data)
        self.plot_forgets()
        self.plot_entropy()
        self.plot_classmaps()

    def scatter_it(self, x_axis="", y_axis="", hue_metric='', title='', model='', show_hist=False):
        # Subsample data to plot, so the plot is not too busy.
        draw_df = self.df.sample(n=25000 if self.df.shape[0] > 25000 else len(self.df))

        #    # Normalize correctness to a value between 0 and 1.
        #    dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
        #    dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

        if not show_hist:
            fig, axs = plt.subplots(1, 1, figsize=(8, 4))
            ax0 = axs
        else:
            fig = plt.figure(figsize=(16, 10), )
            gs = fig.add_gridspec(2, 3, height_ratios=[5, 1])

            ax0 = fig.add_subplot(gs[0, :])

        ### Make the scatterplot.

        # Choose a palette.

        plot = sns.scatterplot(x=x_axis,
                               y=y_axis,
                               hue=hue_metric,
                               ax=ax0,
                               data=draw_df,
                               palette="RdYlGn",
                               s=30)

        if not show_hist:
            plot.legend(ncol=1, bbox_to_anchor=(1.01, 0.5), loc='center left', fancybox=True, shadow=True)
        else:
            plot.legend(fancybox=True, shadow=True, ncol=1)
        plot.set_xlabel(x_axis)
        plot.set_ylabel(y_axis)

        if show_hist:
            plot.set_title(f"{model}-{title} Data Map", fontsize=17)

            # Make the histograms.
            ax1 = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[1, 1])
            ax3 = fig.add_subplot(gs[1, 2])

            plott0 = draw_df.hist(column=[y_axis], ax=ax1, color='#622a87')
            plott0[0].set_title('')
            plott0[0].set_xlabel(y_axis)
            plott0[0].set_ylabel('density')

            plott1 = draw_df.hist(column=[x_axis], ax=ax2, color='teal')
            plott1[0].set_title('')
            plott1[0].set_xlabel(x_axis)

            plot2 = sns.histplot(x=hue_metric, data=draw_df, color='#86bf91', ax=ax3, bins=10)
            ax3.xaxis.grid(True)  # Show the vertical gridlines

            plot2.set_title('')
            plot2.set_xlabel(hue_metric)
            plot2.set_ylabel('')