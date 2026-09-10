import sys
from collections import Counter, defaultdict
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.decomposition import PCA
import matplotlib.colors as mcolours
from sklearn.linear_model import LinearRegression

def main():

    df = pd.read_csv("raid_results/raid_clean_model_sweep_summary.csv")

    print(df)

    models = df["model"].unique()
    attacks = df["attack"].unique()
    decodings = df["decoding"].unique()
    repetition_penalties = df["repetition_penalty"].unique()

    cols = df.columns[8:]
    ncols = len(cols)
    print(f"num cols is {ncols}")    

    # cm = plt.get_cmap('tab20')
    # colours = list(cm.colors)
    # for i in range(int(ncols/2)):
    #     col_1 = cols[2*i]
    #     col_2 = cols[2*i + 1]
    #     data = df[[col_1, col_2]].values
        
    #     fig = plt.figure(figsize = (20, 20))
    #     gs = fig.add_gridspec(2,2)
    #     ax = []

    #     for i in range(2):
    #         temp_ax = []
    #         for j in range(2):
    #             temp_ax.append(fig.add_subplot(gs[i,j]))
    #         ax.append(temp_ax)

    #     for i, model in enumerate(models):
    #         mask = df["model"] == model
    #         ax[0][0].scatter(data[mask, 0], data[mask,1], color = colours[i], label = model)
    #     ax[0][0].legend(title="Models")
    #     ax[0][0].set_xlabel(col_1)
    #     ax[0][0].set_ylabel(col_2)

    #     for i, attack in enumerate(attacks):
    #         mask = df["attack"] == attack
    #         ax[0][1].scatter(data[mask, 0], data[mask,1], color = colours[i], label = attack)
    #     ax[0][1].legend(title="Attacks")
    #     ax[0][1].set_xlabel(col_1)
    #     ax[0][1].set_ylabel(col_2)
            
    #     for i, decoding in enumerate(decodings):
    #         mask = df["decoding"] == decoding
    #         ax[1][0].scatter(data[mask, 0], data[mask,1], color = colours[i], label = decoding)
    #     ax[1][0].legend(title="Decodings")
    #     ax[1][0].set_xlabel(col_1)
    #     ax[1][0].set_ylabel(col_2)

    #     for i, repeat in enumerate(repetition_penalties):
    #         mask = df["repetition_penalty"] == repeat
    #         ax[1][1].scatter(data[mask, 0], data[mask,1], color = colours[i], label = repeat)
    #     ax[1][1].legend(title="Repetition penalities")
    #     ax[1][1].set_xlabel(col_1)
    #     ax[1][1].set_ylabel(col_2)

    #     plt.savefig(f"../test_plot_{col_1}.png", bbox_inches = "tight")

    mean_cols = ["auroc_mean", "tpr_at_fpr_5pct_mean", "tpr_at_fpr_1pct_mean"]

    num_models = len(models)
    num_attacks = len(attacks)
    num_decodings = len(decodings)
    num_repeats = len(repetition_penalties)

    all_vars = []
    for model in models:
        all_vars.append(f"model is {model}")

    for attack in attacks:
        all_vars.append(f"attack is {attack}")

    for decoding in decodings:
        all_vars.append(f"decoding is {decoding}")

    for repeat in repetition_penalties:
        all_vars.append(f"repetition penalty is {repeat}")

    total_length = num_models + num_attacks + num_decodings + num_repeats
    for col in mean_cols:
        target_vals = df[col].values
        for j in range(4):
            data_array = []
            for i, model in enumerate(df["model"]):
                temp_array = [0 for i in range(total_length)]
                model_hot_ind = int(np.where(models == model)[0][0])
                temp_array[model_hot_ind] = 1
                if j > 0:
                    attack_hot_ind = int(np.where(attacks == df["attack"].iloc[i])[0][0])
                    temp_array[num_models + attack_hot_ind] = 1
                if j > 1:
                    decoding_hot_ind = int(np.where(decodings == df["decoding"].iloc[i])[0][0])
                    temp_array[num_models + num_attacks + decoding_hot_ind] = 1
                if j > 2:
                    repeat_hot_ind = int(np.where(repetition_penalties == df["repetition_penalty"].iloc[i])[0][0])
                    temp_array[num_models + num_attacks + num_decodings + repeat_hot_ind] = 1

                data_array.append(temp_array)
        # print(data_array)
            reg = LinearRegression().fit(data_array, target_vals)
            print(f"For accuracy measure {col}")
            print(f"The score is {reg.score(data_array, target_vals)}")
            print(f"The intercept is {reg.intercept_}")
            for i, var in enumerate(all_vars):
                print(f"For variable {var} the coefficient is {reg.coef_[i]}")
    return 

if __name__ == "__main__":
    main()