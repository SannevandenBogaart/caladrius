"""
Adapted from: https://github.com/rodekruis/caladrius/blob/handle_imbalance/caladrius/evaluation_metrics_classification.py
"""
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

def harmonic_score(scores):
    """
    Calculate the harmonic mean of a list of scores
    Args:
        scores (list): list of scores
    Returns:
        harmonic mean of scores
    """
    return len(scores) / sum((c + 1e-6) ** -1 for c in scores)


def gen_score_overview(preds_filename, binary=False, switch=False):
    """
    Generate a dataframe with several performance measures
    Args:
        preds_filename: name of file where predictions are saved
    Returns:
        score_overview (pd.DataFrame): dataframe with several performance measures
        df_pred (pd.DataFrame): dataframe with the predictions and true labels
    """

    if not binary:
        damage_mapping = {
            "0": "No damage",
            "1": "Minor damage",
            "2": "Major damage",
            "3": "Destroyed",
        }

    else:
        damage_mapping = {
            "0": "No damage",
            "1": "Damage",
        }

    preds_file = open(preds_filename)
    lines = preds_file.readlines()[1:]
    pred_info = []
    for l in lines:
        split_list = l.rstrip().split(" ")
        if len(split_list) == 3:
            pred_info.append(split_list)
    df_pred = pd.DataFrame(pred_info, columns=["OBJECTID", "label", "pred"])
    df_pred.label = df_pred.label.astype(int)
    df_pred.pred = df_pred.pred.astype(int)

    if binary and switch:
        df_pred.label = abs(df_pred.label - 1)
        df_pred.pred = abs(df_pred.pred - 1)

    preds = np.array(df_pred.pred)
    labels = np.array(df_pred.label)
    unique_labels = np.unique(labels)
    unique_preds = np.unique(preds)
    damage_labels = [i for i in list(map(int, damage_mapping.keys())) if i != 0]
    damage_present = any(
        x in damage_labels for x in list(set().union(unique_labels, unique_preds))
    )
    report = classification_report(
        labels,
        preds,
        digits=3,
        output_dict=True,
        labels=list(map(int, damage_mapping.keys())),
        zero_division=1,
    )

    score_overview = pd.DataFrame(report).transpose()
    score_overview = score_overview.append(pd.Series(name="harmonized avg"))

    score_overview.loc["harmonized avg", ["precision", "recall", "f1-score"]] = [
        harmonic_score(r)
        for i, r in score_overview.loc[
            list(map(str, unique_labels)), ["precision", "recall", "f1-score"]
        ].T.iterrows()
    ]

    if damage_present:
        # create report only for damage categories (represented by 1,2,3)
        dam_report = classification_report(
            labels, preds, labels=damage_labels, output_dict=True, zero_division=1
        )
    else:
        dam_report = classification_report(
            np.array([1]),
            np.array([1]),
            labels=damage_labels,
            output_dict=True,
            zero_division=1,
        )

        dam_report = pd.DataFrame(dam_report).transpose()

        score_overview = score_overview.append(pd.Series(name="damage macro avg"))
        score_overview = score_overview.append(pd.Series(name="damage weighted avg"))
        score_overview = score_overview.append(pd.Series(name="damage harmonized avg"))

        score_overview.loc[
            "damage macro avg", ["precision", "recall", "f1-score", "support"]
        ] = (
            dam_report.loc[
                ["macro avg"], ["precision", "recall", "f1-score", "support"]
            ]
            .values.flatten()
            .tolist()
        )

        score_overview.loc[
            "damage weighted avg", ["precision", "recall", "f1-score", "support"]
        ] = (
            dam_report.loc[
                ["weighted avg"], ["precision", "recall", "f1-score", "support"]
            ]
            .values.flatten()
            .tolist()
        )

        score_overview.loc[
            "damage harmonized avg", ["precision", "recall", "f1-score"]
        ] = [
            harmonic_score(r)
            for i, r in score_overview.loc[
                list(map(str, damage_labels)), ["precision", "recall", "f1-score"]
            ].T.iterrows()
        ]

    if damage_mapping:
        score_overview.rename(index=damage_mapping, inplace=True)
    return score_overview, df_pred, damage_mapping

def main():
    # give an overview of the score
    score_overview, df_pred, damage_mapping = gen_score_overview("./results/joplin_all")
    print(score_overview)

    # uncomment below to obtain confusion matrix
    # from sklearn.metrics import confusion_matrix
    # import matplotlib.pyplot as plt
    # my_joplin = open("results/tuscaloosa_all-split_test-epoch_001-model_inception-predictions.txt")
    # my_joplin_df = pd.read_table(my_joplin, delim_whitespace=True)
    # print(my_joplin_df.head())
    # y_true = my_joplin_df["label"]
    # y_pred =my_joplin_df["prediction"]
    # cm = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(15, 15))

    # uncomment below to obtain heat map cooresponding to confusion matrix
    # import seaborn as sns
    # sns.heatmap(cm, annot=True, fmt=".2f", cbar=True, cmap="Blues", xticklabels = ["No damage","Minor damage","Major damage","Destroyed"],yticklabels = ["No damage","Minor damage","Major damage","Destroyed"])
    # plt.xticks(rotation=90)
    # plt.yticks(rotation=0)
    # plt.xlabel("Predicted label")
    # plt.ylabel("Actual label")
    # plt.show()
    # print(cm)


if __name__ == "__main__":
    main()