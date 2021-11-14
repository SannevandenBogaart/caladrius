import pandas as pd
import math
import numpy as np

# function to read csv obtained from last cnn layer and output it as a pandas dataframe
def read_csv():
    # read csv file
    relu = pd.read_csv("./relu/matthew_all_train.csv", na_filter=True)
    test = relu.to_numpy().flatten()

    # make sure all values become numeric and same format
    cleaned_test = [x for x in test if (str(x).find('.png') >= 0 or (str(x).replace(".", "").replace(" ", "")
                                                                       .replace("E+", "").replace("E-", "").replace("e+", "").replace("e-", "").replace(",", "")
                                                                       .isnumeric()))]
    cleaned_test2 = [i if str(i).find(".png") >= 0 else float(str(i).replace(",", "")) for i in cleaned_test]
    size = len(cleaned_test2)
    idx_list = [idx for idx, val in enumerate(cleaned_test2) if str(val).find(".png") >= 0]
    res = [cleaned_test2[i: j] for i, j in zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]
    res.pop(0)
    # Merge all lists, to create dataframe
    df = pd.DataFrame(res)
    # Save dataframe to csv
    df = df.dropna()
    df.to_csv("./matthew_all_correct.csv")
    return df

def main():
    print(read_csv().head())

if __name__ == "__main__":
    main()