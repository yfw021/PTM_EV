
import pandas as pd
import os

def load_from_pickle(file_path):
    return pd.read_pickle(file_path)


#import argparse



import pandas as pd

#def read_data(file_path):
 #   df = pd.read_csv(file_path)
  #  print("read in the data",df.head())
   # print("this is the data shape:",df.shape)
    #return df


import numpy as np
import random
# read in csv file (total combined pelm and ppa)

import multiprocessing

#step 1 
def whole_view(df):
    print("whole view of the data")
    pos_data= df[df['p_n']==1]
    neg_data= df[df['p_n']==0]
    print("this is the pos data length:",len(pos_data))
    print("this is the neg data length:",len(neg_data))
    print("this is the whole data length:",len(df))
    pelms_pos=df[df['database']=='pelmspos']
    pelms_neg=df[df['database']=='pelmsneg']
    ppas_pos=df[df['database']=='ppaspos']
    ppas_neg=df[df['database']=='ppasneg']
    print("SETP 2, Data info from different databases")
    print("this is the pelms_pos data length:",len(pelms_pos))
    print("this is the pelms_neg data length:",len(pelms_neg))
    print("this is the ppas_pos data length:",len(ppas_pos))
    print("this is the ppas_neg data length:",len(ppas_neg))
    return pos_data,neg_data

#step 2 here is to selected train data!!
def shuffle_data(df,pos_num_pelm,neg_num_pelm,pos_num_ppa,neg_num_ppa):
    print("Step 2: shuffle the data by input the number of pos and neg data and from different databases that you select")
    pelms_pos=df[df['database']=='pelmspos']
    pelms_neg=df[df['database']=='pelmsneg']
    ppas_pos=df[df['database']=='ppaspos']
    ppas_neg=df[df['database']=='ppasneg']
    pelm_shuf_pos=pelms_pos.sample(n=pos_num_pelm)
    pelm_shuf_neg=pelms_neg.sample(n=neg_num_pelm)
    ppa_shuf_pos=ppas_pos.sample(n=pos_num_ppa)
    ppa_shuf_neg=ppas_neg.sample(n=neg_num_ppa)
    shuffled_df=pd.concat([pelm_shuf_pos,pelm_shuf_neg,ppa_shuf_pos,ppa_shuf_neg])
    print(shuffled_df.shape)
    return shuffled_df

# step 3
def ev_cal(df):
    print("Step 3: calculate the EV value for the train data")
    col_4_EV=df[['P5seq', 'P4seq','P3seq','P2seq','P1seq','P0seq','N1seq','N2seq','N3seq','N4seq','N5seq','p_n','database']]
    col_4_EV_pos=col_4_EV[col_4_EV['p_n']==1]
    col_4_EV_neg=col_4_EV[col_4_EV['p_n']==0]
    print("this is pos data length:",len(col_4_EV_pos))
    print("this is neg data length:",len(col_4_EV_neg))

    prefix_list = ['P5', 'P4', 'P3', 'P2', 'P1', 'P0', 'N1', 'N2', 'N3','N4','N5']
    for prefix in prefix_list:
        col_name = f"Frequency_{prefix}"
        group_col = prefix + "seq"
        col_4_EV_pos[col_name] = col_4_EV_pos.groupby(group_col)[group_col].transform('size')
        col_4_EV_neg[col_name] = col_4_EV_neg.groupby(group_col)[group_col].transform('size')
    #check duplicates rows in the dataframe
    print(" this is the dulicate rows number:",col_4_EV_pos.duplicated().sum())
    print(" this is the dulicate rows number:",col_4_EV_neg.duplicated().sum()) 
    cols = col_4_EV_pos.filter(like='Frequency_')
    for col in cols:
        print("pos_unique_count:",col, col_4_EV_pos[col].nunique())
        print("neg_uniqie_count:",col, col_4_EV_neg[col].nunique())
  
    combined_df = pd.concat([col_4_EV_pos, col_4_EV_neg], axis=0)
    print("this is the combined data length:",len(combined_df))
    for prefix in prefix_list:
        col_name = f"Fre_whole_{prefix}"
        group_col = prefix + "seq"
        combined_df[col_name] = combined_df.groupby(group_col)[group_col].transform('size')

    print(" this is the dulicate rows number in combined_df:",combined_df.duplicated().sum())
  # Filter for columns that start with 'Frequency_'
    cols_com = combined_df.filter(like='Fre_whole')

  # Count the number of unique elements in each column
    for col in cols_com:
        print(col, combined_df[col].nunique())
  
    cal_ev_in_combined=combined_df[combined_df['p_n'] == 1]
    print("this is the cal_ev_in_combined data length(which is the pos length):",len(cal_ev_in_combined))
    cols = ['P5','P4', 'P3', 'P2', 'P1', 'P0', 'N1', 'N2', 'N3', 'N4','N5']
    for col in cols:
        cal_ev_in_combined[f'EV_{col}'] = ((cal_ev_in_combined[f'Frequency_{col}'] / len(cal_ev_in_combined)) /  ((cal_ev_in_combined[f'Fre_whole_{col}'])/cal_ev_in_combined['Fre_whole_P0']))
        cal_ev_in_combined[f'top_EV_{col}'] = (cal_ev_in_combined[f'Frequency_{col}'] / len(cal_ev_in_combined))
        cal_ev_in_combined[f'bottom_EV_{col}'] = (cal_ev_in_combined[f'Fre_whole_{col}'] / cal_ev_in_combined['Fre_whole_P0'])
       
  #check the nan value in the dataframe
    print("this is the nan value in the dataframe:",cal_ev_in_combined.isnull().sum())
    print("col_4_EV_pos:",col_4_EV_pos.shape)
    print("col_4_EV_neg:",col_4_EV_neg.shape)
    print("cal_ev_in_combined:",cal_ev_in_combined.shape)
    #print unique number of the columns
    print("this is the unique number of the columns:",cal_ev_in_combined.nunique())



    return col_4_EV_pos,col_4_EV_neg,cal_ev_in_combined

# step 4 Get the EV table from train data
def get_ev_table(df):
    print("Step 4: get the EV table for the data")
    ev_cols = [col for col in df.columns if col.startswith('EV_') or col.endswith('seq')]
    print(ev_cols)
    columns = [('P5seq', 'EV_P5'), ('P4seq', 'EV_P4'), ('P3seq', 'EV_P3'), ('P2seq', 'EV_P2'), ('P1seq', 'EV_P1'),
           ('N1seq', 'EV_N1'), ('N2seq', 'EV_N2'), ('N3seq', 'EV_N3'), ('N4seq', 'EV_N4'), ('N5seq', 'EV_N5')]
  #unique_dfs = {f'unique_{col[0][:2].lower()}{col[0][2:]}': df[[col[0], col[1]]].drop_duplicates() for col in columns}
    all_unique_df = pd.concat([df[[col[0], col[1]]].drop_duplicates() for col in columns], ignore_index=True)
    final_table = all_unique_df.apply(lambda x: pd.Series(x.dropna().values)).fillna('')
    return final_table

# step 5 mapping the EV value to the train
def add_result_col(shuffle_df, ev_table):
    print("Step 5: add the EV result column for the data")
    column_map = {
        'P5seq': 'EV_P5', 'P4seq': 'EV_P4', 'P3seq': 'EV_P3', 'P2seq': 'EV_P2', 'P1seq': 'EV_P1',
        'N1seq': 'EV_N1', 'N2seq': 'EV_N2', 'N3seq': 'EV_N3', 'N4seq': 'EV_N4', 'N5seq': 'EV_N5'}

# Loop through the dictionary and map the columns
    for seq_col, ev_col in column_map.items():
        shuffle_df[f'results_{seq_col.lower()}'] = shuffle_df[seq_col].map(ev_table.set_index(seq_col)[ev_col])
  
  # Select columns that start with "results"
    results_cols = [col for col in shuffle_df.columns if col.startswith("results")]
    shuffle_df["product"] = shuffle_df[results_cols].product(axis=1)

    return shuffle_df

# step 6
def count_values_above_and_below_1(df, column_name):
    print("Step 6: count the values above and below 1 for the data")

    above_1 = 0
    below_1 = 0
    for value in df[column_name]:
        if value > 1:
            above_1 += 1
        elif value < 1:
         below_1 += 1
    print("above_1:",above_1)
    print("below_1:",below_1)
    return above_1, below_1

# step 7

def evaluate_cutoffs(df, column_name, cutoffs, target_column='target'):
    from sklearn.metrics import confusion_matrix
    results = {}
    for cutoff in cutoffs:

        predictions = (df[column_name] > cutoff).astype(int)  # Convert predictions to 0/1
        tn, fp, fn, tp = confusion_matrix(df[target_column], predictions).ravel()
        #print(confusion_matrix(df[target_column], predictions))
        accuracy = (tp + tn) / (len(df))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        results[cutoff] = {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'sensitivity': sensitivity
        }

    return results


# step 8
def plot_specificity_sensitivity(results, cutoffs, output_file=None):
  import pandas as pd
  from sklearn.metrics import confusion_matrix
  import matplotlib.pyplot as plt  # Import for plotting
  specificities = [results[cutoff]['specificity'] for cutoff in cutoffs]
  sensitivities = [results[cutoff]['sensitivity'] for cutoff in cutoffs]

  plt.figure(figsize=(8, 5))
  plt.plot(cutoffs, specificities, label='Specificity')
  plt.plot(cutoffs, sensitivities, label='Sensitivity')
  plt.xlabel('Cutoff')
  plt.ylabel('Value')
  plt.title('Specificity and Sensitivity by Cutoff')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()



    # Save results to text file if output_file specified
  if output_file:
    with open(output_file, 'w') as f:
      f.write(f"Evaluation Results test_ppa_pos5028_neg166942 for Cutoff Values\n\n")
      for cutoff, metrics in results.items():
        f.write(f"Cutoff: {cutoff:.2f}\n")
        f.write(f"  TP: {metrics['TP']}, TN: {metrics['TN']}\n")
        f.write(f"  FP: {metrics['FP']}, FN: {metrics['FN']}\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}\n")
        f.write(f"  F1-score: {metrics['f1_score']:.4f}\n\n")
  return results

# step 9
def AUC_ROC(results, cutoffs):
    import numpy as np
    from sklearn.metrics import auc
    sensitivities = [results[cutoff]['sensitivity'] for cutoff in cutoffs]
    one_minus_specificities = [1 - results[cutoff]['specificity'] for cutoff in cutoffs]
    auc_score = auc(one_minus_specificities, sensitivities)
    return auc_score





def main(file_path):
    #df = read_data(file_path)
    # load pickle file
    print("start to load the pickle file, start time", pd.Timestamp.now())
    df = load_from_pickle(file_path)
    print("load the pickle file, end time", pd.Timestamp.now())
    #step 1
    pos_data,neg_data=whole_view(df)
    #step 2
    shuffled_df=shuffle_data(df,20128,20128,1000,1000)
    #step 3
    #col_4_EV_pos,col_4_EV_neg,cal_ev_in_combined=ev_cal(shuffled_df)
    #step 4
    #final_table=get_ev_table(cal_ev_in_combined)
    #print("this is the final_table:",final_table)
    #step 5
    #final_df=add_result_col(shuffled_df,final_table)
    #step 6
    #above_1, below_1 = count_values_above_and_below_1(final_df, "product")
    #step 7
    #column_name = 'product'
    #cutoffs = np.arange(0, 4, 0.05)
    #results = evaluate_cutoffs(final_df, column_name, cutoffs, target_column='p_n')
    #output_file = 'train_combied_vali_12000.txt'
    #plot_specificity_sensitivity(evaluation_results, cutoffs)
    #auc_score = AUC_ROC(evaluation_results, cutoffs)
    #print("AUC score:",auc_score)
    #print("evaluation_results:",evaluation_results)












if __name__ == "__main__":
    #file_path = "filtered_combined_pelm_ppa.csv"
    file_path = "all_data_final.pkl"
    main(file_path)