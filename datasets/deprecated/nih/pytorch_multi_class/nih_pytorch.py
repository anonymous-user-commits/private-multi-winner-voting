import os
import random as rn
from glob import glob

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

import torch

import matplotlib

if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
else:
    pass
    # %matplotlib inline
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from itertools import chain
import getpass

from datasets.deprecated.nih import DensModel
from datasets.deprecated.nih import get_data_loaders


def set_environment():
    # fix random seed
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    torch.backends.cudnn.deterministic = True


def compute_class_freqs(labels):
    """
    Compute positive and negative frequencies for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequencies for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequencies for each
                                         class, size (num_classes)
    """

    # total number of patients (rows)
    N = len(labels)

    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies


def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there are any patients in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs

    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """
    df1_patients_unique = set(df1[patient_col].unique().tolist())
    df2_patients_unique = set(df2[patient_col].unique().tolist())

    patients_in_both_groups = df1_patients_unique.intersection(
        df2_patients_unique)

    # leakage contains true if there is patient overlap, otherwise false.
    # boolean (true if there is at least 1 patient in both groups)
    leakage = len(patients_in_both_groups) >= 1
    return leakage


def get_roc_curve(labels, predicted_vals, gt_labels):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = gt_labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.savefig(
        'roc_curve.png',
        bbox_inches='tight',
        transparent=False
    )
    return auc_roc_vals


def prevent_leakage(first_df, other_df):
    """
    Prevent data leakage.

    :param first_df: leave as is
    :param other_df: remove ids that overlap with first_df
    """

    ids_first = first_df['Patient ID'].values
    ids_other = other_df['Patient ID'].values

    # Create a "set" data structure of the training set id's to identify
    # unique id's.
    ids_first_set = set(ids_first)
    print(
        f'There are {len(ids_first_set)} unique Patient IDs '
        f'in the first set.')

    ids_other_set = set(ids_other)
    print(
        f'There are {len(ids_other_set)} unique Patient IDs '
        f'in the other set.')

    # Identify patient overlap by looking at the intersection between the sets.
    patient_overlap = list(ids_first_set.intersection(ids_other_set))
    n_overlap = len(patient_overlap)
    print(f'There are {n_overlap} Patient IDs in both sets')
    # print(f'These patients are in both datasets: {patient_overlap}')

    first_overlap_idxs = []
    other_overlap_idxs = []
    for idx in range(n_overlap):
        first_overlap_idxs.extend(
            first_df.index[first_df['Patient ID'] == patient_overlap[
                idx]].tolist())
        other_overlap_idxs.extend(
            other_df.index[other_df['Patient ID'] == patient_overlap[
                idx]].tolist())

    # print(
    #     f'These are the indices of overlapping patients in the first set: {first_overlap_idxs}')
    # print(
    #     f'These are the indices of overlapping patients in the other set: {other_overlap_idxs}')

    # Drop the overlapping rows from the validation set
    other_df.drop(other_overlap_idxs, inplace=True)

    # Extract patient id's for the other set
    ids_other = other_df['Patient ID'].values
    # Create a "set" datastructure of the validation set id's to
    # identify unique id's.
    ids_other_set = set(ids_other)
    print(
        f'There are {len(ids_other_set)} unique '
        f'Patient IDs in the other set.')

    # Identify patient overlap by looking at the intersection between the sets
    patient_overlap = list(ids_first_set.intersection(ids_other_set))
    n_overlap = len(patient_overlap)
    print(
        f'There are {n_overlap} Patient IDs in both sets')
    return first_df, other_df


def prepare_data():
    set_environment()

    # access the dataset
    user = getpass.getuser()
    data_dir = f"/home/{user}/data/nih/"
    print(f'all dirs in {data_dir}: ', os.listdir(data_dir))
    all_xray_df = pd.read_csv(data_dir + 'Data_Entry_2017.csv')
    # all_xray_df = all_xray_df.iloc[:1000]

    all_image_paths = {os.path.basename(x): x for x in glob(
        os.path.join(data_dir, 'images*', '*', '*.png'))}
    # print('all_image_paths: ', all_image_paths)

    print('Scans found:', len(all_image_paths), ', Total Headers',
          all_xray_df.shape[0])
    all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
    # all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x[:-1]))
    pd.set_option('display.max_columns', None)
    print(all_xray_df.sample(3))
    # print('all_xray_df.head(): ', all_xray_df.head())

    """
    Preprocessing Labels

    We take the labels and make them into a more clear format. 
    The primary step is to see the distribution of findings and then to convert them to simple binary labels.

    """

    label_counts = all_xray_df['Finding Labels'].value_counts()
    # This 15 is an arbitrary but reasonable.
    label_counts = label_counts[:15]
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax1.bar(np.arange(len(label_counts)) + 0.5, label_counts)
    ax1.set_xticks(np.arange(len(label_counts)) + 0.5)
    _ = ax1.set_xticklabels(label_counts.index, rotation=90)
    _ = ax1.set_ylabel('Frequency count')
    destination = 'initial-nih-class-distribution.png'
    fig.savefig(destination,
                bbox_inches='tight',
                transparent=False
                )

    labels = ['Cardiomegaly',
              'Emphysema',
              'Effusion',
              'Hernia',
              'Infiltration',
              'Mass',
              'Nodule',
              'Atelectasis',
              'Pneumothorax',
              'Pleural_Thickening',
              'Pneumonia',
              'Fibrosis',
              'Edema',
              'Consolidation']

    all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(
        lambda x: x.replace('No Finding', ''))
    all_labels = np.unique(list(chain(
        *all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x) > 0]
    print('All Labels ({}): {}'.format(
        len(all_labels), all_labels))
    # Create column for each disease. The value for the new column in a given
    # row is 1 if patient was diagnosed with this disease, otherwise the value is 0.
    for c_label in all_labels:
        if len(c_label) > 1:  # leave out empty labels
            all_xray_df[c_label] = all_xray_df['Finding Labels'].map(
                lambda finding: 1 if c_label in finding else 0)

    print(all_xray_df.sample(3))

    drop_column = ['Patient Age', 'Patient Gender', 'View Position',
                   'Follow-up #', 'OriginalImagePixelSpacing[x', 'y]',
                   'OriginalImage[Width', 'Height]', 'Unnamed: 11']
    all_xray_df = all_xray_df.drop(drop_column, axis=1)

    # all_labels_values = all_xray_df[all_labels].values
    # Frequency (%) of a given label.
    label_counts = 100 * np.mean(all_xray_df[all_labels].values, 0)
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax1.bar(np.arange(len(label_counts)) + 0.5, label_counts)
    ax1.set_xticks(np.arange(len(label_counts)) + 0.5)
    ax1.set_xticklabels(all_labels, rotation=90)
    ax1.set_title('Adjusted Frequency of Diseases in Patient Group')
    _ = ax1.set_ylabel('Frequency (%)')
    destination = 'nih-class-distribution.png'
    fig.savefig(destination,
                bbox_inches='tight',
                transparent=False
                )

    # Prepare dataset.

    all_xray_df['disease_vec'] = all_xray_df.apply(
        lambda x: x[all_labels].values, axis=1)  # axis = 1, apply to each row
    # x = all_xray_df.iloc[0]
    # x_all_labels = x[all_labels]
    # x_all_labels_values = x_all_labels.values
    # disease_vector = all_xray_df['disease_vec']
    # all_xray_df['disease_vec'] = all_xray_df['disease_vec'].map(lambda x: x[0])
    train_df, valid_df, test_df = np.split(
        all_xray_df.sample(frac=1),  # shuffle the rows
        [int(.6 * len(all_xray_df)),  # indices of data sections
         int(.8 * len(all_xray_df))])

    print('train_df:\n', train_df.shape[0])
    print('validation_df:\n', valid_df.shape[0])
    print('test:\n', test_df.shape[0])
    print('train_df.head()\n: ', train_df.head())
    print('valid_df.head()\n: ', valid_df.head())
    print('test_df.head()\n: ', test_df.head())

    # """
    # Prepare Training Data - here we split the data into training and validation sets and create a single vector (disease_vec) with the 0/1 outputs for the disease status (what the model will try and predict).
    # """
    # all_xray_df['disease_vec'] = all_xray_df.apply(
    #     lambda x: [x[all_labels].values])
    # train_df, valid_df = train_test_split(
    #     all_xray_df,
    #     test_size=0.25,
    #     random_state=2018,
    #     stratify=all_xray_df[
    #         'Finding Labels'].map(
    #         lambda x: x[:4]))
    # print('train_df:\n', train_df.shape[0])
    # print('validation_df:\n', valid_df.shape[0])
    # print('train_df.head(): ', train_df.head())
    # print('valid_df.head(): ', valid_df.head())

    train_labels = []
    ds_len = train_df.shape[0]
    print(f'length of the training set: {ds_len}')

    for inx in range(ds_len):
        row = train_df.iloc[inx]
        vec = np.array(row['disease_vec'], dtype=np.int)
        train_labels.append(vec)

    freq_pos, freq_neg = compute_class_freqs(train_labels)
    print('freq_pos: ', freq_pos)
    print('freq_neg: ', freq_neg)

    data = pd.DataFrame(
        {"Class": labels, "Label": "Positive", "Value": freq_pos})
    data = data.append(
        [{"Class": labels[l], "Label": "Negative", "Value": v} for l, v in
         enumerate(freq_neg)], ignore_index=True)
    plt.xticks(rotation=90)
    f = sns.barplot(x="Class", y="Value", hue="Label", data=data)
    fig.savefig('first-barplot.png',
                bbox_inches='tight',
                transparent=False
                )
    pos_weights = freq_neg
    neg_weights = freq_pos
    pos_contribution = freq_pos * pos_weights
    neg_contribution = freq_neg * neg_weights

    data = pd.DataFrame(
        {"Class": labels, "Label": "Positive", "Value": pos_contribution})
    # Append rows of other to the end of caller, returning a new object.
    data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v}
                        for l, v in enumerate(neg_contribution)],
                       ignore_index=True)
    plt.xticks(rotation=90)
    sns.barplot(x="Class", y="Value", hue="Label", data=data)
    fig.savefig('second-barplot.png',
                bbox_inches='tight',
                transparent=False
                )

    train_df, valid_df = prevent_leakage(first_df=train_df, other_df=valid_df)
    train_df, test_df = prevent_leakage(first_df=train_df, other_df=test_df)
    test_df, valid_df = prevent_leakage(first_df=test_df, other_df=valid_df)

    print("leakage between train and valid: {}".format(
        check_for_leakage(train_df, valid_df, 'Patient ID')))
    print("leakage between train and test: {}".format(
        check_for_leakage(train_df, test_df, 'Patient ID')))
    print("leakage between validation and test: {}".format(
        check_for_leakage(valid_df, test_df, 'Patient ID')))

    save_data(train_df=train_df,
              valid_df=valid_df,
              test_df=test_df,
              neg_weights=neg_weights,
              pos_weights=pos_weights,
              )


def save_data(train_df, valid_df, test_df, neg_weights, pos_weights):
    train_df.to_pickle('train.pkl')
    valid_df.to_pickle('valid.pkl')
    test_df.to_pickle('test.pkl')

    weights = np.array([neg_weights, pos_weights])
    np.save(file='weights.npy', arr=weights)


def load_data():
    weights = np.load('weights.npy')
    neg_weights = weights[0]
    pos_weights = weights[1]
    train_df = pd.read_pickle('train.pkl')
    valid_df = pd.read_pickle('valid.pkl')
    test_df = pd.read_pickle('test.pkl')
    return train_df, valid_df, test_df, neg_weights, pos_weights


def train():
    train_df, valid_df, test_df, neg_weights, pos_weights = load_data()

    print('len train: ', len(train_df))
    print('len val: ', len(valid_df))
    print('len test: ', len(test_df))

    trainload, valload, testload = get_data_loaders(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )
    model = DensModel(pos_weights=pos_weights, neg_weights=neg_weights)

    early_stopping = EarlyStopping('val_loss')
    checkpoint_callback = ModelCheckpoint(
        verbose=False,
        monitor='avg_val_loss',
        mode='min')

    trainer = Trainer(gpus=4, max_epochs=5,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=early_stopping)
    trainer.fit(model, trainload, valload)

    trainer.test(model, test_dataloaders=testload)


if __name__ == "__main__":
    prepare_data()
    train()
