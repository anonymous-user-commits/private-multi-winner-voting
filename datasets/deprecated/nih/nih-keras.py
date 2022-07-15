# Process NIH Chest X-Ray data.

import os
import random as rn
from glob import glob
from sklearn.model_selection import train_test_split

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import getpass
from itertools import chain
import matplotlib

if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
else:
    pass
    # %matplotlib inline
import matplotlib.pyplot as plt


def set_environment():
    # fix random seed
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    torch.backends.cudnn.deterministic = True


def prepare_data():
    set_environment()
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

    info = """
    Preprocessing labels - we take the labels and make them into a more clear format. 
    The primary step is to see the distribution of findings and then to convert them to simple binary labels (many-hot vector encoding).
    """
    print(info)

    print('The 15 labels is an arbitrary but reasonable.')
    max_label_count = 15
    label_counts = all_xray_df['Finding Labels'].value_counts()[
                   :max_label_count]

    print('Draw the initial bar plot with labels and their counts.')
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    x_axis_pos = np.arange(len(label_counts)) + 0.5
    ax1.bar(x_axis_pos, label_counts)
    ax1.set_xticks(x_axis_pos)
    _ = ax1.set_xticklabels(label_counts.index, rotation=90)
    _ = ax1.set_ylabel('Frequency count')
    destination = 'initial-nih-class-distribution.png'
    fig.savefig(destination,
                bbox_inches='tight',
                transparent=False
                )

    # labels = ['Cardiomegaly',
    #           'Emphysema',
    #           'Effusion',
    #           'Hernia',
    #           'Infiltration',
    #           'Mass',
    #           'Nodule',
    #           'Atelectasis',
    #           'Pneumothorax',
    #           'Pleural_Thickening',
    #           'Pneumonia',
    #           'Fibrosis',
    #           'Edema',
    #           'Consolidation']

    drop_column = ['Patient Age', 'Patient Gender', 'View Position',
                   'Follow-up #', 'OriginalImagePixelSpacing[x', 'y]',
                   'OriginalImage[Width', 'Height]', 'Unnamed: 11']

    all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(
        lambda x: x.replace('No Finding', ''))
    all_labels = np.unique(list(chain(
        *all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    print("Remove the 'No Finding' label that currently is expressed "
          "as an empty string.")
    all_labels = [x for x in all_labels if len(x) > 0]
    print(f'All Labels ({len(all_labels)}): {all_labels}')
    print('Create the new many-hot vector encoding.')
    for c_label in all_labels:  # for each class label
        if len(c_label) > 1:  # leave out empty labels
            all_xray_df[c_label] = all_xray_df['Finding Labels'].map(
                lambda finding: 1 if c_label in finding else 0)

    print('many-hot vector encoding:\n', all_xray_df.sample(3))

    print('Show the labels with their counts.')
    print('All Labels ({})'.format(len(all_labels)),
          [(c_label, int(all_xray_df[c_label].sum())) for c_label in
           all_labels])

    info = """
    Clean categories - since we have too many categories, 
    we can prune a few out by taking the ones with only a few examples.
    """
    print(info)
    print('Current state:')
    label_count = [(c_label, int(all_xray_df[c_label].sum())) for c_label in
                   all_labels]
    print('label_count: ', label_count)
    print("Keep at least 1000 cases.")
    MIN_CASES = 1000
    all_labels = [c_label for c_label in all_labels if
                  all_xray_df[c_label].sum() > MIN_CASES]
    print(f'Clean Labels ({len(all_labels)})')
    label_count = [(c_label, int(all_xray_df[c_label].sum())) for c_label in
                   all_labels]
    print('label_count: ', label_count)

    info = """Since the dataset is very imbalanced, we can resample it to be 
    a more reasonable collection with weight being const + number of findings."""
    print(info)
    sample_weights = all_xray_df['Finding Labels'].map(
        lambda x: len(x.split('|')) if len(x) > 0 else 0)
    const = 4e-2
    sample_weights = sample_weights.values + const
    sample_weights /= sample_weights.sum()
    sample_count = 40000
    print('sample_count: ', sample_count)
    print('current count: ', all_xray_df.shape[0])
    all_xray_df = all_xray_df.sample(sample_count, weights=sample_weights)

    label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    x_axis_pos = np.arange(len(label_counts)) + 0.5
    ax1.bar(x_axis_pos, label_counts)
    ax1.set_xticks(x_axis_pos)
    _ = ax1.set_xticklabels(label_counts.index, rotation=90)
    ax1.set_title('After weighted sampling')
    _ = ax1.set_ylabel('Counts')
    destination = 'nih-class-distribution-after-sampling.png'
    fig.savefig(destination,
                bbox_inches='tight',
                transparent=False
                )

    all_xray_df = all_xray_df.drop(drop_column, axis=1)
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

    info = """
    Prepare Training Data - here we split the data into training and validation sets and create a single vector (disease_vec) with the 0/1 outputs for the disease status (what the model will try and predict).
    """
    print(f'info: {info}')
    ax1.set_title('Adjusted Frequency of Diseases in Patient Group')
    _ = ax1.set_ylabel('Frequency (%)')
    destination = 'nih-class-distribution.png'
    fig.savefig(destination,
                bbox_inches='tight',
                transparent=False
                )
    all_xray_df['disease_vec'] = all_xray_df.apply(
        lambda x: x[all_labels].values, axis=1)
    train_df, valid_df = train_test_split(
        all_xray_df,
        test_size=0.25,
        random_state=2018,
        stratify=all_xray_df[
            'Finding Labels'].map(
            lambda x: x[:4]))
    print('len train_df:\n', train_df.shape[0])
    print('len validation_df:\n', valid_df.shape[0])
    print('train_df.head(): ', train_df.head())
    print('valid_df.head(): ', valid_df.head())

    all_labels = np.array(all_labels)

    save_data(train_df=train_df, valid_df=valid_df,
              all_labels=all_labels)


def save_data(train_df, valid_df, all_labels):
    train_df.to_pickle('train-keras.pkl')
    valid_df.to_pickle('valid-keras.pkl')
    np.save('all_labels_keras.npy', all_labels)


def load_data():
    train_df = pd.read_pickle('train-keras.pkl')
    valid_df = pd.read_pickle('valid-keras.pkl')
    all_labels = np.load('all_labels_keras.npy')
    return train_df, valid_df, all_labels


def plot_samples(train_gen, all_labels):
    t_x, t_y = next(train_gen)
    fig, m_axs = plt.subplots(4, 4, figsize=(16, 16))
    for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
        c_ax.imshow(c_x[:, :, 0], cmap='bone', vmin=-1.5, vmax=1.5)
        c_ax.set_title(
            ', '.join([n_class for n_class, n_score in zip(all_labels, c_y)
                       if n_score > 0.5]))
        c_ax.axis('off')
    fig.savefig('plot-samples-train-gen.png',
                bbox_inches='tight',
                transparent=False
                )


def create_data_generators():
    """
    Make the data generators for loading and randomly transforming images.
    """
    train_df, valid_df, all_labels = load_data()

    from keras_preprocessing.image.image_data_generator import \
        ImageDataGenerator
    IMG_SIZE = (128, 128)
    core_idg = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        horizontal_flip=True,
        vertical_flip=False,
        height_shift_range=0.05,
        width_shift_range=0.1,
        rotation_range=5,
        shear_range=0.1,
        fill_mode='reflect',
        zoom_range=0.15)

    train_gen = flow_from_dataframe(core_idg, train_df,
                                    path_col='path',
                                    y_col='disease_vec',
                                    target_size=IMG_SIZE,
                                    color_mode='grayscale',
                                    batch_size=32)
    plot_samples(train_gen=train_gen, all_labels=all_labels)

    valid_gen = flow_from_dataframe(core_idg, valid_df,
                                    path_col='path',
                                    y_col='disease_vec',
                                    target_size=IMG_SIZE,
                                    color_mode='grayscale',
                                    batch_size=256)  # we can use much larger batches for evaluation

    # used a fixed dataset for evaluating the algorithm
    test_X, test_Y = next(flow_from_dataframe(core_idg,
                                              valid_df,
                                              path_col='path',
                                              y_col='disease_vec',
                                              target_size=IMG_SIZE,
                                              color_mode='grayscale',
                                              batch_size=1024))  # one big batch
    return train_gen, valid_gen, test_X, test_Y


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                              class_mode='sparse',
                                              **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = ''  # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


def create_model(train_gen, valid_gen, test_X, test_Y):
    """
    Make a simple model to train using MobileNet as a base and then adding
    a GAP layer (Flatten could also be added), dropout, and a fully-connected
    layer to calculate specific features.
    """
    from keras.applications.mobilenet import MobileNet
    from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
    from keras.models import Sequential
    base_mobilenet_model = MobileNet(input_shape=t_x.shape[1:],
                                     include_top=False, weights=None)
    multi_disease_model = Sequential()
    multi_disease_model.add(base_mobilenet_model)
    multi_disease_model.add(GlobalAveragePooling2D())
    multi_disease_model.add(Dropout(0.5))
    multi_disease_model.add(Dense(512))
    multi_disease_model.add(Dropout(0.5))
    multi_disease_model.add(Dense(len(all_labels), activation='sigmoid'))
    multi_disease_model.compile(optimizer='adam', loss='binary_crossentropy',
                                metrics=['binary_accuracy', 'mae'])
    multi_disease_model.summary()

    from keras.callbacks import ModelCheckpoint, LearningRateScheduler, \
        EarlyStopping, ReduceLROnPlateau
    weight_path = "{}_weights.best.hdf5".format('xray_class')

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min',
                                 save_weights_only=True)

    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=3)
    callbacks_list = [checkpoint, early]

    info = """
    Here we do a first round of training to get a few initial low hanging fruit results.
    """
    multi_disease_model.fit_generator(train_gen,
                                      steps_per_epoch=100,
                                      validation_data=(test_X, test_Y),
                                      epochs=1,
                                      callbacks=callbacks_list)


if __name__ == "__main__":
    prepare_data()
    train_gen, valid_gen, test_X, test_Y = create_data_generators()
    create_model(train_gen=train_gen, valid_gen=valid_gen,
                 test_X=test_X, test_Y=test_Y)

