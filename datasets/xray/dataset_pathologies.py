"""
Pure pathologies extracted from the default pathologies for each dataset.
"""

default_pathologies = (
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Enlarged Cardiomediastinum',
    'Fibrosis',
    'Fracture',
    'Hernia',
    'Infiltration',
    'Lung Lesion',
    'Lung Opacity',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax',
)

cxpert_pathologies = (
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    # no Emphysema
    'Enlarged Cardiomediastinum',
    # no Fibrosis
    'Fracture',
    # no Hernia
    # no Infiltration
    'Lung Lesion',
    'Lung Opacity',
    # no Mass
    # no Nodule
    # no Pleural_Thickening
    'Pneumonia',
    'Pneumothorax'
)

padchest_pathologies = (
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    # 'Enlarged Cardiomediastinum',
    'Fibrosis',
    'Fracture',
    'Hernia',
    'Infiltration',
    # 'Lung Lesion',
    # 'Lung Opacity',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax'
)

mimic_pathologies = (
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
)

vin_pathologies = (
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Enlarged Cardiomediastinum',
    'Fibrosis',
    'Fracture',
    'Hernia',
    'Infiltration',
    'Lung Lesion',
    'Lung Opacity',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax',
)

default_pathologies_indexes = {pathology: idx
                               for idx, pathology in
                               enumerate(default_pathologies)}


def get_chexpert_intersect_padchest_indexes():
    return get_padchest_intersect_chexpert_indexes()


def get_indexes_for_pathologies(from_pathologies):
    indexes = []
    for pathology, idx in default_pathologies_indexes.items():
        if pathology in from_pathologies:
            indexes.append(idx)
    return indexes


def get_padchest_intersect_chexpert_indexes():
    common = set(padchest_pathologies).intersection(set(cxpert_pathologies))
    return get_indexes_for_pathologies(from_pathologies=common)


def get_chexpert_indexes():
    return get_indexes_for_pathologies(from_pathologies=cxpert_pathologies)


def get_padchest_indexes():
    return get_indexes_for_pathologies(from_pathologies=padchest_pathologies)

def get_mimic_indexes():
    return get_indexes_for_pathologies(from_pathologies=mimic_pathologies)

def get_indexes(dataset):
    if dataset in ['cxpert', 'CheXpert']:
        return get_chexpert_indexes()
    elif dataset == 'padchest':
        return get_padchest_indexes()
    elif dataset == 'mimic':
        return get_mimic_indexes()
    else:
        raise Exception(f'Unsupported dataset: {dataset}')


if __name__ == "__main__":
    indexes = get_chexpert_intersect_padchest_indexes()
    print('chexpert intersect padchest: ', indexes)
    print('chexpert indexes: ', get_chexpert_indexes())
    print('padchest indexes: ', get_padchest_indexes())
    for pathology in mimic_pathologies:
        print(pathology)
    print(','.join(default_pathologies))
