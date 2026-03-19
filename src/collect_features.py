import utils
import numpy as np
import pandas as pd
import pingouin as pg

def get_feature_names():

    mask_names  = utils.get_mask_names()
    pairs       = ['ant_post','ant_body','body_post']
    z_thresh    = {'hipp':'0', 'cortex':'0p5'}

    feature_names = []
    for pair in pairs:

        feature_names.append(f'hipp__whole_hipp__{pair}__spatial_corr')
        feature_names.append(f"hipp__whole_hipp__{pair}__pct_overlap_z={z_thresh['hipp']}")

        for mask_name in mask_names:
            feature_names.append(f'cortex__{mask_name}__{pair}__spatial_corr')
            feature_names.append(f"cortex__{mask_name}__{pair}__pct_overlap_z={z_thresh['cortex']}")

            feature_names.append(f'hipp_cortex__{mask_name}__{pair}__spatial_corr')
            feature_names.append(f"hipp_cortex__{mask_name}__{pair}__pct_overlap_z={z_thresh['cortex']}")

    feature_names.sort()

    return feature_names


def collect_features(output, subjects, feature_names, hemi=None):

    results = {}
    for subject in subjects:

        if not hemi:
            features = pd.read_csv(f'{output}/{subject}/features.csv', index_col=0)
        else:
            features = pd.read_csv(f'{output}/{subject}/features_{hemi}.csv', index_col=0)

        results[subject] = features.loc[feature_names]['features'].to_dict()

    results = pd.DataFrame.from_dict(results, orient='index')
    results = results.reset_index(names=['id'])

    return results


def regress_confound(X, confound):

    confound = confound.reshape(-1, 1)

    design = np.hstack([np.ones((confound.shape[0], 1)), confound])
    beta = np.linalg.lstsq(design, X, rcond=None)[0]
    X_hat = design @ beta

    X_resid = X - X_hat
    X_clean = X_resid + X.mean(axis=0)

    return X_clean



# Colect BANDA features.

output = '/host/corin/tank/jonah/MPN/analyses/hipp_system_mapping/output_BANDA'

df = pd.read_csv('/host/corin/tank/jonah/MPN/subjects.csv')
df = df[~np.isin(df.id, ['BANDA059','BANDA181'])]
df = df[df.n_minutes > 10]
subjects = df['id'].tolist()


feature_names = get_feature_names()
results = collect_features(output, subjects, feature_names, hemi=None)

merged = pd.merge(df, results, on='id')

# Regress out mean-fd.
for feature in feature_names:
    merged[feature] = regress_confound(merged[feature], confound=merged['mean_fd'].to_numpy())

merged.to_csv(f'{output}/BANDA_feature_set.csv', index=False)





# Colect HCPD features.
output = '/host/corin/tank/jonah/MPN/analyses/hipp_system_mapping/output_HCPD'

df = pd.read_csv('/host/corin/tank/jonah/pmat/src_hipp/src/workspaces/supplementary_data.csv')
df = df[['id','age','sex','site','nih_picseq_raw','nih_picseq_ageadjusted']]
subjects = df['id'].tolist()

results = collect_features(output, subjects, feature_names, hemi=None)
merged = pd.merge(df, results, on='id')
merged.to_csv(f'{output}/HCPD_feature_set.csv', index=False)
