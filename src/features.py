import utils
import numpy as np
import pandas as pd
import pingouin as pg


def get_features(fc_profile_name, fc_profile_gii, surf_area, mask_name, hemi, system_names=['ant','body','post'], z_thresholds=[0,.5,1,1.5]):
    '''fc_profile_name: One of ['hipp','hipp_cortex','cortex]'''

    mask = utils.get_mask(mask_name, hemi)
    fc_profiles = {name:darray.data for name, darray in zip(system_names, fc_profile_gii.darrays)}

    features = {}
    for pairs in [('ant','body'), ('ant','post'), ('body','post')]:

        i, j = pairs
        features[f'{fc_profile_name}__{mask_name}__{i}_{j}__spatial_corr'] = pg.corr(fc_profiles[i][mask], fc_profiles[j][mask])['r'].item()

        for z in z_thresholds:

            i_idx = np.argwhere(fc_profiles[i][mask] > z).flatten()
            j_idx = np.argwhere(fc_profiles[j][mask] > z).flatten()

            overlap_idx = list(set(i_idx).intersection(set(j_idx)))
            union_idx   = list(set(i_idx).union(set(j_idx)))

            surf_area_i = surf_area[i_idx].sum()
            surf_area_j = surf_area[j_idx].sum()

            surf_area_overlap = surf_area[overlap_idx].sum()
            surf_area_union   = surf_area[union_idx].sum()

            pct_overlap = (surf_area_overlap / surf_area_union) * 100

            z_str = str(z).replace('.','p')
            features[f'{fc_profile_name}__{mask_name}__{i}__surf_area_z={z_str}'] = surf_area_i
            features[f'{fc_profile_name}__{mask_name}__{j}__surf_area_z={z_str}'] = surf_area_j

            features[f'{fc_profile_name}__{mask_name}__{i}_{j}__surf_area_overlap_z={z_str}'] = surf_area_overlap
            features[f'{fc_profile_name}__{mask_name}__{i}_{j}__surf_area_union_z={z_str}'] = surf_area_union
            features[f'{fc_profile_name}__{mask_name}__{i}_{j}__pct_overlap_z={z_str}'] = pct_overlap

    features = pd.Series(features, name='features').sort_index()

    return features
