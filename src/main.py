import os
import pandas as pd
import nibabel as nib
import utils, fc_profiles, features
import warnings
warnings.filterwarnings('ignore')

mask_names = utils.get_mask_names()
utils.create_hipp_system_templates(z_thresh=1.5)

def run_pipeline(config):

    output = config['output']
    subject = config['subject']

    os.makedirs(f'{output}/{subject}', exist_ok=True)
    for hemi in ['L','R']:

        path_hipp_systems   = config[f'path_hipp_systems_{hemi}']
        path_hipp_xs        = config[f'path_hipp_xs_{hemi}']
        path_hipp_surf_area = config[f'path_hipp_surf_area_{hemi}']
        path_ctx_xs         = config[f'path_ctx_xs_{hemi}']
        path_ctx_surf_area  = config[f'path_ctx_surf_area_{hemi}']

        df_features = pd.DataFrame()

        fc_profiles.get_hipp_FC_profiles(output, subject, path_hipp_systems, path_hipp_xs, hemi)
        fc_profiles.get_hipp_cortex_FC_profiles(output, subject, path_hipp_systems, path_hipp_xs, path_ctx_xs, hemi)
        fc_profiles.get_cortex_FC_profiles(output, subject, path_ctx_xs, hemi)


        # Get hipp features.
        surf_area = nib.load(path_hipp_surf_area).darrays[0].data
        fc_profile_gii = nib.load(f'{output}/{subject}/hipp_FC.{hemi}.func.gii')

        hipp_features = features.get_features(
            fc_profile_name='hipp',
            fc_profile_gii=fc_profile_gii,
            surf_area=surf_area,
            mask_name='whole_hipp',
            hemi=hemi
        )
        df_features = pd.concat([df_features, hipp_features])


        for mask_name in mask_names:

            # Get hipp-cortex features.
            surf_area = nib.load(path_ctx_surf_area).darrays[0].data
            fc_profile_gii = nib.load(f'{output}/{subject}/hipp_cortex_FC.{hemi}.func.gii')

            hipp_cortex_features = features.get_features(
                fc_profile_name='hipp_cortex',
                fc_profile_gii=fc_profile_gii,
                surf_area=surf_area,
                mask_name=mask_name,
                hemi=hemi
            )
            df_features = pd.concat([df_features, hipp_cortex_features])

            # Get cortex features.
            surf_area = nib.load(path_ctx_surf_area).darrays[0].data
            fc_profile_gii = nib.load(f'{output}/{subject}/cortex_FC.{hemi}.func.gii')

            cortex_features = features.get_features(
                fc_profile_name='cortex',
                fc_profile_gii=fc_profile_gii,
                surf_area=surf_area,
                mask_name=mask_name,
                hemi=hemi
            )
            df_features = pd.concat([df_features, cortex_features])

        df_features.to_csv(f'{output}/{subject}/features_{hemi}.csv')


    # Average across hemispheres.
    L = pd.read_csv(f'{output}/{subject}/features_L.csv', index_col=0)
    R = pd.read_csv(f'{output}/{subject}/features_R.csv', index_col=0)

    mean_features = (L + R) / 2
    mean_features.to_csv(f'{output}/{subject}/features.csv')
