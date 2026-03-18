import numpy as np
import nibabel as nib
from scipy.stats import zscore


def create_hipp_system_templates(z_thresh=1.5):

    system_names = ['ant','body','post']

    for hemi in ['L','R']:

        tpl_darrays = nib.load(f'templates/template.{hemi}.shape.gii').darrays

        for idx, name in enumerate(system_names):

            system_data = tpl_darrays[idx].data
            system_data[system_data < z_thresh] = 0
            gii = create_func_gii([system_data], hemi=hemi, map_names=[name])
            nib.save(gii, f'templates/{name}_hipp_cortex_system.{hemi}.func.gii')


def create_func_gii(data, hemi, map_names):
    '''Convert data-arrays to func GIFTI.'''

    darrays = []
    for x, map_name in zip(data, map_names):
        darray = nib.gifti.GiftiDataArray(
            np.array(x, dtype='float32'),
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'])
        darray.meta = nib.gifti.GiftiMetaData({'Name':map_name})
        darrays.append(darray)

    # Create meta-data.
    if hemi == 'L': meta = nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary':'CortexLeft'})
    if hemi == 'R': meta = nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary':'CortexRight'})

    # Create final GIFTI.
    gifti = nib.GiftiImage(darrays=darrays, meta=meta)
    return gifti


def get_mask_names():
    mask_names = [
        'Anterior_Cingulate_and_Medial_Prefrontal',
        'Inferior_Parietal',
        'Posterior_Cingulate'
    ]
    return mask_names


def get_mask(mask_name, hemi):

    if mask_name == 'whole_hipp': return np.arange(0,7262)

    mask_data = nib.load(f'templates/{mask_name}.{hemi}.label.gii').darrays[0].data
    mask = np.argwhere(mask_data > 0).flatten()

    return mask
