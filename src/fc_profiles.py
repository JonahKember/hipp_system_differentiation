import utils
import numpy as np
import nibabel as nib
from scipy.stats import zscore



def get_cortex_FC_profiles(output, subject, path_ctx_xs, hemi):

    # Load cortical time-series.
    ctx_xs_gii = nib.load(path_ctx_xs)
    ctx_xs     = np.array([darray.data for darray in ctx_xs_gii.darrays])
    _, n_vertex = ctx_xs.shape

    # Get average cortical time-series of cortex systems.
    system_xs = {}
    for  name in ['ant','body','post']:

        template = nib.load(f'templates/{name}_hipp_cortex_system.{hemi}.func.gii').darrays[0].data
        system_vertices = np.argwhere(template > 0).flatten()
        system_xs[name] = ctx_xs[:, system_vertices].mean(axis=1)

    # Get cortical connectivity profile of cortex systems.
    corrs = {'ant':[],'body':[],'post':[]}
    for vertex in range(n_vertex):
        for name in ['ant','body','post']:

            (_, r), _ = np.corrcoef(system_xs[name], ctx_xs[:,vertex])
            corrs[name].append(r)

    for name in ['ant','body','post']:
        corrs[name] = zscore(np.array(corrs[name]), nan_policy='omit')

    gii = utils.create_func_gii(list(corrs.values()), hemi=hemi, map_names=['ant','body','post'])
    nib.save(gii, f'{output}/{subject}/cortex_FC.{hemi}.func.gii')

    return


def get_hipp_FC_profiles(output, subject, path_hipp_systems, path_hipp_xs, hemi):

    # Load hippocampal systems and time-series.
    hipp_systems = nib.load(path_hipp_systems).darrays[0].data
    hipp_xs_gii  = nib.load(path_hipp_xs)
    hipp_xs      = np.array([darray.data for darray in hipp_xs_gii.darrays])

    _, n_vertices = hipp_xs.shape

    corrs = {'ant':[],'body':[],'post':[]}
    for idx, system in enumerate(['ant','body','post']):

        xs = hipp_xs[:,hipp_systems == idx].mean(axis=1)

        for vertex in range(n_vertices):
            (_, r), _ = np.corrcoef(xs, hipp_xs[:, vertex])
            corrs[system].append(r)

        corrs[system] = zscore(np.array(corrs[system]), nan_policy='omit')

    gii = utils.create_func_gii(list(corrs.values()), hemi=hemi, map_names=['ant','body','post'])
    nib.save(gii, f'{output}/{subject}/hipp_FC.{hemi}.func.gii')

    return


def get_hipp_cortex_FC_profiles(output, subject, path_hipp_systems, path_hipp_xs, path_ctx_xs, hemi):

    # Load hippocampal systems and time-series.
    hipp_systems = nib.load(path_hipp_systems).darrays[0].data
    hipp_xs_gii  = nib.load(path_hipp_xs)
    hipp_xs      = np.array([darray.data for darray in hipp_xs_gii.darrays])

    # Load cortical time-series.
    ctx_xs_gii = nib.load(path_ctx_xs)
    ctx_xs     = np.array([darray.data for darray in ctx_xs_gii.darrays])
    _, n_vertex = ctx_xs.shape

    # Get average hippocampal system time-series.
    hipp_system_xs = {}
    for idx, system in enumerate(['ant','body','post']):
        sys_vertices = np.argwhere(hipp_systems == idx).flatten()
        hipp_system_xs[system] = hipp_xs[:, sys_vertices].mean(axis=1)

    # Get cortical connectivity profile of hippocampal systems.
    corrs = {'ant':[],'body':[],'post':[]}
    for vertex in range(n_vertex):
        for system in ['ant','body','post']:

            system_xs = hipp_system_xs[system]
            (_, r), _ = np.corrcoef(system_xs, ctx_xs[:,vertex])
            corrs[system].append(r)

    for system in ['ant','body','post']:
        corrs[system] = zscore(np.array(corrs[system]), nan_policy='omit')

    gii = utils.create_func_gii(list(corrs.values()), hemi=hemi, map_names=['ant','body','post'])
    nib.save(gii, f'{output}/{subject}/hipp_cortex_FC.{hemi}.func.gii')

    return

