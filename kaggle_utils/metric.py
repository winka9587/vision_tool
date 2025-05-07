# This is the IMC 3D error metric code

import warnings
import csv
import math
import numpy as np


_EPS = np.finfo(float).eps * 4.0


def read_csv(filename, header=True, print_header=False):
    data = {}
    label_idx = {}    
    
    with open(filename, newline='\n') as csvfile:    
        csv_lines = csv.reader(csvfile, delimiter=',')
        for row in csv_lines:
            if header:
                header = False
                for i, name in enumerate(row): label_idx[name] = i
                if print_header:
                    print(f'Skipping header for file {filename}: {row}')
                continue
            dataset = row[label_idx['dataset']]
            scene = row[label_idx['scene']]
            image = row[label_idx['image']]
            R = np.array([float(x) for x in (row[label_idx['rotation_matrix']].split(';'))]).reshape(3,3)
            t = np.array([float(x) for x in (row[label_idx['translation_vector']].split(';'))]).reshape(3)
            c = -R.T @ t

            if not (dataset in data):
                data[dataset] = {}            
            if not (scene in data[dataset]):
                data[dataset][scene] = {}
            data[dataset][scene][image] = {'R': R, 't': t, 'c': c}
    return data


def quaternion_matrix(quaternion):
    '''Return homogeneous rotation matrix from quaternion.'''
    
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        # print("special case")
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [
                1.0 - q[2, 2] - q[3, 3],
                q[1, 2] - q[3, 0],
                q[1, 3] + q[2, 0],
                0.0,
            ],
            [
                q[1, 2] + q[3, 0],
                1.0 - q[1, 1] - q[3, 3],
                q[2, 3] - q[1, 0],
                0.0,
            ],
            [
                q[1, 3] - q[2, 0],
                q[2, 3] + q[1, 0],
                1.0 - q[1, 1] - q[2, 2],
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def mAA_on_cameras(err, thresholds, n, skip_top_thresholds, to_dec=3):
    '''mAA is the mean of mAA_i, where for each threshold th_i in <thresholds>, excluding the first <skip_top_thresholds values>,
    mAA_i = max(0, sum(err_i < th_i) - <to_dec>) / (n - <to_dec>)
    where <n> is the number of ground-truth cameras and err_i is the camera registration error for the best 
    registration corresponding to threshold th_i'''
    
    aux = err[:, skip_top_thresholds:] < np.expand_dims(np.asarray(thresholds[skip_top_thresholds:]), axis=0)
    numerator = np.sum(np.maximum(np.sum(aux, axis=0) - to_dec, 0))
    # Skip warnings.
    return 0 if numerator == 0 else numerator / (len(thresholds[skip_top_thresholds:]) * (n - to_dec))


def mAA_on_cameras_per_th(err, thresholds, n, to_dec=3):
    '''as mAA_on_cameras, to be used in score_all_ext with per_th=True'''
    aux = err < np.expand_dims(np.asarray(thresholds), axis=0)
    return np.maximum(np.sum(aux, axis=0) - to_dec, 0) / (n - to_dec)


def check_data(gt_data, user_data, print_error=False):    
    '''check if the gt/submission data are correct -
    <gt_data> - images in different scenes in the same dataset cannot have the same name
    <user_data> - there must be exactly an entry for each dataset, scene, image entry in the gt
    <print_error> - print the error *ATTENTION: must be disable when called from score_all_ext to avoid possible data leaks!*'''
    
    for dataset in gt_data.keys():
        aux = {}
        for scene in gt_data[dataset].keys():
            for image in gt_data[dataset][scene].keys():
                if image in aux:
                    if print_error: warnings.warn(f'image {image} found duplicated in the GT dataset {dataset}')
                    return False
                else:
                    aux[image] = 1

        if not dataset in user_data.keys():
            if print_error: warnings.warn(f'dataset {dataset} not found in submission')
            return False
        
        for scene in user_data[dataset].keys():
            for image in user_data[dataset][scene].keys():
                if not (image in aux):
                    if print_error: warnings.warn(f'image {image} does not belong to the GT dataset {dataset}')
                    return False
                else:
                    aux.pop(image)
 
        if len(aux) > 0:
            if print_error:  warnings.warn(f'submission dataset {dataset} missing some GT images')            
            return False           

    return True


def register_by_Horn(ev_coord, gt_coord, ransac_threshold, inl_cf, strict_cf):
    '''Return the best similarity transforms T that registers 3D points pt_ev in <ev_coord> to
    the corresponding ones pt_gt in <gt_coord> according to a RANSAC-like approach for each
    threshold value th in <ransac_threshold>.
    
    Given th, each triplet of 3D correspondences is examined if not already present as strict inlier,
    a correspondence is a strict inlier if <strict_cf> * err_best < th, where err_best is the registration
    error for the best model so far.
    The minimal model given by the triplet is then refined using also its inliers if their total is greater
    than <inl_cf> * ninl_best, where ninl_best is th number of inliers for the best model so far. Inliers
    are 3D correspondences (pt_ev, pt_gt) for which the Euclidean distance |pt_gt-T*pt_ev| is less than th.'''
    
    # remove invalid cameras, the index is returned
    idx_cams = np.all(np.isfinite(ev_coord), axis=0)
    ev_coord = ev_coord[:, idx_cams]
    gt_coord = gt_coord[:, idx_cams]

    # initialization
    n = ev_coord.shape[1]
    r = ransac_threshold.shape[0]
    ransac_threshold = np.expand_dims(ransac_threshold, axis=0)
    ransac_threshold2 = ransac_threshold**2
    ev_coord_1 = np.vstack((ev_coord, np.ones(n)))

    max_no_inl = np.zeros((1, r))
    best_inl_err = np.full(r, np.inf)
    best_transf_matrix = np.zeros((r, 4, 4))
    best_err = np.full((n, r), np.inf)
    strict_inl = np.full((n, r), False)
    triplets_used = np.zeros((3, r))

    # run on camera triplets
    for ii in range(n-2):
        for jj in range(ii+1, n-1):
            for kk in range(jj+1, n):
                i = [ii, jj, kk]
                triplets_used_now = np.full((n), False)
                triplets_used_now[i] = True
                # if both ii, jj, kk are strict inliers for the best current model just skip
                if np.all(strict_inl[i]):
                    continue
                # get transformation T by Horn on the triplet camera center correspondences
                transf_matrix = affine_matrix_from_points(ev_coord[:, i], gt_coord[:, i], usesvd=False)
                # apply transformation T to test camera centres
                rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                # compute error and inliers
                err = np.sum((rotranslated - gt_coord)**2, axis=0)
                inl = np.expand_dims(err, axis=1) < ransac_threshold2
                no_inl = np.sum(inl, axis=0)
                # if the number of inliers is close to that of the best model so far, go for refinement
                to_ref = np.squeeze(((no_inl > 2) & (no_inl > max_no_inl * inl_cf)), axis=0)
                for q in np.argwhere(to_ref):                        
                    qq = q[0]
                    if np.any(np.all((np.expand_dims(inl[:, qq], axis=1) == inl[:, :qq]), axis=0)):
                        # already done for this set of inliers
                        continue
                    # get transformation T by Horn on the inlier camera center correspondences
                    transf_matrix = affine_matrix_from_points(ev_coord[:, inl[:, qq]], gt_coord[:, inl[:, qq]])
                    # apply transformation T to test camera centres
                    rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)
                    # compute error and inliers
                    err_ref = np.sum((rotranslated - gt_coord)**2, axis=0)
                    err_ref_sum = np.sum(err_ref, axis=0)
                    err_ref = np.expand_dims(err_ref, axis=1)
                    inl_ref = err_ref < ransac_threshold2
                    no_inl_ref = np.sum(inl_ref, axis=0)
                    # update the model if better for each threshold
                    to_update = np.squeeze((no_inl_ref > max_no_inl) | ((no_inl_ref == max_no_inl) & (err_ref_sum < best_inl_err)), axis=0)
                    if np.any(to_update):
                        triplets_used[0, to_update] = ii
                        triplets_used[1, to_update] = jj
                        triplets_used[2, to_update] = kk
                        max_no_inl[:, to_update] = no_inl_ref[to_update]
                        best_err[:, to_update] = np.sqrt(err_ref)
                        best_inl_err[to_update] = err_ref_sum
                        strict_inl[:, to_update] = (best_err[:, to_update] < strict_cf * ransac_threshold[:, to_update])
                        best_transf_matrix[to_update] = transf_matrix

    best_model = {
        "valid_cams": idx_cams,        
        "no_inl": max_no_inl,
        "err": best_err,
        "triplets_used": triplets_used,
        "transf_matrix": best_transf_matrix}
    return best_model


def affine_matrix_from_points(v0, v1, shear=False, scale=True, usesvd=True):
    '''Return affine transform matrix to register two point sets.
    v0 and v1 are shape (ndims, -1) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.
    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean traffansformation matrix
    is returned.
    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.
    The returned matrix performs rotation, translation and uniform scaling
    (if specified).'''
    
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims: 2 * ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [
            [xx + yy + zz, 0.0, 0.0, 0.0],
            [yz - zy, xx - yy - zz, 0.0, 0.0],
            [zx - xz, xy + yx, yy - xx - zz, 0.0],
            [xy - yx, zx + xz, yz + zy, zz - xx - yy],
        ]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= np.linalg.norm(q + _EPS)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]

    return M


def tth_from_csv(csv_file):
    '''read thresholds from csv file <csv_file>'''

    tth = {}
    label_idx = {}    
    n_thresholds = []
    with open(csv_file, newline='\n') as csvfile:    
        csv_lines = csv.reader(csvfile, delimiter=',')
        header = True
        for row in csv_lines:
            if header:
                header = False
                for i, name in enumerate(row): label_idx[name] = i                                
                continue
            if not row:
                continue
            dataset = row[label_idx['dataset']]
            scene = row[label_idx['scene']]
            th = np.array([float(x) for x in (row[label_idx['thresholds']].split(';'))])
            n_thresholds.append(len(th))

            if not dataset in tth:
                tth[dataset] = {}
            tth[dataset][scene] = th
    if len(set(n_thresholds)) != 1:
        raise ValueError(f'Number of thresholds vary per scene: {list(set(n_thresholds))}')
                
    return tth, n_thresholds[0]


def generate_mask_all_public(gt_data):
    mask = {}
    for dataset in gt_data:
        if dataset not in mask:
            mask[dataset] = {}
        for scene in gt_data[dataset]:
            if scene not in mask[dataset]:
                mask[dataset][scene] = {}
            for image in gt_data[dataset][scene]:
                mask[dataset][scene][image] = True
    return mask


def fuse_score(mAA_score, cluster_score, combo_mode):
    if combo_mode =='harmonic':
        # it is basically the F1 score
        if (mAA_score + cluster_score) == 0:
            score = 0
        else:
            score = 2 * mAA_score * cluster_score / (mAA_score + cluster_score)
    elif combo_mode == 'geometric':
        score = (mAA_score * cluster_score) ** 0.5
    elif combo_mode == 'arithmetic':
        # to be avoided, since if one of the mAA or clusterness score is zero is not zero
        score = (mAA_score + cluster_score) * 0.5
    elif combo_mode == 'mAA':
        score = mAA_score
    elif combo_mode == 'clusterness':
        score = cluster_score
    
    return score


def get_clusterness_score(best_cluster, best_user_scene_sum):
    n = np.sum(best_cluster)
    m = np.sum(best_user_scene_sum)
    if m == 0:
        cluster_score = 0
    else:
        cluster_score = n / m  

    return cluster_score


def get_mAA_score(best_gt_scene_sum, best_gt_scene, thresholds, dataset, best_model, best_err, skip_top_thresholds, to_dec, lt):
    n = np.sum(best_gt_scene_sum)
    a = 0
    for i, scene in enumerate(best_gt_scene):
        ths = thresholds[dataset][scene]
        
        if len(best_model[i]) < 1:
            continue
        
        tmp = best_err[i][:, skip_top_thresholds:] < np.expand_dims(np.asarray(ths[skip_top_thresholds:]), axis=0)
        a = a + np.sum(np.maximum(np.sum(tmp, axis=0) - to_dec, 0))
        
    b = max(0, lt * (n - len(best_gt_scene) * to_dec))
    if b == 0:
        mAA_score = 0
    else:
        mAA_score = a / b        

    return mAA_score


def read_mask_csv(mask_filename='split_mask.csv'):
    '''IMC2025 read split labels'''

    data =  {}
    label_idx = {}
    with open(mask_filename, newline='\n') as csvfile:    
        csv_lines = csv.reader(csvfile, delimiter=',')
    
        header = True
        for row in csv_lines:
            if header:
                header = False
                for i, name in enumerate(row): label_idx[name] = i
                continue
            
            dataset = row[label_idx['dataset']]
            scene = row[label_idx['scene']]
            image = row[label_idx['image']]            
            label = row[label_idx['mask']] == 'True'

            if not (dataset in data):
                data[dataset] = {}
            
            if not (scene in data[dataset]):
                data[dataset][scene] = {}
                
            data[dataset][scene][image] = label
    
    return data


def score(
    *,
    gt_csv,
    user_csv,
    thresholds_csv,
    mask_csv=None,
    combo_mode='harmonic',
    inl_cf=0,
    strict_cf=-1,
    skip_top_thresholds=2,
    to_dec=3,
    verbose=False,
):
    '''compute the score: <gt_csv>/<user_csv> - gt/submission csv file;
    <combo_mode> - how to mix mAA_score and clusterness score ["harmonic", "geometric", "arithmetic"];
    <inl_cf>, <strict_cf>, <skip_threshold>, <to_dec> - parameters to be passed to mAA computation, see previous IMC challenge;
    <thresholds> - the threshold dict tth, <mask_csv> - public/private label csv file'''

    gt_data = read_csv(gt_csv)
    user_data = read_csv(user_csv)

    assert check_data(gt_data, user_data, print_error=True)
    
    mask = read_mask_csv(mask_csv) if mask_csv else generate_mask_all_public(gt_data)
    one_mask = 0
    all_mask = 0
    for dataset in mask:
        for scene in mask[dataset]:
            one_mask = one_mask + sum([1 for image in mask[dataset][scene] if mask[dataset][scene][image]])
            all_mask = all_mask + len(mask[dataset][scene])
    pct = one_mask / all_mask
    
    thresholds, th_n = tth_from_csv(thresholds_csv)
    lt = th_n - skip_top_thresholds

    # stat full
    stat_score = []
    stat_mAA = []
    stat_clusterness = []

    # stat public split
    stat_score_mask_a = []
    stat_mAA_mask_a = []
    stat_clusterness_mask_a = []

    # stat private split
    stat_score_mask_b = []
    stat_mAA_mask_b = []
    stat_clusterness_mask_b = []
        
    for dataset in gt_data.keys():
        gt_dataset = gt_data[dataset]
        user_dataset = user_data[dataset]

        lg = len(gt_dataset)
        lu = len(user_dataset)

        # full table               
        model_table = []
        err_table = []
        mAA_table = np.zeros((lg, lu))
        cluster_table = np.zeros((lg, lu))
        gt_scene_sum_table = np.zeros((lg, lu))
        user_scene_sum_table = np.zeros((lg, lu))

        # public split table               
        err_table_mask_a = []
        mAA_table_mask_a = np.zeros((lg, lu))
        cluster_table_mask_a = np.zeros((lg, lu))
        gt_scene_sum_table_mask_a = np.zeros((lg, lu))
        user_scene_sum_table_mask_a = np.zeros((lg, lu))

        # private split table               
        err_table_mask_b = []
        mAA_table_mask_b = np.zeros((lg, lu))
        cluster_table_mask_b = np.zeros((lg, lu))
        gt_scene_sum_table_mask_b = np.zeros((lg, lu))
        user_scene_sum_table_mask_b = np.zeros((lg, lu))

        # best full
        best_gt_scene = []
        best_user_scene = []
        best_model = []
        best_err = []
        best_mAA = np.zeros(lg)
        best_cluster = np.zeros(lg)
        best_gt_scene_sum = np.zeros(lg)
        best_user_scene_sum = np.zeros(lg)

        # best public split
        best_err_mask_a = []
        best_mAA_mask_a = np.zeros(lg)
        best_cluster_mask_a = np.zeros(lg)
        best_gt_scene_sum_mask_a = np.zeros(lg)
        best_user_scene_sum_mask_a = np.zeros(lg)

        # best private split
        best_err_mask_b = []
        best_mAA_mask_b = np.zeros(lg)
        best_cluster_mask_b = np.zeros(lg)
        best_gt_scene_sum_mask_b = np.zeros(lg)
        best_user_scene_sum_mask_b = np.zeros(lg)

        # all possible gt/submission cluster association per dataset
        gt_scene_list = []        
        for i, gt_scene in enumerate(gt_dataset.keys()):
            gt_scene_list.append(gt_scene)

            model_row = []
            err_row = []
            err_row_mask_a = []
            err_row_mask_b = []
            
            user_scene_list = []
            for j, user_scene in enumerate(user_dataset.keys()):                
                user_scene_list.append(user_scene)

                if (gt_scene == 'outliers') or (user_scene == 'outliers'):
                    model_row.append([])
                    err_row.append([])
                    err_row_mask_a.append([])
                    err_row_mask_b.append([])
                    continue
                                
                ths = thresholds[dataset][gt_scene]
                
                gt_cams = gt_data[dataset][gt_scene]
                user_cams = user_data[dataset][user_scene]
                                                
                # the denominator for mAA ratio
                m = len(gt_cams)
                m_mask_a = np.sum([mask[dataset][gt_scene][image] for image in mask[dataset][gt_scene].keys()])
                m_mask_b = np.sum([not mask[dataset][gt_scene][image] for image in mask[dataset][gt_scene].keys()])
                
                # get the image list to use
                good_cams = []
                for image_path in gt_cams.keys():
                    if image_path in user_cams.keys():
                        good_cams.append(image_path)                        
                
                good_cams_mask = []
                for image in good_cams:
                    good_cams_mask.append(mask[dataset][gt_scene][image])
                good_cams_mask_a = np.asarray(good_cams_mask)

                good_cams_mask = []
                for image in good_cams:
                    good_cams_mask.append(not mask[dataset][gt_scene][image])
                good_cams_mask_b = np.asarray(good_cams_mask)

                # put corresponding camera centers into matrices
                n = len(good_cams)
                n_mask_a = np.sum(good_cams_mask_a)
                n_mask_b = np.sum(good_cams_mask_b)
                
                u_cameras = np.zeros((3, n))
                g_cameras = np.zeros((3, n))
                
                ii = 0
                for k in good_cams:
                    u_cameras[:, ii] = user_cams[k]['c']
                    g_cameras[:, ii] = gt_cams[k]['c']
                    ii += 1
                    
                # Horn camera centers registration, a different best model for each camera threshold
                model = register_by_Horn(u_cameras, g_cameras, np.asarray(ths), inl_cf, strict_cf)

                # mAA                
                mAA = mAA_on_cameras(model["err"], ths, m, skip_top_thresholds, to_dec)
                
                if (len(model['valid_cams']) == 0) or (len(good_cams_mask_a) == 0): mAA_mask_a = np.float64(0.0)
                else: mAA_mask_a = mAA_on_cameras(model["err"][good_cams_mask_a[model['valid_cams']]], ths, m_mask_a, skip_top_thresholds, to_dec * pct)

                if (len(model['valid_cams']) == 0) or (len(good_cams_mask_b) == 0): mAA_mask_b = np.float64(0.0)
                else: mAA_mask_b = mAA_on_cameras(model["err"][good_cams_mask_b[model['valid_cams']]], ths, m_mask_b, skip_top_thresholds, to_dec * (1 - pct))
                
                len_user_scene = len(user_data[dataset][user_scene])
                
                aux_masked = {}
                masked_dataset = mask[dataset]
                for scene in masked_dataset.keys():
                    for image in masked_dataset[scene]:
                        aux_masked[image] = masked_dataset[scene][image]
                
                user_data_masked = []
                for image in user_data[dataset][user_scene]:
                    if (image in aux_masked): user_data_masked.append(aux_masked[image])
                    
                len_user_scene_mask_a = np.sum(np.asarray(user_data_masked))
                len_user_scene_mask_b = np.sum(~np.asarray(user_data_masked))

                # full                
                err_row.append(model["err"])
                mAA_table[i, j] = mAA                
                cluster_table[i, j] = n
                gt_scene_sum_table[i, j] = m
                user_scene_sum_table[i, j] = len_user_scene

                if (len(model['valid_cams']) == 0) or (len(good_cams_mask_a) == 0): err_row_mask_a.append(np.zeros((0, th_n)))
                else: err_row_mask_a.append(model["err"][good_cams_mask_a[model['valid_cams']]])

                if (len(model['valid_cams']) == 0) or (len(good_cams_mask_b) == 0): err_row_mask_b.append(np.zeros((0, th_n)))
                else: err_row_mask_b.append(model["err"][good_cams_mask_b[model['valid_cams']]])

                # public split
                mAA_table_mask_a[i, j] = mAA_mask_a              
                cluster_table_mask_a[i, j] = n_mask_a
                gt_scene_sum_table_mask_a[i, j] = m_mask_a
                user_scene_sum_table_mask_a[i, j] = len_user_scene_mask_a

                # private split
                mAA_table_mask_b[i, j] = mAA_mask_b              
                cluster_table_mask_b[i, j] = n_mask_b
                gt_scene_sum_table_mask_b[i, j] = m_mask_b
                user_scene_sum_table_mask_b[i, j] = len_user_scene_mask_b

                model_row.append(model)
    
            model_table.append(model_row)
            err_table.append(err_row)
            err_table_mask_a.append(err_row_mask_a)
            err_table_mask_b.append(err_row_mask_b)

        # best greedy cluster association per dataset
        for i, gt_scene in enumerate(gt_dataset.keys()):                                  
            best_ind = np.lexsort((-mAA_table[i], -cluster_table[i]))[0]
            best_gt_scene.append(gt_scene)
            best_user_scene.append(user_scene_list[best_ind])
            best_model.append(model_table[i][best_ind])

            # full
            best_err.append(err_table[i][best_ind])
            best_mAA[i] = mAA_table[i, best_ind]
            best_cluster[i] = cluster_table[i, best_ind]
            best_gt_scene_sum[i] = gt_scene_sum_table[i, best_ind]
            best_user_scene_sum[i] = user_scene_sum_table[i, best_ind]

            # public split
            best_err_mask_a.append(err_table_mask_a[i][best_ind])
            best_mAA_mask_a[i] = mAA_table_mask_a[i, best_ind]
            best_cluster_mask_a[i] = cluster_table_mask_a[i, best_ind]
            best_gt_scene_sum_mask_a[i] = gt_scene_sum_table_mask_a[i, best_ind]
            best_user_scene_sum_mask_a[i] = user_scene_sum_table_mask_a[i, best_ind]

            # private split
            best_err_mask_b.append(err_table_mask_b[i][best_ind])
            best_mAA_mask_b[i] = mAA_table_mask_b[i, best_ind]
            best_cluster_mask_b[i] = cluster_table_mask_b[i, best_ind]
            best_gt_scene_sum_mask_b[i] = gt_scene_sum_table_mask_b[i, best_ind]
            best_user_scene_sum_mask_b[i] = user_scene_sum_table_mask_b[i, best_ind]

        # exclude outliers cluster            
        outlier_idx = -1
        for i, scene in enumerate(best_gt_scene):
            if scene == 'outliers':
                outlier_idx = i
                break            
            
        if outlier_idx > -1:
            best_gt_scene.pop(outlier_idx)
            best_user_scene.pop(outlier_idx)
            best_model.pop(outlier_idx)
 
            # full            
            best_err.pop(outlier_idx)
            best_mAA = np.delete(best_mAA, outlier_idx)            
            best_cluster = np.delete(best_cluster, outlier_idx)            
            best_gt_scene_sum = np.delete(best_gt_scene_sum, outlier_idx)            
            best_user_scene_sum = np.delete(best_user_scene_sum, outlier_idx)

            # public split
            best_err_mask_a.pop(outlier_idx)
            best_mAA_mask_a = np.delete(best_mAA_mask_a, outlier_idx)            
            best_cluster_mask_a = np.delete(best_cluster_mask_a, outlier_idx)            
            best_gt_scene_sum_mask_a = np.delete(best_gt_scene_sum_mask_a, outlier_idx)            
            best_user_scene_sum_mask_a = np.delete(best_user_scene_sum_mask_a, outlier_idx)

            # private split
            best_err_mask_b.pop(outlier_idx)
            best_mAA_mask_b = np.delete(best_mAA_mask_b, outlier_idx)            
            best_cluster_mask_b = np.delete(best_cluster_mask_b, outlier_idx)            
            best_gt_scene_sum_mask_b = np.delete(best_gt_scene_sum_mask_b, outlier_idx)            
            best_user_scene_sum_mask_b = np.delete(best_user_scene_sum_mask_b, outlier_idx)

        # compute the clusterness score
        # basically the precision: images in the both  gt and user cluster / images in the user cluster only
        cluster_score = get_clusterness_score(best_cluster, best_user_scene_sum)
        cluster_score_mask_a = get_clusterness_score(best_cluster_mask_a, best_user_scene_sum_mask_a)
        cluster_score_mask_b = get_clusterness_score(best_cluster_mask_b, best_user_scene_sum_mask_b)

        # compute the mAA score
        # basically the recall: images in the both gt and user cluster correctly registered / images in the gt cluster only
        mAA_score = get_mAA_score(best_gt_scene_sum, best_gt_scene, thresholds, dataset, best_model, best_err, skip_top_thresholds, to_dec, lt)            
        mAA_score_mask_a = get_mAA_score(best_gt_scene_sum_mask_a, best_gt_scene, thresholds, dataset, best_model, best_err_mask_a, skip_top_thresholds, to_dec * pct, lt)            
        mAA_score_mask_b = get_mAA_score(best_gt_scene_sum_mask_b, best_gt_scene, thresholds, dataset, best_model, best_err_mask_b, skip_top_thresholds, to_dec * (1 - pct), lt)            
            
        # merge mAA and clusterness score
        score = fuse_score(mAA_score, cluster_score, combo_mode)        
        score_mask_a = fuse_score(mAA_score_mask_a, cluster_score_mask_a, combo_mode)        
        score_mask_b = fuse_score(mAA_score_mask_b, cluster_score_mask_b, combo_mode)        

        if verbose:
            print(f'{dataset}: score={score * 100:.2f}% (mAA={mAA_score * 100:.2f}%, clusterness={cluster_score * 100:.2f}%)')
     
            if mask_csv:
                print(f'\tPublic split: score={score_mask_a * 100:.2f}% (mAA={mAA_score_mask_a * 100:.2f}%, clusterness={cluster_score_mask_a * 100:.2f}%)')
                print(f'\tPrivate split: score={score_mask_b * 100:.2f}% (mAA={mAA_score_mask_b * 100:.2f}%, clusterness={cluster_score_mask_b * 100:.2f}%)')

        # full
        stat_mAA.append(mAA_score)
        stat_clusterness.append(cluster_score)
        stat_score.append(score)
        
        # public split
        stat_mAA_mask_a.append(mAA_score_mask_a)
        stat_clusterness_mask_a.append(cluster_score_mask_a)
        stat_score_mask_a.append(score_mask_a)

        # public split
        stat_mAA_mask_b.append(mAA_score_mask_b)
        stat_clusterness_mask_b.append(cluster_score_mask_b)
        stat_score_mask_b.append(score_mask_b)

    # full
    final_score = 100 * np.mean(stat_score)
    final_mAA = 100 * np.mean(stat_mAA)
    final_clusterness = 100 * np.mean(stat_clusterness)

    # public split
    final_score_mask_a = 100 * np.mean(stat_score_mask_a)
    final_mAA_mask_a = 100 * np.mean(stat_mAA_mask_a)
    final_clusterness_mask_a = 100 * np.mean(stat_clusterness_mask_a)

    # private split
    final_score_mask_b = 100 * np.mean(stat_score_mask_b)
    final_mAA_mask_b = 100 * np.mean(stat_mAA_mask_b)
    final_clusterness_mask_b = 100 * np.mean(stat_clusterness_mask_b)

    if verbose:
        print(f'Average over all datasets: score={final_score:.2f}% (mAA={final_mAA:.2f}%, clusterness={final_clusterness:.2f}%)')
        if mask_csv:
            print(f'\tPublic split: score={final_score_mask_a:.2f}% (mAA={final_mAA_mask_a:.2f}%, clusterness={final_clusterness_mask_a:.2f}%)')
            print(f'\tPrivate split: score={final_score_mask_b:.2f}% (mAA={final_mAA_mask_b:.2f}%, clusterness={final_clusterness_mask_b:.2f}%)')

    scene_score_dict = {dataset: score * 100 for dataset, score in zip(gt_data, stat_score)}
    scene_score_dict_mask_a = None if mask_csv is None else {dataset: score * 100 for dataset, score in zip(gt_data, stat_score_mask_a)}
    scene_score_dict_mask_b = None if mask_csv is None else {dataset: score * 100 for dataset, score in zip(gt_data, stat_score_mask_b)}
    
    return (
        (final_score, final_score_mask_a, final_score_mask_b),
        (scene_score_dict, scene_score_dict_mask_a, scene_score_dict_mask_b)
    )

def score_transf(
    *,
    gt_csv,
    user_csv,
    thresholds_csv,
    mask_csv=None,
    combo_mode='harmonic',
    inl_cf=0,
    strict_cf=-1,
    skip_top_thresholds=2,
    to_dec=3,
    verbose=True,
):
    (
        (final_score, final_score_mask_a, final_score_mask_b),
        (scene_score_dict, scene_score_dict_mask_a, scene_score_dict_mask_b)
    ) = score(
        gt_csv=gt_csv,
        user_csv=user_csv,
        thresholds_csv=thresholds_csv,
        mask_csv=mask_csv,
        combo_mode=combo_mode,
        inl_cf=inl_cf,
        strict_cf=strict_cf,
        skip_top_thresholds=skip_top_thresholds,
        to_dec=to_dec,
        verbose=verbose,
    )

    gt_data = read_csv(gt_csv)
    user_data = read_csv(user_csv)
    # mask = read_mask_csv(mask_csv) if mask_csv else generate_mask_all_public(gt_data)

    thresholds, _ = tth_from_csv(thresholds_csv)

    dataset_transf = {}
    for dataset in gt_data:
        dataset_transf[dataset] = {}

        gt_dataset = gt_data[dataset]
        user_dataset = user_data[dataset]

        for i, gt_scene in enumerate(gt_dataset):
            best_transf = None
            best_score = -1
            gt_cams = gt_dataset[gt_scene]

            for j, user_scene in enumerate(user_dataset):
                if gt_scene == 'outliers' or user_scene == 'outliers':
                    continue

                ths = thresholds[dataset][gt_scene]
                user_cams = user_dataset[user_scene]

                good_cams = [img for img in gt_cams if img in user_cams]
                if not good_cams:
                    continue

                u_cameras = np.array([user_cams[k]['c'] for k in good_cams]).T
                g_cameras = np.array([gt_cams[k]['c'] for k in good_cams]).T

                model = register_by_Horn(u_cameras, g_cameras, np.asarray(ths), inl_cf, strict_cf)
                if len(model["valid_cams"]) == 0:
                    continue

                score_here = mAA_on_cameras(model["err"], ths, len(gt_cams), skip_top_thresholds, to_dec)
                if score_here > best_score:
                    transf_matrix = model["transf_matrix"]
                    if transf_matrix.ndim == 3:
                        transf_matrix = transf_matrix[0]  # 取第一个（默认只有一个）
                    R = transf_matrix[:3, :3]
                    t = transf_matrix[:3, 3]
                    best_transf = (R, t)

            if best_transf is not None:
                dataset_transf[dataset][gt_scene] = best_transf

    return (
        (final_score, final_score_mask_a, final_score_mask_b),
        (scene_score_dict, scene_score_dict_mask_a, scene_score_dict_mask_b),
        dataset_transf
    )
