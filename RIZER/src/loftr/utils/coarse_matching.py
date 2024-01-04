import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import faiss
import numpy as np
import time
import math
from matplotlib import pyplot as plt
INF = 1e9

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        if self.match_type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        elif self.match_type == 'sinkhorn':
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(
                torch.tensor(config['skh_init_bin_score'], requires_grad=True))
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
        else:
            raise NotImplementedError()

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_c0, feat_c1])

        feature1 = torch.squeeze(feat_c0, dim=0)
        feature2 = torch.squeeze(feat_c1, dim=0)
        feature1 = feature1.detach().cpu().contiguous().numpy()
        feature2= feature2.detach().cpu().contiguous().numpy()
        jihe_yuzhi = 0.1
        start_time = time.time()

        Length=int(math.sqrt(L))
        mask = np.ones((Length,Length))
        mask[:2, :] = 0
        mask[:, :2] = 0
        mask[-2:, :] = 0
        mask[:, -2:] = 0
        mask = mask.reshape(-1)

        def indice_xy(i_indices, j_indices):
            x1 = i_indices % Length
            y1 = i_indices // Length
            x2 = j_indices % Length
            y2 = j_indices // Length
            return (x1, y1, x2, y2)

        def theta_jisuan(x0, y0, x1, y1, x2, y2):
            len_AB = math.hypot(x0 - x1, y0 - y1)
            len_AC = math.hypot(x0 - x2, y0 - y2)
            dot_product = (x0 - x1) * (x0 - x2) + (y0 - y1) * (y0 - y2)
            cos_theta = dot_product / (len_AB * len_AC)
            if cos_theta > 1 - 1e-2:
                cos_theta = 1 - 1e-2
            elif cos_theta < -1 + 1e-2:
                cos_theta = -1 + 1e-2
            theta = math.acos(cos_theta)
            return theta

        def dist_zhixindu(dist_matrix1, dist_matrix2):
            dist_matrix = dist_matrix1 / dist_matrix2
            dist_center = np.mean(dist_matrix)
            dist_matrix_xishu = abs((dist_matrix / dist_center) - 1)
            dist_means = np.mean(dist_matrix_xishu, axis=1)
            return dist_means

        def angle_zhixindu(angle_matrix1, angle_matrix2):
            angle_matrix = angle_matrix1 / angle_matrix2
            angle_matrix_xishu = abs(angle_matrix - 1)
            angle_means = np.mean(angle_matrix_xishu, axis=1)
            return angle_means

        def HNSW(index_feature, query_feature):
            index = faiss.IndexHNSWFlat(256, 32)
            index.hnsw.efConstruction = 16
            index.hnsw.efSearch = 16
            index.add(index_feature)
            # query hnsw
            top_k = 3
            yuzhi = 0.85
            distance, preds = index.search(query_feature, k=top_k)
            result = distance[:, 0] / distance[:, 1]
            result[result > yuzhi] = 0
            result = result * mask
            i_indices = np.nonzero(result)[0]
            j_indices = preds[i_indices, 0]
            index_img = [[a, b] for a, b in zip(i_indices, j_indices)]
            return distance, preds, index_img

        distance1, preds1, index_img1 = HNSW(feature2, feature1)
        distance2, preds2, index_img2 = HNSW(feature1, feature2)


        keys = set()
        for x, y in index_img1:
            keys.add(f"{x},{y}")
        matching = []
        for i, (x, y) in enumerate(index_img2):
            if f"{y},{x}" in keys:
                matching.append((y, x))
        i_indices, j_indices = zip(*matching)
        i_indices, j_indices = np.array(i_indices), np.array(j_indices)
        (x1, y1, x2, y2) = indice_xy(i_indices, j_indices)

        end_time1 = time.time()
        run_time1 = end_time1 - start_time
        run_time1 = '{:.4f}'.format(run_time1)

        num_samples = 5
        index_matrix = np.zeros((len(matching), num_samples))
        for i in range(len(matching)):
            indices = np.arange(len(matching))
            indices = indices[indices != i]
            index_matrix[i] = np.random.choice(indices, num_samples, replace=False)
        index_matrix = index_matrix.astype(int)

        dist_matrix1, dist_matrix2, angle_matrix1, angle_matrix2 = \
            [np.zeros((len(matching), num_samples)) for _ in range(4)]
        for i in range(len(matching)):
            for j in range(num_samples):
                idx = index_matrix[i, j]
                dist1 = np.sqrt((x1[i] - x1[idx]) ** 2 + (y1[i] - y1[idx]) ** 2)
                dist2 = np.sqrt((x2[i] - x2[idx]) ** 2 + (y2[i] - y2[idx]) ** 2)

                if j == 0:
                    theta1 = theta_jisuan(x1[i], y1[i], x1[index_matrix[i, 4]], y1[index_matrix[i, 4]], x1[idx],
                                          y1[idx])
                    theta2 = theta_jisuan(x2[i], y2[i], x2[index_matrix[i, 4]], y2[index_matrix[i, 4]], x2[idx],
                                          y2[idx])
                else:
                    theta1 = theta_jisuan(x1[i], y1[i], x1[idx], y1[idx], x21, y21)
                    theta2 = theta_jisuan(x2[i], y2[i], x2[idx], y2[idx], x22, y22)

                angle_matrix1[i, j] = math.degrees(theta1 + 0.1)  # 防止除数为0
                angle_matrix2[i, j] = math.degrees(theta2 + 0.1)
                dist_matrix1[i, j] = dist1
                dist_matrix2[i, j] = dist2

                x21 = x1[idx]
                y21 = y1[idx]
                x22 = x2[idx]
                y22 = y2[idx]

        dist_means = dist_zhixindu(dist_matrix1, dist_matrix2)
        angle_means = angle_zhixindu(angle_matrix1, angle_matrix2)
        idx_del = np.where((dist_means < jihe_yuzhi) & (angle_means < jihe_yuzhi))[0]
        x1_del, y1_del, x2_del, y2_del = x1[idx_del], y1[idx_del], x2[idx_del], y2[idx_del]
        i_indices1 = y1_del * Length + x1_del
        j_indices1 = y2_del * Length + x2_del
        DIST1 = distance1[i_indices1, 0]
        DIST1 = np.sqrt(DIST1)
        print(f"The Control Matchpoints：{len(i_indices1)}")

        ##最小二乘拟合
        X = np.stack([x2_del, y2_del, x2_del * y2_del, np.ones(len(x2_del))], axis=1)
        Y = np.stack([x1_del, y1_del], axis=1)
        rcond = np.finfo(X.dtype).eps * max(X.shape)
        coeff = np.linalg.lstsq(X, Y, rcond=rcond)[0]
        print(f"The Least Squares Fitting Transform Model：{coeff}")


        j_else = np.ones(L, dtype=int)
        j_else = j_else * mask
        j_else[j_indices] = 0
        j_else_idx = np.nonzero(j_else)[0]
        x2_else = j_else_idx % Length
        y2_else = j_else_idx // Length
        feat = np.stack([x2_else, y2_else, x2_else * y2_else, np.ones(len(x2_else))], axis=1)
        x1_pre = np.rint(feat.dot(coeff[:, 0]))
        x1_pre = x1_pre.astype(int)
        y1_pre = np.rint(feat.dot(coeff[:, 1]))
        y1_pre = y1_pre.astype(int)

        mask1 = (y1_pre > 1) & (y1_pre < Length-2)
        x1_filtered = x1_pre[mask1]
        y1_filtered = y1_pre[mask1]
        x2_else = x2_else[mask1]
        y2_else = y2_else[mask1]
        mask2 = (x1_filtered > 1) & (x1_filtered < Length-2)
        y1_predict = y1_filtered[mask2]
        x1_predict = x1_filtered[mask2]
        x2_else = x2_else[mask2]
        y2_else = y2_else[mask2]

        j_else = y2_else * Length + x2_else
        i_predict = y1_predict * Length + x1_predict
        j_else_same, i_predict_same, j_else_unsame, i_predict_unsame, DIST2 = [[] for _ in range(5)]
        for i, j in enumerate(j_else):
            row = preds2[j]
            dist = distance2[j]
            if np.any(row == i_predict[i]):

                index = np.array(np.where(row == i_predict[i])[0])
                DIST2.append(dist[index])
                j_else_same.append(j)
                i_predict_same.append(i_predict[i])
            else:

                j_else_unsame.append(j)
                i_predict_unsame.append(i_predict[i])
        DIST2 = np.concatenate(DIST2)
        DIST2 = np.sqrt(DIST2)
        j_indices2 = np.array(j_else_same)
        i_indices2 = np.array(i_predict_same)
        j_else_unsame = np.array(j_else_unsame)
        i_predict_unsame = np.array(i_predict_unsame)
        print(f"The Three-Nearest-Neighbor Matchpoints：{len(j_indices2)}")
        (x1_unsame, y1_unsame, x2_unsame, y2_unsame) = indice_xy(i_predict_unsame, j_else_unsame)

        #####################################################################
        start_time2 = time.time()
        index_matrix_yuce = np.zeros((len(j_else_unsame), num_samples))
        for i in range(len(j_else_unsame)):
            indices = np.arange(len(i_indices1))  # 0到len(vec)-1的索引数组
            index_matrix_yuce[i] = np.random.choice(indices, num_samples, replace=False)
        index_matrix_yuce = index_matrix_yuce.astype(int)

        dist_matrix1_yuce, dist_matrix2_yuce, angle_matrix1_yuce, angle_matrix2_yuce = \
            [np.zeros((len(j_else_unsame), num_samples)) for _ in range(4)]
        for i in range(len(j_else_unsame)):
            for j in range(num_samples):
                idx = index_matrix_yuce[i, j]
                dist1 = np.sqrt((x1_unsame[i] - x1_del[idx]) ** 2 + (y1_unsame[i] - y1_del[idx]) ** 2)
                dist2 = np.sqrt((x2_unsame[i] - x2_del[idx]) ** 2 + (y2_unsame[i] - y2_del[idx]) ** 2)
                if j == 0:
                    theta1 = theta_jisuan(x1_unsame[i], y1_unsame[i], x1_del[index_matrix_yuce[i, 4]],
                                          y1_del[index_matrix_yuce[i, 4]], x1_del[idx], y1_del[idx])
                    theta2 = theta_jisuan(x2_unsame[i], y2_unsame[i], x2_del[index_matrix_yuce[i, 4]],
                                          y2_del[index_matrix_yuce[i, 4]], x2_del[idx], y2_del[idx])
                else:
                    theta1 = theta_jisuan(x1_unsame[i], y1_unsame[i], x1_del[idx], y1_del[idx], x21, y21)
                    theta2 = theta_jisuan(x2_unsame[i], y2_unsame[i], x2_del[idx], y2_del[idx], x22, y22)

                angle_matrix1_yuce[i, j] = math.degrees(theta1 + 0.1)
                angle_matrix2_yuce[i, j] = math.degrees(theta2 + 0.1)
                dist_matrix1_yuce[i, j] = dist1
                dist_matrix2_yuce[i, j] = dist2
                x21, y21, x22, y22 = x1[idx], y1[idx], x2[idx], y2[idx]

        dist_means_yuce = dist_zhixindu(dist_matrix1_yuce, dist_matrix2_yuce)
        angle_means_yuce = angle_zhixindu(angle_matrix1_yuce, angle_matrix2_yuce)
        idx_del_yuce = np.where((dist_means_yuce < 0.05) & (angle_means_yuce < 0.05))[0]
        x1_del = x1_unsame[idx_del_yuce]
        y1_del = y1_unsame[idx_del_yuce]
        x2_del = x2_unsame[idx_del_yuce]
        y2_del = y2_unsame[idx_del_yuce]
        i_indices3 = y1_del * Length + x1_del
        j_indices3 = y2_del * Length + x2_del
        feature1_3 = feature1[i_indices3, :]
        feature2_3 = feature2[i_indices3, :]
        diff = feature1_3 - feature2_3
        sq_diff = diff ** 2
        sq_diff = torch.from_numpy(sq_diff)
        sum_sq_diff = torch.sum(sq_diff, dim=1)
        dist_3 = np.sqrt(sum_sq_diff)
        DIST3 = np.array(dist_3)
        end_time2 = time.time()
        run_time2 = end_time2 - start_time2
        print(f"The Spatial Structural Similarity Matchpoints：{len(i_indices3)}")

        total_len = len(i_indices1) + len(i_indices2) + len(i_indices3)
        i_indices_all = np.empty(total_len, dtype=i_indices1.dtype)
        j_indices_all = np.empty(total_len, dtype=j_indices1.dtype)
        DIST = np.empty(total_len, dtype=i_indices1.dtype)
        i_indices_all = np.concatenate((i_indices1, i_indices2, i_indices3), axis=0)
        j_indices_all = np.concatenate((j_indices1, j_indices2, j_indices3), axis=0)
        DIST = np.concatenate((DIST1, DIST2, DIST3), axis=0)
        mconf = torch.from_numpy(1 / DIST)
        print(f"The Coarse-Level Matchpoints：{len(i_indices_all)}")

        i_ids = torch.from_numpy(i_indices_all)
        j_ids = torch.from_numpy(j_indices_all)
        b_ids = torch.zeros(len(i_ids), dtype=torch.float)
        b_ids = b_ids.long()
        i_ids = i_ids.long()
        j_ids = j_ids.long()

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}
        ##coarse_matches
        # 4. Update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        ##scale0对scale进行缩放校正,考虑了图像0可能被预处理缩放的情况。
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
            dim=1) * scale1
        # These matches is the current prediction (for visualization)
        xuhao123 = (i_indices1.shape[0], i_indices1.shape[0] + i_indices2.shape[0],
                    i_indices1.shape[0] + i_indices2.shape[0] + i_indices3.shape[0])
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],  ##（2441，2）
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0],
            'xuhao': xuhao123
        })
        data.update(coarse_matches)

        # data.update(**self.get_coarse_match(conf_matrix, data))



