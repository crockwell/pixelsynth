# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn

from pytorch3d.structures import Pointclouds

EPS = 1e-2


def get_splatter(
    name, depth_values, opt=None, size=256, C=64, points_per_pixel=8
):
    if name == "xyblending":
        from models.layers.z_buffer_layers import RasterizePointsXYsBlending

        return RasterizePointsXYsBlending(
            C,
            learn_feature=opt.learn_default_feature,
            radius=opt.radius,
            size=size,
            points_per_pixel=points_per_pixel,
            opts=opt,
        )
    else:
        raise NotImplementedError()


class PtsManipulator(nn.Module):
    def __init__(self, W, C=64, opt=None):
        super().__init__()
        self.opt = opt

        self.splatter = get_splatter(
            opt.splatter, None, opt, size=W, C=C, points_per_pixel=opt.pp_pixel
        )

        xs = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1
        ys = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1

        xs = xs.view(1, 1, 1, W).repeat(1, 1, W, 1)
        ys = ys.view(1, 1, W, 1).repeat(1, 1, 1, W)

        xyzs = torch.cat(
            (xs, -ys, -torch.ones(xs.size()), torch.ones(xs.size())), 1
        ).view(1, 4, -1)

        self.register_buffer("xyzs", xyzs)

    def project_pts(
        self, pts3D, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
    ):
        # PERFORM PROJECTION
        # Project the world points into the new view
        projected_coors = self.xyzs * pts3D
        projected_coors[:, -1, :] = 1

        # Transform into camera coordinate of the first view
        cam1_X = K_inv.bmm(projected_coors)

        # Transform into world coordinates
        RT = RT_cam2.bmm(RTinv_cam1)

        wrld_X = RT.bmm(cam1_X)

        # And intrinsics
        xy_proj = K.bmm(wrld_X)

        # And finally we project to get the final result
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / -zs, xy_proj[:, 2:3, :]), 1)
        sampler[mask.repeat(1, 3, 1)] = -10
        # Flip the ys
        sampler = sampler * torch.Tensor([1, -1, -1]).unsqueeze(0).unsqueeze(
            2
        ).to(sampler.device)

        return sampler

    def forward_justpts(
        self, src, pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
    ):
        # Now project these points into a new view
        bs, c, w, h = src.size()

        if len(pred_pts.size()) > 3:
            # reshape into the right positioning
            pred_pts = pred_pts.view(bs, 1, -1)
            src = src.view(bs, c, -1)

        pts3D = self.project_pts(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
        )

        
        
        pointcloud = pts3D.permute(0, 2, 1).contiguous()
        result, background_mask = self.splatter(pointcloud, src)
        #import pdb
        #pdb.set_trace()

        return result, background_mask

    def forward_justpts2(
        self, src, pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2, 
            src2,
            pred_pts2,
            RT_cam_half,
            RTinv_cam_half,
    ):
        # Now project these points into a new view
        bs, c, w, h = src.size()

        if len(pred_pts.size()) > 3:
            # reshape into the right positioning
            pred_pts = pred_pts.view(bs, 1, -1)
            pred_pts2 = pred_pts2.view(bs, 1, -1)
            src1 = src.view(bs, c, -1)
            src2 = src2.view(bs, c, -1)
            src = torch.cat([src1,src2],axis=2)

        pts3D = self.project_pts2(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2, 
            pred_pts2, RT_cam_half, RTinv_cam_half
        )


        
        pointcloud = pts3D.permute(0, 2, 1).contiguous()
        result, background_mask = self.splatter(pointcloud, src)
        #import pdb
        #pdb.set_trace()

        return result, background_mask

    def project_pts2(
        self, pts3D, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2, pts3D2, RT_cam_half, RTinv_cam_half
    ):
        # PERFORM PROJECTION
        # Project the world points into the new view
        projected_coors = self.xyzs * pts3D
        projected_coors[:, -1, :] = 1
        projected_coors2 = self.xyzs * pts3D2
        projected_coors2[:, -1, :] = 1

        # Transform into camera coordinate of the first view
        cam1_X = K_inv.bmm(projected_coors)
        cam1_X2 = K_inv.bmm(projected_coors2)

        # Transform into world coordinates
        RT = RT_cam2.bmm(RTinv_cam1)
        RT2 = RT_cam2.bmm(RTinv_cam_half)

        wrld_X = RT.bmm(cam1_X)
        wrld_X2 = RT2.bmm(cam1_X2)

        # And intrinsics
        xy_proj1 = K.bmm(wrld_X)
        xy_proj2 = K.bmm(wrld_X2)
        
        xy_proj = torch.cat([xy_proj1, xy_proj2], axis=2)

        # And finally we project to get the final result
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / -zs, xy_proj[:, 2:3, :]), 1)
        sampler[mask.repeat(1, 3, 1)] = -10
        # Flip the ys
        sampler = sampler * torch.Tensor([1, -1, -1]).unsqueeze(0).unsqueeze(
            2
        ).to(sampler.device)

        return sampler

    def forward_justpts_cumulative(
        self, src1, pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2, prior_point_cloud, src2, last_background_mask, RTinv_cam3
    ):
        # Now project these points into a new view
        bs, c, w, h = src1.size()

        # we only want to add new (outpainted) points to the point cloud
        # so we multiple the inputs by whether the last_background_mask is True
        if last_background_mask is not None:
            last_background_mask = last_background_mask.view(bs, 1, -1)

        if len(pred_pts.size()) > 3:
            # reshape into the right positioning
            pred_pts = pred_pts.view(bs, 1, -1)
            src1 = src1.view(bs, c, -1)
            if src2 is not None:
                #print('b4',src1.shape, pred_pts.shape)
                pred_pts = pred_pts[last_background_mask==True].view(bs, 1, -1)
                src1 = src1[last_background_mask.repeat(1,c,1)==True].view(bs, c, -1)

                src2 = src2.view(bs, c, -1)
                #print(src1.shape, pred_pts.shape)
                src = torch.cat([src1,src2],axis=2)
            else:
                src = src1

        pts3D, new_point_cloud = self.project_pts_cumulative(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2, prior_point_cloud, last_background_mask, RTinv_cam3
        )
        
        pointcloud = pts3D.permute(0, 2, 1).contiguous()
        result, background_mask = self.splatter(pointcloud, src)
        #import pdb
        #pdb.set_trace()

        return result, background_mask, new_point_cloud, src

    def project_pts_cumulative(
        self, pts3D, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2, prior_point_cloud=None, last_background_mask=None, RTinv_cam3=None
    ):
        # PERFORM PROJECTION
        # Project the world points into the new view
        if last_background_mask is not None:
            bs, c, l = self.xyzs.shape
            projected_coors = self.xyzs[last_background_mask.repeat(1,4,1)==True].view([bs,c,-1]) * pts3D
        else:
            projected_coors = self.xyzs * pts3D
        projected_coors[:, -1, :] = 1

        # Transform into camera coordinate of the first view
        cam1_X = K_inv.bmm(projected_coors)

        # Transform into world coordinates
        RT = RT_cam2.bmm(RTinv_cam1)

        wrld_X = RT.bmm(cam1_X)

        # And intrinsics
        xy_proj1 = K.bmm(wrld_X)
        
        if prior_point_cloud is not None:
            RT_last = RT_cam2.bmm(RTinv_cam3)
            wrld_Xlast = RT_last.bmm(prior_point_cloud)
            xy_proj2 = K.bmm(wrld_Xlast)
            xy_proj = torch.cat([xy_proj1, xy_proj2], axis=2)
        else:
            xy_proj = xy_proj1

        # And finally we project to get the final result
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / -zs, xy_proj[:, 2:3, :]), 1)
        sampler[mask.repeat(1, 3, 1)] = -10
        # Flip the ys
        sampler = sampler * torch.Tensor([1, -1, -1]).unsqueeze(0).unsqueeze(
            2
        ).to(sampler.device)

        return sampler, xy_proj

    def forward(
        self,
        alphas,
        src,
        pred_pts,
        K,
        K_inv,
        RT_cam1,
        RTinv_cam1,
        RT_cam2,
        RTinv_cam2,
    ):
        # Now project these points into a new view
        bs, c, w, h = src.size()

        if len(pred_pts.size()) > 3:
            # reshape into the right positioning
            pred_pts = pred_pts.view(bs, 1, -1)
            src = src.view(bs, c, -1)
            alphas = alphas.view(bs, 1, -1).permute(0, 2, 1).contiguous()

        pts3D = self.project_pts(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
        )
        result = self.splatter(pts3D.permute(0, 2, 1).contiguous(), alphas, src)

        return result
