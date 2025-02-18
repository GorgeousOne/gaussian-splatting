import torch
from random import randint
import sys
from scene import Scene, GaussianModel
from gaussian_renderer import render
import depth_pruning.training_render as tr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams



from scene.cameras import Camera
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
import torch.nn.functional as F

def render_normals(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    cov3D_precomp = None

    scales = pc.get_scaling
    rotations = pc.get_rotation

    shs = None
    colors_precomp = get_viewspace_normals(viewpoint_camera, pc)

    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = F.normalize(rendered_image, dim=0)
    rendered_image = rendered_image * 0.5 + 0.5

    # i think this is not necessary... theoretically
    rendered_image = rendered_image.clamp(0, 1)

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }

    return out


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py 13.02.25
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

# https://github.com/maturk/dn-splatter/blob/main/dn_splatter/dn_model.py 12.02.25
def get_viewspace_normals(viewpoint_camera:Camera, pc:GaussianModel):
    scales = pc.get_scaling
    quats = pc.get_rotation
    cam_pos = torch.from_numpy(viewpoint_camera.T).float().cuda()
    cam_rot = torch.from_numpy(viewpoint_camera.R).float().cuda()

    # get the normals of the gaussians (axis vector of smallest radius, but rotated)
    normals = F.one_hot(torch.argmin(scales, dim=-1), num_classes=3).float()
    rots = quaternion_to_matrix(quats)
    normals = F.normalize(normals, dim=1)
    normals = torch.bmm(rots, normals[:, :, None]).squeeze(-1)

    # calculate view direction
    viewdirs = pc.get_xyz - cam_pos
    viewdirs = F.normalize(viewdirs, dim=-1)

    # flip normals facing away from the camera (both sides of a gaussian would face the camera)
    dots = (normals * viewdirs).sum(-1)
    negative_dot_indices = dots < 0
    # normals[negative_dot_indices] = -normals[negative_dot_indices]

    # normals = normals @ cam_rot
    return normals


def training(dataset, opt, pipe, ply_path):
    gaussians = GaussianModel(dataset.sh_degree, "default")
    scene = Scene(dataset, gaussians)
    #dont load ply before scene initialization, otherwise gaussians get overwritten
    gaussians.load_ply(ply_path)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    cam_name = "DSC00087_DxO_83.jpg"
    cam_name = "DSC00074_DxO_71.jpg"
    viewpoint_cam = next(cam for cam in viewpoint_stack if cam.image_name == cam_name)
    # pipe.debug = True

    bg = torch.rand((3), device="cuda") if opt.random_background else background

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=False)
    image = render_pkg["render"]

    # render_pkg = render_normals(viewpoint_cam,gaussians,pipe,bg)
    render_pkg_n = render_normals(viewpoint_cam, gaussians, pipe, bg)
    image_n =render_pkg_n["render"]
    gt_image = viewpoint_cam.original_image.cuda()

    diff = torch.abs(image_n - image)
    tr.show_images_side_by_side(image_n, gt_image, None)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)
    # Start GUI server, configure and run training

    ply_path = "/home/mighty/repos/gaussian-splatting/output/02-06_esszimmer_vlate/point_cloud/iteration_30000/point_cloud.ply"

    training(lp.extract(args), op.extract(args), pp.extract(args), ply_path)
