import torch
import argparse
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

def save_camera_trajectory_tum(camera_poses, save_path):
    """
    camera_poses: torch.Tensor (N, 4, 4), Twc
    """
    poses = camera_poses.cpu().numpy()

    Rmat = poses[:, :3, :3]
    t = poses[:, :3, 3]
    q = R.from_matrix(Rmat).as_quat()  # (qx, qy, qz, qw)

    with open(save_path, 'w') as f:
        for i in range(len(poses)):
            f.write(
                f"{i/len(poses):.6f} "
                f"{t[i,0]:.6f} {t[i,1]:.6f} {t[i,2]:.6f} "
                f"{q[i,0]:.6f} {q[i,1]:.6f} {q[i,2]:.6f} {q[i,3]:.6f}\n"
            )

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")

    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")

    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # from pi3.utils.debug import setup_debug
    # setup_debug()

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file

            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)

        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input dataset
    # The load_images_as_tensor function will print the loading path
    imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device)  # (N, 3, H, W)

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None])  # Add batch dimension

    # 4. process mask
    # masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    # non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    # masks = torch.logical_and(masks, non_edge)[0]
    #
    # # 5. Save points
    # print(f"Saving point cloud to: {args.save_path}")
    # write_ply(res['points'][0][masks].cpu(), imgs.permute(0, 2, 3, 1)[masks], args.save_path)
    save_path = "./out"
    save_file = os.path.join(save_path,"estimate_traj")
    save_camera_trajectory_tum(res['camera_poses'],save_file)
    print("Done.")