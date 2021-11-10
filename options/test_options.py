# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_eval_parameters()

    def add_eval_parameters(self):
        eval_params = self.parser.add_argument_group("eval")

        eval_params.add_argument("--old_model", type=str, default="")
        eval_params.add_argument("--short_name", type=str, default="")
        eval_params.add_argument("--result_folder", type=str, default="")
        eval_params.add_argument("--test_folder", type=str, default="")
        eval_params.add_argument("--model_setting", type=str, choices=("train", "gen_paired_img", "gen_img", \
                                    "gen_scene", 'get_gen_order', 'gen_two_imgs'), default="train")
        eval_params.add_argument("--dataset_folder", type=str, default="")
        eval_params.add_argument("--demo_img_name", type=str, default="")
        eval_params.add_argument("--gt_folder", type=str, default="")
        eval_params.add_argument("--batch_size", type=int, default=1)
        eval_params.add_argument("--num_views", type=int, default=2)
        eval_params.add_argument("--num_workers", type=int, default=1)
        eval_params.add_argument(
            "--sampling_mixture_temp",
            type=float,
            default=1.0,
            help="mixture sampling temperature",
        )
        eval_params.add_argument(
            "--num_samples",
            type=int,
            default=1,
            help="num samples from which to optimize",
        )
        eval_params.add_argument(
            "--sampling_logistic_temp",
            type=float,
            default=1.0,
            help="logistic sampling temperature",
        )
        eval_params.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="temperature for vqvae",
        )
        eval_params.add_argument(
            "--temp_eps",
            type=float,
            default=0.05,
            help="max / min temp (distance from 0 & 1) when drawing during sampling",
        )
        eval_params.add_argument(
            "--rotation",
            type=float,
            default=0.3,
            help="rotation (in radians) of camera for image generation",
        )
        eval_params.add_argument(
            "--decoder_truncation_threshold",
            type=float,
            default=2,
            help="resample if value above this drawn for decoder sampling",
        )
        eval_params.add_argument(
            "--homography", action="store_true", default=False
        ) 
        eval_params.add_argument(
            "--load_autoregressive", action="store_true", default=False
        ) 
        eval_params.add_argument("--no_outpainting", action="store_true", default=False)
        eval_params.add_argument(
            "--render_ids", type=int, nargs="+", default=[1]
        )
        eval_params.add_argument(
            "--directions", type=str, nargs="+", default=[], help="directions for scene generation"
        )
        eval_params.add_argument(
            "--direction", type=str, default="", help="direction for image generation"
        )
        eval_params.add_argument(
            "--background_smoothing_kernel_size", type=int, default=13
        )
        eval_params.add_argument(
            "--normalize_before_residual", action="store_true", default=False
        )
        eval_params.add_argument(
            "--sequential_outpainting", action="store_true", default=False
        )
        eval_params.add_argument(
            "--pretrain", action="store_true", default=False
        )
        eval_params.add_argument(
            "--val_rotation",
            type=int,
            default=10,
            help="size of rotation in single l/r direction in degrees for validation",
        )
        eval_params.add_argument(
            "--num_visualize_imgs", type=int, default=10
        )
        eval_params.add_argument(
            "--eval_iters", type=int, default=3600
        )
        eval_params.add_argument(
            "--eval_real_estate", action="store_true", default=False
        )
        eval_params.add_argument(
            "--intermediate", action="store_true", default=False
        )
        eval_params.add_argument(
            "--gen_fs", action="store_true", default=False
        )
        eval_params.add_argument(
            "--gen_order", action="store_true", default=False
        )
        eval_params.add_argument(
            "--gt_histogram", 
            type=str,
            help="rgb",
        )
        eval_params.add_argument(
            "--pred_histogram", 
            type=str,
            help="rgb",
        )
        eval_params.add_argument(
            "--image_type",
            type=str,
            default="both",
            choices=(
                "both"
            ),
        )
        eval_params.add_argument("--gpu_ids", type=str, default="0")
        eval_params.add_argument("--images_before_reset", type=int, default=100)
        eval_params.add_argument(
            "--test_input_image", action="store_true", default=False
        )
        eval_params.add_argument(
            "--use_custom_testset", action="store_true", default=False
        )
        eval_params.add_argument(
            "--use_fixed_testset", action="store_true", default=False
        )
        eval_params.add_argument(
            "--use_videos", action="store_true", default=False
        )
        eval_params.add_argument("--autoregressive", type=str, default="")
        eval_params.add_argument(
            "--num_split",
            type=int,
            default=1,
            help='number to split autoregressive steps into'
        )
        eval_params.add_argument(
            "--vqvae",action="store_true", default=False,
        )
        eval_params.add_argument(
            "--load_vqvae",action="store_true", default=False,
        )
        eval_params.add_argument("--vqvae_path", type=str, default="")
        eval_params.add_argument("--use_gt", action="store_true", default=False)
        eval_params.add_argument("--save_data", action="store_true", default=False)
        eval_params.add_argument("--dataset", type=str, default="")
        eval_params.add_argument(
            "--use_higher_res", action="store_true", default=False
        )
        eval_params.add_argument(
            "--use_3_discrim", action="store_true", default=False
        )
        eval_params.add_argument(
            "--max_rotation",
            type=int,
            default=10,
            help="size of rotation in single l/r direction in degrees (double for max difference)",
        )
        eval_params.add_argument(
            "--num_beam_samples",
            type=int,
            default=1,
            help="number of samples per beam",
        )
        eval_params.add_argument(
            "--num_beams",
            type=int,
            default=1,
            help="number of beams to sample",
        )

    def parse(self, arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())

        arg_groups = {}
        for group in self.parser._action_groups:
            group_dict = {
                a.dest: getattr(args, a.dest, None)
                for a in group._group_actions
            }
            arg_groups[group.title] = group_dict

        return (args, arg_groups)
