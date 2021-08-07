import os
import numpy as np
'''
evalset = [0, 8, 15, 16, 21, 23, 25, 33, 35, 51, 73]
parent = 'output/realestate_final_eval/videos/e2e_vqvae_50comps_coordconv/'
for eval in evalset:
    dir='output/realestate_final_eval/videos/e2e_vqvae_50comps_coordconv/%04d/video/' % (eval)

    cmd = f"/Pool1/users/cnris/ffmpeg-git-20191111-amd64-static/ffmpeg -r 6 -i {dir}%d.png {parent}{str(eval)}.mp4"
    #ffmpeg -r 1/5 -i img%03d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p out.mp4
    if (os.system(cmd) == 0):
        print(f"    success making video at {parent}{str(eval)}!")
'''

# mp3d
#evalset = [1, 6, 10, 14, 18, 19, 36, 42, 51, 54, 55, 78, 81, 89, 109, 112, 122, 130, 132, 139, 189, 191, 194, 195, 196, \
#                205, 207, 210, 215, 230, 253, 282, 304, 308, 309, 335, 358, 390, 392, 503, 505, 510, 530, 532, 540, \
#                595, 596, 604, 606, 622, 633, 644, 645, 655, 665, 667, 669, 690, 699, 701]
#evalset = np.arange(1000,1114)
#[6]#[436,404,511,514,547,523,823,822,827,846,860,859,1008,1011,1013,1016]
#parent = 'output/mp3d_final_eval/videos/end_to_end_6x_vqvae_e2e_ep70_.7tmp/'
parent = 'output/realestate_final_eval_20-60/videos/e2e_final_20-60_e2e_vqvae_frozen_ep50_.7_50comps_ours/'
evalset = [1011, 1000, 606, 54, 1062, 1102]
for eval in evalset:
    #dir='output/mp3d_final_eval/videos/end_to_end_6x_vqvae_e2e_ep70_.7tmp/%04d/video/' % (eval)
    dir = os.path.join(parent,'%04d/video/' % (eval))

    cmd = f"/Pool1/users/cnris/ffmpeg-git-20191111-amd64-static/ffmpeg -framerate 6 -i {dir}%d.png -s 256x256 -pix_fmt yuv420p -vcodec libx264 -r 6 {parent}{str(eval)}_v2.mp4"
    #ffmpeg -r 1/5 -i img%03d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p out.mp4
    if (os.system(cmd) == 0):
        print(f"    success making video at {parent}{str(eval)}!")