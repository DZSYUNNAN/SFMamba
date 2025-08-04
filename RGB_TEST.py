import cv2
import numpy as np
import os
import torch
import time
from PIL import Image, ImageOps
from models.vmamba_Fusion_efficross import VSSM_Fusion as net

# from loss import Fusionloss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = net(in_channel=1)
# model_path = "G:\dzs\FusionMamba-main\FusionMamba_model/KAIST_1ssim_10int_1grad_.pth" # put the path of model.pth
model_path = "/media/dzs/系统/dzs/FusionMamba-main/our_checkpoints/msrs/fusion_model_10ssim101_2kd.pth"
use_gpu = torch.cuda.is_available()

if use_gpu:
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
else:
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)


def imresize(arr, size, interp='bilinear', mode=None):
    numpydata = np.asarray(arr)
    im = Image.fromarray(numpydata, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size) * percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return np.array(imnew)


def resize(image1, image2, crop_size_img, crop_size_label):
    image1 = imresize(image1, crop_size_img, interp='bicubic')
    image2 = imresize(image2, crop_size_label, interp='bicubic')
    return image1, image2


def get_image_files(input_folder):
    valid_extensions = (".bmp", ".tif", ".jpg", ".jpeg", ".png")
    return sorted([f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)])


def fusion(input_folder_ir, input_folder_vis, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tic = time.time()
    # criteria_fusion = Fusionloss()

    ir_images = get_image_files(input_folder_ir)
    vis_images = get_image_files(input_folder_vis)

    for ir_image, vis_image in zip(ir_images, vis_images):
        path1 = os.path.join(input_folder_ir, ir_image)
        path2 = os.path.join(input_folder_vis, vis_image)

        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)
        H, W, C = img2.shape
        img1, img2 = resize(img1, img2, [256, 256], [256, 256])  #
        img2Y = np.reshape(img2[:, :, 0:1], (H, W))


        img1 = np.asarray(img1, dtype=np.float32) / 255.0
        img2Y = np.asarray(img2Y, dtype=np.float32) / 255.0

        img1 = np.expand_dims(img1, axis=0)
        img2Y = np.expand_dims(img2Y, axis=0)

        img1_tensor = torch.from_numpy(img1).unsqueeze(0).to(device)
        img2Y_tensor = torch.from_numpy(img2Y).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            out = model(img1_tensor, img2Y_tensor)
            out = torch.clamp(out, 0, 1)
            fused_img = np.zeros([H,W,C])
            out_np = out.cpu().numpy()
            out_np = (out_np - np.min(out_np)) / (np.max(out_np) - np.min(out_np))

        d = np.squeeze(out_np)
        result = (d * 255).astype(np.uint8)
        result = np.reshape(result, (H, W, -1))
        fused_img[:, :, 0:1] = result
        fused_img[:, :, 1:2] = img2[:, :, 1:2]
        fused_img[:, :, 2] = img2[:, :, 2]
        result = cv2.cvtColor(fused_img.astype(np.uint8), cv2.COLOR_YUV2BGR)
        output_filename = os.path.splitext(ir_image)[0] + os.path.splitext(ir_image)[1]
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, result)

    toc = time.time()
    print('Processing time: {}'.format(toc - tic))


if __name__ == '__main__':
    input_folder_1 = '/media/dzs/系统/dzs/source_images/medical/1/'
    input_folder_2 = '/media/dzs/系统/dzs/source_images/medical/2/'
    output_folder = './outputs/ct_mr_2kd_10101/'
    # input_folder_1 = '/media/dzs/系统/dzs/imagefusion_mdlatlrr/IV_images/new_dataset/IR/'
    # input_folder_2 = '/media/dzs/系统/dzs/imagefusion_mdlatlrr/IV_images/new_dataset/VIS/'
    # output_folder = './outputs/ct_mr_2kd_vif/'
    fusion(input_folder_2, input_folder_1, output_folder)
