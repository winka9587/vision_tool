# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def img_to_arr( img ):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

# folder_or_list为输入的图像序列
# size为缩放后允许的最长边的大小 
# square_ok为是否允许图像不是正方形, square_ok=True, 则图像是正方形; square_ok=False, 则图像是4:3的长宽比
# verbose为是否输出信息
def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)  # size是长边允许的最大尺寸
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8  # 确保是16的倍数
            if not (square_ok) and W == H:
                halfh = 3*halfw/4  # 如果不要求正方形(square_ok=False), 则长宽比为4:3
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs


def load_masks(folder_or_list, size, square_ok=False, verbose=True):
    """ Open and convert all masks in a list or folder to proper input format (single channel). """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading masks from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} masks')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.png', '.jpg', '.jpeg']  # mask常用格式
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    masks = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        # Load the mask as a single channel grayscale image
        mask = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('L')
        W1, H1 = mask.size
        
        if size == 224:
            # resize short side to 224 (then crop)
            mask = _resize_pil_image(mask, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 'size'
            mask = _resize_pil_image(mask, size)
            
        W, H = mask.size
        cx, cy = W // 2, H // 2
        
        if size == 224:
            half = min(cx, cy)
            mask = mask.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8  # ensure size is a multiple of 16
            if not (square_ok) and W == H:
                halfh = 3 * halfw // 4  # adjust aspect ratio to 4:3 if not square
            mask = mask.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = mask.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        
        # Convert the mask to a tensor and store it
        mask_tensor = tvf.ToTensor()(mask)[None]  # Convert to tensor and add batch dimension
        masks.append(dict(mask=mask_tensor, true_shape=np.int32(
            [mask.size[::-1]]), idx=len(masks), instance=str(len(masks))))

    assert masks, 'no masks found at ' + root
    if verbose:
        print(f' (Found {len(masks)} masks)')
    return masks


def resize_image(img, size, square_ok=True, verbose=True):
    W_origin, H_origin = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W_origin/H_origin, H_origin/W_origin)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)  # size是长边允许的最大尺寸
    W_, H_ = img.size
    cx, cy = W_//2, H_//2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8  # 确保是16的倍数
        if not (square_ok) and W_ == H_:
            halfh = 3*halfw/4  # 如果不要求正方形(square_ok=False), 则长宽比为4:3
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    if verbose:
        W2, H2 = img.size
        print(f' - resize image resolution {W_origin}x{H_origin} --> {W2}x{H2}')
    return img

# convert torch.tensor to PIL.Image
# why use permute? because the tensor is in BCHW format, the PIL.Image is in HWC format
def tensor2PIL(img, channel="BCHW", data_range="0+1"):
    if channel == "BCHW":
        assert img.ndim == 4 and img.shape[0] == 1, f"img.ndim={img.ndim}, img.shape[0]={img.shape[0]}, (BCHW) B must be 1"
        img = img.squeeze(0).permute(1, 2, 0)
    if channel == "CHW":
        assert img.ndim == 3
        img = img.permute(1, 2, 0)
    
    if data_range == "-1+1":
        assert img.min() >= -1 and img.max() <= 1
        img = (img + 1) / 2
    else:
        assert data_range == "0+1"
        assert img.min() >= 0 and img.max() <= 1
        
    # img = img.numpy()
    # img_ret = PIL.Image.fromarray((img * 255).astype(np.uint8))
    img = (img * 255).clamp(0, 255).byte().numpy()
    img_ret = PIL.Image.fromarray(img)

    return img_ret  # 不能添加B====================================================================================================================================================

# img: (C, H, W)
def convertImgToShow(img, channel="BCHW", data_range="0+1"):
    if channel == "BCHW":
        assert img.ndim == 4 and img.shape[0] == 1
        img = img.squeeze(0).permute(1, 2, 0)
    if channel == "CHW":
        assert img.ndim == 3
        img = img.permute(1, 2, 0)
    
    if data_range == "-1+1":
        assert img.min() >= -1 and img.max() <= 1
        img = (img + 1) / 2
    else:
        assert data_range == "0+1"
        assert img.min() >= 0 and img.max() <= 1
        
    # img = img.numpy()
    # img_ret = PIL.Image.fromarray((img * 255).astype(np.uint8))
    img_ret = (img * 255).clamp(0, 255).byte()
    return img_ret


# def PIL2tensor(img, data_range="0+1"):
#     img_tensor=tvf.ToTensor()(img)[None]
#     if data_range == "-1+1":
#         assert img_tensor.min() >= 0 and img_tensor.max() <= 1
#         img_tensor = img_tensor * 2.0 - 1.0
#     return img_tensor

# import torchvision.transforms.functional as tvf

def PIL2tensor(img, data_range="0+1", device='cpu'):
    img_tensor=tvf.ToTensor()(img)[None]
    if data_range == "-1+1":
        assert img_tensor.min() >= 0 and img_tensor.max() <= 1
        img_tensor = img_tensor * 2.0 - 1.0
    return img_tensor.to(device, non_blocking=True)