from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from typing import List, Optional, Tuple, Union
import piq


totensor = T.ToTensor()
topil = T.ToPILImage()

def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)



def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def prepare_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image[0]
 

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


@torch.no_grad()
def evaluate_image(x,y):
    # Read RGB image and it's noisy version
    if torch.cuda.is_available():
        # Move to GPU to make computaions faster
        x = x.cuda()
        y = y.cuda()
   
    # To compute LPIPS as a loss function, use corresponding PyTorch module
    lpips_loss: torch.Tensor = piq.LPIPS(reduction='none')(x, y)
    #print(f"LPIPS: {lpips_loss.item():0.4f}")

    # To compute PSNR as a measure, use lower case function from the library.
    psnr_index = piq.psnr(x, y, data_range=1., reduction='none')
    #print(f"PSNR index: {psnr_index.item():0.4f}")

    # To compute SSIM index as a measure, use lower case function from the library:
    ssim_index = piq.ssim(x, y, data_range=1.)
    # In order to use SSIM as a loss function, use corresponding PyTorch module:
    ssim_loss: torch.Tensor = piq.SSIMLoss(data_range=1.)(x, y)
    #print(f"SSIM index: {ssim_index.item():0.4f}, loss: {ssim_loss.item():0.4f}")


    # To compute VIF as a measure, use lower case function from the library:
    vif_index: torch.Tensor = piq.vif_p(x, y, data_range=1.)
    # In order to use VIF as a loss function, use corresponding PyTorch class:
    vif_loss: torch.Tensor = piq.VIFLoss(sigma_n_sq=2.0, data_range=1.)(x, y)
    #print(f"VIFp index: {vif_index.item():0.4f}, loss: {vif_loss.item():0.4f}")

    # To compute FSIM as a measure, use lower case function from the library
    fsim_index: torch.Tensor = piq.fsim(x, y, data_range=1., reduction='none')
    # In order to use FSIM as a loss function, use corresponding PyTorch module
    fsim_loss = piq.FSIMLoss(data_range=1., reduction='none')(x, y)
    #print(f"FSIM index: {fsim_index.item():0.4f}, loss: {fsim_loss.item():0.4f}")

    return lpips_loss, psnr_index, ssim_index, ssim_loss, vif_index, vif_loss, fsim_index, fsim_loss




