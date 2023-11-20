import torch
import piq
from skimage.io import imread
import os 
import pandas as pd 

@torch.no_grad()
def main():
    # Read RGB image and it's noisy version
    x = torch.tensor(imread('output_STEP_02/ADVERSARIAL_EDIT_IMAGE/dog_25_seed_0.png')).permute(2, 0, 1)[None, ...] / 255.
    y = torch.tensor(imread('output_STEP_02/ORIGINAL_EDIT_IMAGE/dog_25_seed_0.png')).permute(2, 0, 1)[None, ...] / 255.

    print("x:{} y:{}".format(x.shape,y.shape))
    if torch.cuda.is_available():
        # Move to GPU to make computaions faster
        x = x.cuda()
        y = y.cuda()
   
    # To compute LPIPS as a loss function, use corresponding PyTorch module
    lpips_loss: torch.Tensor = piq.LPIPS(reduction='none')(x, y)
    print(f"LPIPS: {lpips_loss.item():0.4f}")

    # To compute PSNR as a measure, use lower case function from the library.
    psnr_index = piq.psnr(x, y, data_range=1., reduction='none')
    print(f"PSNR index: {psnr_index.item():0.4f}")

    # To compute SSIM index as a measure, use lower case function from the library:
    ssim_index = piq.ssim(x, y, data_range=1.)
    # In order to use SSIM as a loss function, use corresponding PyTorch module:
    ssim_loss: torch.Tensor = piq.SSIMLoss(data_range=1.)(x, y)
    print(f"SSIM index: {ssim_index.item():0.4f}, loss: {ssim_loss.item():0.4f}")


    # To compute VIF as a measure, use lower case function from the library:
    vif_index: torch.Tensor = piq.vif_p(x, y, data_range=1.)
    # In order to use VIF as a loss function, use corresponding PyTorch class:
    vif_loss: torch.Tensor = piq.VIFLoss(sigma_n_sq=2.0, data_range=1.)(x, y)
    print(f"VIFp index: {vif_index.item():0.4f}, loss: {vif_loss.item():0.4f}")

    # To compute FSIM as a measure, use lower case function from the library
    fsim_index: torch.Tensor = piq.fsim(x, y, data_range=1., reduction='none')
    # In order to use FSIM as a loss function, use corresponding PyTorch module
    fsim_loss = piq.FSIMLoss(data_range=1., reduction='none')(x, y)
    print(f"FSIM index: {fsim_index.item():0.4f}, loss: {fsim_loss.item():0.4f}")

   


def test():
    # Read RGB image and it's noisy version
    x_path = 'output_STEP_10/ADVERSARIAL_EDIT_IMAGE/dog_25_seed_{}.png'
    y_path = 'output_STEP_10/ORIGINAL_EDIT_IMAGE/dog_25_seed_{}.png'
    #total_lpips_loss, total_psnr_index, total_ssim_index, total_vif_index, total_fsim_index = 0,0,0,0,0
    total_lpips_loss, total_psnr_index, total_ssim_index, total_vif_index, total_fsim_index, total_fsim_loss, total_ssim_loss, total_vif_loss = 0,0,0,0,0,0,0,0

    for i in range(5):
        x = torch.tensor(imread(x_path.format(i))).permute(2, 0, 1)[None, ...] / 255.
        y = torch.tensor(imread(y_path.format(i))).permute(2, 0, 1)[None, ...] / 255.

        #print("x:{} y:{}".format(x.shape,y.shape))
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

        total_lpips_loss += lpips_loss.cpu().item()
        total_psnr_index += psnr_index.cpu().item()
        total_ssim_index += ssim_index.cpu().item()
        total_ssim_loss  += ssim_loss.cpu().item()
        total_vif_index  += vif_index.cpu().item()
        total_vif_loss   += vif_loss.cpu().item()
        total_fsim_index += fsim_index.cpu().item()
        total_fsim_loss  += fsim_loss.cpu().item()

    
    total_lpips_loss /= 10
    total_psnr_index /= 10
    total_ssim_index /= 10
    total_ssim_loss  /= 10
    total_vif_index  /= 10
    total_vif_loss   /= 10
    total_fsim_index /= 10
    total_fsim_loss  /= 10

    print("lpips_loss:{} psnr:{} ssim_index:{} ssim_loss:{} vif_index:{} vif_loss:{} fsim_index:{} fsim_loss:{}".format(total_lpips_loss,total_psnr_index,total_ssim_index,total_ssim_loss,total_vif_index,total_vif_loss,total_fsim_index,total_fsim_loss))


def test_folder():
    # Read RGB image and it's noisy version

    NUM_INFERENCE_STEP = 10
    folder_path = 'output_STEP_{}/ADVERSARIAL_EDIT_IMAGE/'.format(str(NUM_INFERENCE_STEP).zfill(2))
    edit_path = 'ORIGINAL_EDIT_IMAGE'
    #total_lpips_loss, total_psnr_index, total_ssim_index, total_vif_index, total_fsim_index = 0,0,0,0,0
    total_lpips_loss, total_psnr_index, total_ssim_index, total_vif_index, total_fsim_index, total_fsim_loss, total_ssim_loss, total_vif_loss = 0,0,0,0,0,0,0,0
    lpips_list, psnr_list, ssim_index_list, ssim_loss_list, vif_index_list, vif_loss_list, fsim_index_list, fsim_loss_list, image_list = [],[],[],[],[],[],[],[],[]

    device = torch.device('cuda:0')
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            x_path = folder_path + file
            y_path = x_path.replace('ADVERSARIAL_EDIT_IMAGE',edit_path)
            print("x_path:{} y_path:{}".format(x_path,y_path))

            x = torch.tensor(imread(x_path)).permute(2, 0, 1)[None, ...] / 255.
            y = torch.tensor(imread(y_path)).permute(2, 0, 1)[None, ...] / 255.

            #print("x:{} y:{}".format(x.shape,y.shape))
            if torch.cuda.is_available():
                # Move to GPU to make computaions faster
                x = x.to(device)
                y = y.to(device)
        
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

            total_lpips_loss += lpips_loss.cpu().item()
            total_psnr_index += psnr_index.cpu().item()
            total_ssim_index += ssim_index.cpu().item()
            total_ssim_loss  += ssim_loss.cpu().item()
            total_vif_index  += vif_index.cpu().item()
            total_vif_loss   += vif_loss.cpu().item()
            total_fsim_index += fsim_index.cpu().item()
            total_fsim_loss  += fsim_loss.cpu().item()

            # lpips_list, psnr_list, ssim_index_list, ssim_loss_list, vif_index_list, vif_loss_list, fsim_index_list, fsim_loss_list, image_list = [],[],[],[],[],[],[],[],[]

            lpips_list.append(lpips_loss.cpu().item())
            psnr_list.append(psnr_index.cpu().item())
            ssim_index_list.append(ssim_index.cpu().item())
            ssim_loss_list.append(ssim_loss.cpu().item())
            vif_index_list.append(vif_index.cpu().item())
            vif_loss_list.append(vif_loss.cpu().item())
            fsim_index_list.append(fsim_index.cpu().item())
            fsim_loss_list.append(fsim_loss.cpu().item())
            image_list.append(file)

    
    total_lpips_loss /= 10
    total_psnr_index /= 10
    total_ssim_index /= 10
    total_ssim_loss  /= 10
    total_vif_index  /= 10
    total_vif_loss   /= 10
    total_fsim_index /= 10
    total_fsim_loss  /= 10
    
    df_evaluation_metrics = pd.DataFrame(
    {'IMAGE_PATH':image_list,
     'LPIPS': lpips_list,
     'PSNR': psnr_list,
     'SSIM_INDEX': ssim_index_list,
     'SSIM_LOSS': ssim_loss_list,
     'VIF_INDEX': vif_index_list,
     'VIF_LOSS': vif_loss_list,
     'FSIM_INDEX': fsim_index_list,
     'FSIM_LOSS': fsim_loss_list,
    })        

    df_evaluation_metrics.to_csv('eval_{}.csv'.format(NUM_INFERENCE_STEP),index=False)


    print("lpips_loss:{} psnr:{} ssim_index:{} ssim_loss:{} vif_index:{} vif_loss:{} fsim_index:{} fsim_loss:{}".format(total_lpips_loss,total_psnr_index,total_ssim_index,total_ssim_loss,total_vif_index,total_vif_loss,total_fsim_index,total_fsim_loss))



if __name__ == '__main__':
    test_folder()