
import os
from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import requests
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import torchvision.transforms as T
from typing import Union, List, Optional, Callable
from notebooks.utils import preprocess, prepare_mask_and_masked_image, recover_image, prepare_image, randn_tensor, evaluate_image
import pandas as pd
from absl import flags, app
import random 
to_pil = T.ToPILImage()

FLAGS = flags.FLAGS
flags.DEFINE_string('out_dir', "output", "path of output folder")
flags.DEFINE_string('attack_type', "linf", "type of adversarial attack chosen from ['l2',linf']")
flags.DEFINE_string('input_img', "1~50", "input image sequence from [a to b]")
flags.DEFINE_integer('num_inference_step', 10, "num_inference_step in adversarial attack")
flags.DEFINE_integer('GPU_ID', 0, "rank of GPU used")
flags.DEFINE_integer('num_test', 10, "number of seeds used to evaluate per image attack")



# A differentiable version of the forward function of the inpainting stable diffusion model! See https://github.com/huggingface/diffusers
def attack_forward(
        self,
        image,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
    ):

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        uncond_tokens = [""]
        max_length = text_input_ids.shape[-1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        seq_len = uncond_embeddings.shape[1]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        text_embeddings = text_embeddings.detach()

        image_latents = self.vae.encode(image.unsqueeze(0)).latent_dist.sample()
        image_latents = 0.18215 * image_latents
        ############## TODO: Adding noise to latent image #################
        shape = image_latents.shape
        batch_size = 1
        num_images_per_prompt = 1
        strength = 0.7

        noise = randn_tensor(shape).to(self.device).half()
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, self.device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        image_latents = self.scheduler.add_noise(image_latents, noise, latent_timestep)
        latents = torch.cat([image_latents] * 2)

        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        for i, t in enumerate(timesteps_tensor):

            ########## TODO: check whether *2 is required ##################
            latent_model_input = latents#torch.cat([latents] * 2) 
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            #### TODO: ETA ??? ###
            #latents = self.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            #######################
            #count += 1
        #print('count:',count)

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        return image

    
def compute_grad(pipe_img2img, image, prompt, target_image, **kwargs):
    torch.set_grad_enabled(True)
    image = image.clone()
    image.requires_grad_()
    image_nat = attack_forward(pipe_img2img,
                               image,
                               prompt=prompt,
                               **kwargs)
    
    loss = (image_nat - target_image).norm(p=2)
    grad = torch.autograd.grad(loss, [image])[0] 
        
    return grad, loss.item(), image_nat.data.cpu()

def super_l2(pipe_img2img, X, prompt, step_size, iters, eps, clamp_min, clamp_max, grad_reps = 5, target_image = 0, **kwargs):
    X_adv = X.clone()
    last_image = None
    iterator = tqdm(range(iters))
    for i in iterator:

        all_grads = []
        losses = []
        for i in range(grad_reps):
            c_grad, loss, last_image = compute_grad(pipe_img2img, X_adv, prompt, target_image=target_image, **kwargs)
            all_grads.append(c_grad)
            losses.append(loss)
        grad = torch.stack(all_grads).mean(0)
        
        iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')

        l = len(X.shape) - 1
        grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
        grad_normalized = grad.detach() / (grad_norm + 1e-10)

        # actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        actual_step_size = step_size
        X_adv = X_adv - grad_normalized * actual_step_size

        d_x = X_adv - X.detach()
        d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
        X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)        
    
    torch.cuda.empty_cache()

    return X_adv, last_image

def super_linf(pipe_img2img, X, prompt, step_size, iters, eps, clamp_min, clamp_max, grad_reps = 5, target_image = 0, **kwargs):
    X_adv = X.clone()
    last_image = None
    iterator = tqdm(range(iters))
    for i in iterator:
        all_grads = []
        losses = []
        for i in range(grad_reps):
            c_grad, loss, last_image = compute_grad(pipe_img2img, X_adv, prompt, target_image=target_image, **kwargs)
            all_grads.append(c_grad)
            losses.append(loss)
        grad = torch.stack(all_grads).mean(0)
        
        iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')
        
        # actual_step_size = step_size - (step_size - step_size / 100) / iters * i
        actual_step_size = step_size
        X_adv = X_adv - grad.detach().sign() * actual_step_size

        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        
    torch.cuda.empty_cache()

    return X_adv, last_image





def perform_attack():
    # make sure you're logged in with `huggingface-cli login` - check https://github.com/huggingface/diffusers for more details
    device = torch.device("cuda:{}".format(FLAGS.GPU_ID))
    model_id_or_path = "CompVis/stable-diffusion-v1-4"
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16,revision="fp16")
    pipe_img2img = pipe_img2img.to(device)

    RANDOM_SEED_LIST = [0,1,2,3,4,5,6,7,8,9] 
    target_url = "https://i.pinimg.com/originals/18/37/aa/1837aa6f2c357badf0f588916f3980bd.png"
    response = requests.get(target_url)
    target_image = Image.open(BytesIO(response.content)).convert("RGB")
    target_image = target_image.resize((512, 512))
    target_image_tensor = prepare_image(target_image)
    target_image_tensor = 0*target_image_tensor.to(device) # we can either attack towards a target image or simply the zero tensor

    ORIGINAL_IMAGE_FOLDER = 'original_images'
    ORIGINAL_EDIT_IMAGE_FOLDER = FLAGS.out_dir + '/ORIGINAL_EDIT_IMAGE'
    ADVERSARIAL_IMAGE_FOLDER = FLAGS.out_dir + '/ADVERSARIAL_IMAGE'
    ADVERSARIAL_EDIT_IMAGE_FOLDER = FLAGS.out_dir + '/ADVERSARIAL_EDIT_IMAGE'
    os.makedirs(ORIGINAL_EDIT_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(ADVERSARIAL_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(ADVERSARIAL_EDIT_IMAGE_FOLDER, exist_ok=True)

    lpips_list, psnr_list, ssim_list, vif_list, fsim_list, image_list = [],[],[],[],[],[]
    parsed_input_image_sequence = FLAGS.input_img.split('~')
    start_img_num = int(parsed_input_image_sequence[0])
    end_img_num = int(parsed_input_image_sequence[1])


    for i in range(start_img_num,end_img_num+1):
        total_lpips_loss, total_psnr_index, total_ssim_index, total_vif_index, total_fsim_index = 0,0,0,0,0
        image_path = 'original_images/dog_{}.png'
        image_list.append(image_path.format(i))
        init_image = Image.open(image_path.format(i)).convert('RGB').resize((512,512))
        image = prepare_image(init_image)
        image = image.half().to(device)

        for k in range(FLAGS.num_test):
            path_template = '{}/dog_{}_seed_{}.png'
            SEED = RANDOM_SEED_LIST[k] #786349
            set_seed(SEED)
            prompt = ""
            strength = 0.7
            guidance_scale = 7.5
            num_inference_steps = FLAGS.num_inference_step


            if FLAGS.attack_type == 'l2':
                result, last_image= super_l2(pipe_img2img,
                                image,
                                prompt=prompt,
                                target_image=target_image_tensor,
                                eps=16,
                                step_size=1,
                                iters=100,
                                clamp_min = -1,
                                clamp_max = 1,
                                eta=1,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                grad_reps=10
                                )
            
            elif FLAGS.attack_type == 'linf':
                result, last_image= super_linf(pipe_img2img,
                                image,
                                prompt=prompt,
                                target_image=target_image_tensor,
                                eps=0.1,
                                step_size=0.006,
                                iters=4,
                                clamp_min = -1,
                                clamp_max = 1,
                                eta=1,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                grad_reps=10
                                )

            adv_X = (result / 2 + 0.5).clamp(0, 1) 
            adv_image = to_pil(adv_X).convert("RGB")
            adv_image.save(path_template.format(ADVERSARIAL_IMAGE_FOLDER,i,k))
            adv_image = T.ToTensor()(adv_image)

            SEED = RANDOM_SEED_LIST[k] #9209
            set_seed(SEED)
            prompt = 'a photo of a cat'
            strength = 0.7
            guidance_scale = 7.5
            num_inference_steps = 100

            image_nat = pipe_img2img(prompt=prompt, 
                                image=init_image, 
                                eta=1,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                strength=strength
                                ).images[0]
            image_nat.save(path_template.format(ORIGINAL_EDIT_IMAGE_FOLDER,i,k))
            image_nat = T.ToTensor()(image_nat)

            set_seed(SEED)
            image_adv = pipe_img2img(prompt=prompt, 
                                image=adv_image, 
                                eta=1,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                strength=strength
                                ).images[0]
            image_adv.save(path_template.format(ADVERSARIAL_EDIT_IMAGE_FOLDER,i,k))
            image_adv = T.ToTensor()(image_adv)

            lpips_loss, psnr_index, ssim_index, vif_index, fsim_index = evaluate_image(image_nat.unsqueeze(0),image_adv.unsqueeze(0))
            total_lpips_loss += lpips_loss.cpu().item()
            total_psnr_index += psnr_index.cpu().item()
            total_ssim_index += ssim_index.cpu().item()
            total_vif_index  += vif_index.cpu().item()
            total_fsim_index += fsim_index.cpu().item()


        lpips_list.append(round(total_lpips_loss/FLAGS.num_test,6))
        psnr_list.append(round(total_psnr_index/FLAGS.num_test,6))
        ssim_list.append(round(total_ssim_index/FLAGS.num_test,6))
        vif_list.append(round(total_vif_index/FLAGS.num_test,6))
        fsim_list.append(round(total_fsim_index/FLAGS.num_test,6))
    

    
    df_evaluation_metrics = pd.DataFrame(
    {'IMAGE_PATH':image_list,
     'LPIPS': lpips_list,
     'PSNR': psnr_list,
     'SSIM': ssim_list,
     'VIF': vif_list,
     'FSIM': fsim_list
    })        

    df_evaluation_metrics.to_csv(FLAGS.out_dir + '/eval_step_{}_sequences_{}.csv'.format(FLAGS.num_inference_step,FLAGS.input_img),index=False)

        

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(argv):
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    perform_attack()
    


if __name__ == '__main__':
    app.run(main)




