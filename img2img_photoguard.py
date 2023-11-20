
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
from pipeline_stable_diffusion_img2img import MyStableDiffusionImg2ImgPipeline
from typing import Union, List, Optional, Callable
from utils import preprocess, prepare_mask_and_masked_image, recover_image, prepare_image, randn_tensor, evaluate_image
import pandas as pd
from absl import flags, app
import random 
import torchvision.transforms as T
from typing import Any, Callable, Dict, List, Optional, Union

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
    image = None,
    prompt: Union[str, List[str]] = None,
    strength: float = 1,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: Optional[float] = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
):

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(prompt, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self.device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )
    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Preprocess image
    image = self.image_processor.preprocess(image)

    # 5. set timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

    # 6. Prepare latent variables
    latents = self.prepare_latents(
        image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, generator
    )

    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 8. Denoising loop
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]


    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)
    return image[0]

    
def compute_grad(pipe_img2img, image, prompt, target_image, **kwargs):
    torch.set_grad_enabled(True)
    image = image.clone()
    image.requires_grad_()
    image_nat = attack_forward(pipe_img2img,
                               image,
                               prompt=prompt,
                               **kwargs)
    
    loss = (image_nat - target_image).norm(p=2)
    #print("image_nat:{} target:{}".format(image_nat.shape,target_image.shape))
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
    #print("eps:{} step_size:{}".format(eps,step_size))
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
    pipe_img2img = MyStableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16,revision="fp16")
    pipe_img2img = pipe_img2img.to(device)

    RANDOM_SEED_LIST = [0,1,2,3,4,5,6,7,8,9] 
    target_url = "https://i.pinimg.com/originals/18/37/aa/1837aa6f2c357badf0f588916f3980bd.png"
    response = requests.get(target_url)
    target_image = Image.open(BytesIO(response.content)).convert("RGB")
    target_image = target_image.resize((512, 512))
    target_image_tensor = prepare_image(target_image)
    target_image_tensor = 0*target_image_tensor.to(device) # we can either attack towards a target image or simply the zero tensor

    ORIGINAL_EDIT_IMAGE_FOLDER = FLAGS.out_dir + '/ORIGINAL_EDIT_IMAGE'
    ADVERSARIAL_IMAGE_FOLDER = FLAGS.out_dir + '/ADVERSARIAL_IMAGE'
    ADVERSARIAL_EDIT_IMAGE_FOLDER = FLAGS.out_dir + '/ADVERSARIAL_EDIT_IMAGE'
    os.makedirs(ORIGINAL_EDIT_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(ADVERSARIAL_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(ADVERSARIAL_EDIT_IMAGE_FOLDER, exist_ok=True)

    lpips_list, psnr_list, ssim_index_list, ssim_loss_list, vif_index_list, vif_loss_list, fsim_index_list, fsim_loss_list, image_list = [],[],[],[],[],[],[],[],[]
    parsed_input_image_sequence = FLAGS.input_img.split('~')
    start_img_num = int(parsed_input_image_sequence[0])
    end_img_num = int(parsed_input_image_sequence[1])


    for i in range(start_img_num,end_img_num+1):
        total_lpips_loss, total_psnr_index, total_ssim_index, total_vif_index, total_fsim_index, total_fsim_loss, total_ssim_loss, total_vif_loss = 0,0,0,0,0,0,0,0
        image_path = 'original_images/dog_{}.png'.format(i)
        image_list.append(image_path)
        init_image = Image.open(image_path).convert('RGB').resize((512,512))
        image = prepare_image(init_image)
        image = image.half().to(device)
        print("---------------------------------- CURRENT PROCESSING IMAGE: {} ----------------------------------".format(image_path))

        for k in range(FLAGS.num_test):
            path_template = '{}/dog_{}_seed_{}.png'
            SEED = RANDOM_SEED_LIST[k] #786349
            set_seed(SEED)
            prompt = ""
            guidance_scale = 7.5
            num_inference_steps = FLAGS.num_inference_step


            if FLAGS.attack_type == 'l2':
                result, last_image= super_l2(pipe_img2img,
                                image,
                                prompt=prompt,
                                target_image=target_image_tensor,
                                eps=16,
                                step_size=1,
                                iters=200,
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
                                eps=0.06,
                                step_size=0.01,
                                iters=100,
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

            lpips_loss, psnr_index, ssim_index, ssim_loss, vif_index, vif_loss, fsim_index, fsim_loss = evaluate_image(image_nat.unsqueeze(0),image_adv.unsqueeze(0))
            total_lpips_loss += lpips_loss.cpu().item()
            total_psnr_index += psnr_index.cpu().item()
            total_ssim_index += ssim_index.cpu().item()
            total_ssim_loss  += ssim_loss.cpu().item()
            total_vif_index  += vif_index.cpu().item()
            total_vif_loss   += vif_loss.cpu().item()
            total_fsim_index += fsim_index.cpu().item()
            total_fsim_loss  += fsim_loss.cpu().item()


        lpips_list.append(round(total_lpips_loss/FLAGS.num_test,6))
        psnr_list.append(round(total_psnr_index/FLAGS.num_test,6))
        ssim_index_list.append(round(total_ssim_index/FLAGS.num_test,6))
        ssim_loss_list.append(round(total_ssim_loss/FLAGS.num_test,6))
        vif_index_list.append(round(total_vif_index/FLAGS.num_test,6))
        vif_loss_list.append(round(total_vif_loss/FLAGS.num_test,6))
        fsim_index_list.append(round(total_fsim_index/FLAGS.num_test,6))
        fsim_loss_list.append(round(total_fsim_loss/FLAGS.num_test,6))


    
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




