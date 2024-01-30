
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
from diffusers import StableDiffusionXLImg2ImgPipeline
from typing import Union, List, Optional, Callable
from utils import preprocess, prepare_mask_and_masked_image, recover_image, prepare_image, randn_tensor, evaluate_image
import pandas as pd
from absl import flags, app
import random 
import torchvision.transforms as T
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers import PNDMScheduler, DDIMScheduler
to_pil = T.ToPILImage()
FLAGS = flags.FLAGS
flags.DEFINE_string('out_dir', "output", "path of output folder")
flags.DEFINE_string('attack_type', "linf", "type of adversarial attack chosen from ['l2',linf']")
flags.DEFINE_string('input_img', "1~50", "input image sequence from [a to b]")
flags.DEFINE_integer('num_inference_step', 10, "num_inference_step in adversarial attack")
flags.DEFINE_integer('GPU_ID', 0, "rank of GPU used")
flags.DEFINE_integer('num_test', 10, "number of seeds used to evaluate per image attack")

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

# A differentiable version of the forward function of the inpainting stable diffusion model! See https://github.com/huggingface/diffusers
def attack_forward(
    self,
    image = None,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    strength: float = 0.3,
    num_inference_steps: int = 50,
    denoising_start: Optional[float] = None,
    denoising_end: Optional[float] = None,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    original_size: Tuple[int, int] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Tuple[int, int] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    aesthetic_score: float = 6.0,
    negative_aesthetic_score: float = 2.5,
):
    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        strength,
        num_inference_steps,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. Preprocess image
    image = self.image_processor.preprocess(image)

    # 5. Prepare timesteps
    def denoising_value_valid(dnv):
        return isinstance(denoising_end, float) and 0 < dnv < 1

    self.scheduler.set_timesteps(num_inference_steps, device=device)
    #print("num inference steps:{}".format(num_inference_steps))
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device, denoising_start=denoising_start if denoising_value_valid else None)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

    add_noise = True if denoising_start is None else False
    # 6. Prepare latent variables
    latents = self.prepare_latents(
        image,
        latent_timestep,
        batch_size,
        num_images_per_prompt,
        prompt_embeds.dtype,
        device,
        generator,
        add_noise,
    )
    # 7. Prepare extra step kwargs.
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    height, width = latents.shape[-2:]
    height = height * self.vae_scale_factor
    width = width * self.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 8. Prepare added time ids & embeddings
    if negative_original_size is None:
        negative_original_size = original_size
    if negative_target_size is None:
        negative_target_size = target_size

    add_text_embeds = pooled_prompt_embeds
    add_time_ids, add_neg_time_ids = self._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype=prompt_embeds.dtype,
    )
    add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device)

    # 9. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    #print("inference steps:{} timesteps:{}".format(num_inference_steps, timesteps))
    #with self.progress_bar(total=num_inference_steps) as progress_bar:
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if do_classifier_free_guidance and guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        #print("t:{} latents:{} noise_pred:{}".format(t.dtype,latents.dtype,noise_pred.dtype))

    
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
    #print("loss:{} image:{} image_nat:{} target_image:{}".format(loss.dtype,image.dtype,image_nat.dtype,target_image.dtype))
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
    #print("iters:",iters)
    iters = 5
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
    model_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
    pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.half)
    pipe_img2img.vae.config.force_upcast = False
    #pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.half)
    pipe_img2img.scheduler = PNDMScheduler.from_config(pipe_img2img.scheduler.config)
    print("scheduler:",pipe_img2img.scheduler)
    pipe_img2img = pipe_img2img.to(device)

    RANDOM_SEED_LIST = [0,1,2,3,4,5,6,7,8,9] 
    target_url = "https://i.pinimg.com/originals/18/37/aa/1837aa6f2c357badf0f588916f3980bd.png"
    response = requests.get(target_url)
    target_image = Image.open(BytesIO(response.content)).convert("RGB")
    target_image = target_image.resize((512, 512))
    target_image_tensor = prepare_image(target_image)
    target_image_tensor = 0*target_image_tensor.to(device) # we can either attack towards a target image or simply the zero tensor
    target_image_tensor = target_image_tensor.half().to(device)

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




