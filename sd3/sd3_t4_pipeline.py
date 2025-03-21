import gc, torch, random
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer, T5EncoderModel
from diffusers import StableDiffusion3Pipeline
from sd3.extended_embeddings_sd3 import get_weighted_text_embeddings_sd3


def clear_memory():
  gc.collect()
  torch.cuda.empty_cache()
	

def get_encoders(model_path = "stabilityai/stable-diffusion-3-medium-diffusers"):
  '''downloading all the encoders and tokenizer and loading them in gpu'''
  clear_memory()
  global tokenizer_1, text_encoder_1, tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3, modules_list_text_encoder_3

  tokenizer_1 = CLIPTokenizer.from_pretrained(
    model_path ,
    subfolder='tokenizer',
    device_map="auto",
  )

  text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
    model_path ,
    subfolder='text_encoder',
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant='fp16',
    device_map="auto",
  )

  tokenizer_2 = CLIPTokenizer.from_pretrained(
    model_path ,
    subfolder='tokenizer_2',
    device_map="auto",
  )

  text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    model_path ,
    subfolder='text_encoder_2',
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant='fp16',
    device_map="auto",
  )

  clear_memory()

  tokenizer_3 = AutoTokenizer.from_pretrained(
    model_path ,
    subfolder='tokenizer_3',
    device_map="auto",
  )

  text_encoder_3 = T5EncoderModel.from_pretrained(
      model_path,
      subfolder="text_encoder_3",
      torch_dtype=torch.float16,
      device_map="auto",
      variant="fp16"
  )

  clear_memory()
  modules_list_text_encoder_3=[] # create a list of modules to move them between cpu and gpu
  for modl in text_encoder_3.modules():
    modules_list_text_encoder_3.append(modl)
  modules_list_text_encoder_3 = modules_list_text_encoder_3[::-1]
  clear_memory()

def get_tranformer_vae(model_path = "stabilityai/stable-diffusion-3-medium-diffusers"):
  '''downloading the pipeline without encoders and tokenizers and loading them in cpu'''
  clear_memory()
  global pipe, modules_list_transformer
  pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        tokenizer=None,
        tokenizer_2=None,
        tokenizer_3=None,
      ).to('cpu')
  clear_memory()
  modules_list_transformer=[]
  for modl in pipe.transformer.modules():
    modules_list_transformer.append(modl)
  modules_list_transformer = modules_list_transformer[::-1]
  clear_memory()


def get_text_embeddings(prompt, neg_prompt):
  (
    prompt_embeds
    , prompt_neg_embeds
    , pooled_prompt_embeds
    , negative_pooled_prompt_embeds
  ) = get_weighted_text_embeddings_sd3(
      text_encoder_1,
      text_encoder_2,
      text_encoder_3,
      tokenizer_1,
      tokenizer_2,
      tokenizer_3
      , prompt = prompt
      , neg_prompt = neg_prompt
  )
  clear_memory()

  return prompt_embeds, prompt_neg_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def move_transformer_modules(device, last_layers):

  if last_layers=='all':
    pipe.to(device)
  else:
    a=0
    for i in modules_list_transformer:
      a=a+1
      if a<=last_layers:
        i.to(device)

  clear_memory()


def move_encoder3_modules(device, last_layers):

  a=0
  for i in modules_list_text_encoder_3:
    a=a+1
    if a<=last_layers:
      i.to(device)
  clear_memory()


def get_image(prompt_embeds, prompt_neg_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
              num_inference_steps=28, guidance_scale=7, num_images_per_prompt=1, generator=None,
              timesteps=None, latents=None, output_type='pil', return_dict=True, joint_attention_kwargs= None,
              clip_skip=None, callback_on_step_end=None, callback_on_step_end_tensor_inputs=['latents'], max_sequence_length=512):

  clear_memory()
  with torch.no_grad():
	  image = pipe(
	        prompt_embeds                 = prompt_embeds
	      , negative_prompt_embeds        = prompt_neg_embeds
	      , pooled_prompt_embeds          = pooled_prompt_embeds
	      , negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
	      , num_inference_steps           = num_inference_steps
	      , generator                     = generator
	      , guidance_scale                = guidance_scale
	      , max_sequence_length           = max_sequence_length
	      , timesteps=timesteps, latents=latents, output_type=output_type, return_dict=return_dict, joint_attention_kwargs=joint_attention_kwargs
	      , clip_skip=clip_skip, callback_on_step_end=callback_on_step_end, callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,num_images_per_prompt=num_images_per_prompt
	  ).images
  clear_memory()
  return image

def generate_sd3_t4_image(prompt, neg_prompt='', num_inference_steps=28, guidance_scale=7, num_images_per_prompt=1, generator=None,
         timesteps=None, latents=None, output_type='pil', return_dict=True, joint_attention_kwargs= None,
                          clip_skip=None, callback_on_step_end=None, callback_on_step_end_tensor_inputs=['latents'], max_sequence_length=512):
                          
  prompt_embeds, prompt_neg_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = get_text_embeddings(prompt, neg_prompt)
  move_transformer_modules(device='cuda', last_layers=400)
  move_encoder3_modules(device='cpu', last_layers=250)
  move_transformer_modules(device='cuda', last_layers='all')
  clear_memory()
  image = get_image(prompt_embeds, prompt_neg_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,
                 num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator, max_sequence_length= max_sequence_length,
                 timesteps=timesteps, latents=latents, output_type=output_type, return_dict=return_dict, joint_attention_kwargs=joint_attention_kwargs,
                 clip_skip=clip_skip, callback_on_step_end=callback_on_step_end, callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,num_images_per_prompt=num_images_per_prompt)
  clear_memory()
  move_encoder3_modules(device='cuda', last_layers=150)
  move_transformer_modules(device='cpu', last_layers='all')
  move_encoder3_modules(device='cuda', last_layers=250) 
  clear_memory()
  return image


def load_sd3_t4_pipeline(model_path = "stabilityai/stable-diffusion-3-medium-diffusers"):
  clear_memory()
  get_encoders(model_path)
  get_tranformer_vae(model_path)
  prompt_warmup = "gold ring with diamond with two interlocking bands made of polished yellow gold, each with a smooth, rounded profile that tapers slightly towards the back for comfort. a central floral cluster of diamonds, featuring a larger round diamond surrounded by smaller round diamonds arranged in a petal-like formation, creating a blooming effect. a halo of smaller diamonds encircling the central floral cluster, enhancing the overall sparkle and creating a sense of depth. the base of the ring features a smooth, polished finish that complements the intricate diamond settings, ensuring a seamless transition from the floral designs to the band."
  neg_prompt_warmup = ""
  generator = torch.Generator(device='cuda').manual_seed(random.randint(0,1e6))
  _ = generate_sd3_t4_image(prompt_warmup, neg_prompt_warmup, num_inference_steps=1, guidance_scale=7, generator=generator)
  
  
