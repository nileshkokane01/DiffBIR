Global seed set to 231
using device cuda
Args for ControlLDM :   configs/model/cldm.yaml
controlLDM
ddpm init : 
ControlLDM: Running in eps-prediction mode
diffusionwrapper:  print:  diff_model_config:  {'target': 'model.cldm.ControlledUnetModel', 'params': {'use_checkpoint': True, 'image_size': 32, 'in_channels': 4, 'out_channels': 4, 'model_channels': 320, 'attention_resolutions': [4, 2, 1], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4], 'num_head_channels': 64, 'use_spatial_transformer': True, 'use_linear_in_transformer': True, 'transformer_depth': 1, 'context_dim': 1024, 'legacy': False}}
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 1024 and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 1024 and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 1024 and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 1024 and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 1024 and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 1024 and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 1024 and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 1024 and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 1024 and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 1024 and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 1024 and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 1024 and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 1024 and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 1024 and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 1024 and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 1024 and using 5 heads.
DiffusionWrapper has 865.91 M params.
making attention of type 'vanilla-xformers' with 512 in_channels
building MemoryEfficientAttnBlock with 512 in_channels...
Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
making attention of type 'vanilla-xformers' with 512 in_channels
building MemoryEfficientAttnBlock with 512 in_channels...
open_clip_pytorch_model.bin: 100% 3.94G/3.94G [00:13<00:00, 287MB/s]
control net
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 1024 and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 1024 and using 5 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 1024 and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 1024 and using 10 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 1024 and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 1024 and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 20 heads.
Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 1024 and using 20 heads.
/usr/local/lib/python3.10/dist-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3190.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth" to /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
100% 233M/233M [00:00<00:00, 288MB/s]
Loading model from: /usr/local/lib/python3.10/dist-packages/lpips/weights/v0.1/alex.pth
reload swinir model from weights/general_swinir_v1.ckpt
spacedSampler :  make_schedule
timesteps used in spaced sampler: 
	[0, 20, 41, 61, 82, 102, 122, 143, 163, 183, 204, 224, 245, 265, 285, 306, 326, 347, 367, 387, 408, 428, 449, 469, 489, 510, 530, 550, 571, 591, 612, 632, 652, 673, 693, 714, 734, 754, 775, 795, 816, 836, 856, 877, 897, 917, 938, 958, 979, 999]
Spaced Sampler:   0% 0/50 [00:00<?, ?it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   2% 1/50 [00:00<00:28,  1.73it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   4% 2/50 [00:00<00:18,  2.57it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   6% 3/50 [00:01<00:15,  3.04it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   8% 4/50 [00:01<00:13,  3.32it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  10% 5/50 [00:01<00:12,  3.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  12% 6/50 [00:01<00:12,  3.62it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  14% 7/50 [00:02<00:11,  3.70it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  16% 8/50 [00:02<00:11,  3.76it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  18% 9/50 [00:02<00:10,  3.79it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  20% 10/50 [00:02<00:10,  3.82it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  22% 11/50 [00:03<00:10,  3.84it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  24% 12/50 [00:03<00:09,  3.85it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  26% 13/50 [00:03<00:09,  3.86it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  28% 14/50 [00:03<00:09,  3.87it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  30% 15/50 [00:04<00:09,  3.87it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  32% 16/50 [00:04<00:08,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  34% 17/50 [00:04<00:08,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  36% 18/50 [00:04<00:08,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  38% 19/50 [00:05<00:07,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  40% 20/50 [00:05<00:07,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  42% 21/50 [00:05<00:07,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  44% 22/50 [00:05<00:07,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  46% 23/50 [00:06<00:06,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  48% 24/50 [00:06<00:06,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  50% 25/50 [00:06<00:06,  3.84it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  52% 26/50 [00:07<00:06,  3.85it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  54% 27/50 [00:07<00:05,  3.86it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  56% 28/50 [00:07<00:05,  3.87it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  58% 29/50 [00:07<00:05,  3.87it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  60% 30/50 [00:08<00:05,  3.87it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  62% 31/50 [00:08<00:04,  3.87it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  64% 32/50 [00:08<00:04,  3.87it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  66% 33/50 [00:08<00:04,  3.87it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  68% 34/50 [00:09<00:04,  3.87it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  70% 35/50 [00:09<00:03,  3.87it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  72% 36/50 [00:09<00:03,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  74% 37/50 [00:09<00:03,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  76% 38/50 [00:10<00:03,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  78% 39/50 [00:10<00:02,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  80% 40/50 [00:10<00:02,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  82% 41/50 [00:10<00:02,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  84% 42/50 [00:11<00:02,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  86% 43/50 [00:11<00:01,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  88% 44/50 [00:11<00:01,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  90% 45/50 [00:11<00:01,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  92% 46/50 [00:12<00:01,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  94% 47/50 [00:12<00:00,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  96% 48/50 [00:12<00:00,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  98% 49/50 [00:12<00:00,  3.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler: 100% 50/50 [00:13<00:00,  3.78it/s]
save to results/demo/general/comic3_0.png
spacedSampler :  make_schedule
timesteps used in spaced sampler: 
	[0, 20, 41, 61, 82, 102, 122, 143, 163, 183, 204, 224, 245, 265, 285, 306, 326, 347, 367, 387, 408, 428, 449, 469, 489, 510, 530, 550, 571, 591, 612, 632, 652, 673, 693, 714, 734, 754, 775, 795, 816, 836, 856, 877, 897, 917, 938, 958, 979, 999]
Spaced Sampler:   0% 0/50 [00:00<?, ?it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   2% 1/50 [00:00<00:15,  3.25it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   4% 2/50 [00:00<00:09,  4.87it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   6% 3/50 [00:00<00:08,  5.80it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   8% 4/50 [00:00<00:07,  6.37it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  10% 5/50 [00:00<00:06,  6.73it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  12% 6/50 [00:00<00:06,  6.97it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  14% 7/50 [00:01<00:06,  7.14it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  16% 8/50 [00:01<00:05,  7.25it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  18% 9/50 [00:01<00:05,  7.32it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  20% 10/50 [00:01<00:05,  7.38it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  22% 11/50 [00:01<00:05,  7.41it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  24% 12/50 [00:01<00:05,  7.44it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  26% 13/50 [00:01<00:04,  7.46it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  28% 14/50 [00:02<00:04,  7.47it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  30% 15/50 [00:02<00:04,  7.47it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  32% 16/50 [00:02<00:04,  7.48it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  34% 17/50 [00:02<00:04,  7.48it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  36% 18/50 [00:02<00:04,  7.49it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  38% 19/50 [00:02<00:04,  7.49it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  40% 20/50 [00:02<00:04,  7.49it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  42% 21/50 [00:02<00:03,  7.49it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  44% 22/50 [00:03<00:03,  7.49it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  46% 23/50 [00:03<00:03,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  48% 24/50 [00:03<00:03,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  50% 25/50 [00:03<00:03,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  52% 26/50 [00:03<00:03,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  54% 27/50 [00:03<00:03,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  56% 28/50 [00:03<00:02,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  58% 29/50 [00:04<00:02,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  60% 30/50 [00:04<00:02,  7.49it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  62% 31/50 [00:04<00:02,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  64% 32/50 [00:04<00:02,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  66% 33/50 [00:04<00:02,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  68% 34/50 [00:04<00:02,  7.49it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  70% 35/50 [00:04<00:02,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  72% 36/50 [00:04<00:01,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  74% 37/50 [00:05<00:01,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  76% 38/50 [00:05<00:01,  7.49it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  78% 39/50 [00:05<00:01,  7.48it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  80% 40/50 [00:05<00:01,  7.48it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  82% 41/50 [00:05<00:01,  7.48it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  84% 42/50 [00:05<00:01,  7.49it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  86% 43/50 [00:05<00:00,  7.49it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  88% 44/50 [00:06<00:00,  7.49it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  90% 45/50 [00:06<00:00,  7.49it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  92% 46/50 [00:06<00:00,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  94% 47/50 [00:06<00:00,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  96% 48/50 [00:06<00:00,  7.50it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  98% 49/50 [00:06<00:00,  7.51it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler: 100% 50/50 [00:06<00:00,  7.30it/s]
save to results/demo/general/29_0.png
spacedSampler :  make_schedule
timesteps used in spaced sampler: 
	[0, 20, 41, 61, 82, 102, 122, 143, 163, 183, 204, 224, 245, 265, 285, 306, 326, 347, 367, 387, 408, 428, 449, 469, 489, 510, 530, 550, 571, 591, 612, 632, 652, 673, 693, 714, 734, 754, 775, 795, 816, 836, 856, 877, 897, 917, 938, 958, 979, 999]
Spaced Sampler:   0% 0/50 [00:00<?, ?it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   2% 1/50 [00:00<00:25,  1.91it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   4% 2/50 [00:00<00:16,  2.82it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   6% 3/50 [00:00<00:14,  3.33it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   8% 4/50 [00:01<00:12,  3.64it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  10% 5/50 [00:01<00:11,  3.83it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  12% 6/50 [00:01<00:11,  3.96it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  14% 7/50 [00:01<00:10,  4.05it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  16% 8/50 [00:02<00:10,  4.11it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  18% 9/50 [00:02<00:09,  4.15it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  20% 10/50 [00:02<00:09,  4.18it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  22% 11/50 [00:02<00:09,  4.19it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  24% 12/50 [00:03<00:09,  4.21it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  26% 13/50 [00:03<00:08,  4.22it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  28% 14/50 [00:03<00:08,  4.22it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  30% 15/50 [00:03<00:08,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  32% 16/50 [00:04<00:08,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  34% 17/50 [00:04<00:07,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  36% 18/50 [00:04<00:07,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  38% 19/50 [00:04<00:07,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  40% 20/50 [00:05<00:07,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  42% 21/50 [00:05<00:06,  4.24it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  44% 22/50 [00:05<00:06,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  46% 23/50 [00:05<00:06,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  48% 24/50 [00:05<00:06,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  50% 25/50 [00:06<00:05,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  52% 26/50 [00:06<00:05,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  54% 27/50 [00:06<00:05,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  56% 28/50 [00:06<00:05,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  58% 29/50 [00:07<00:04,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  60% 30/50 [00:07<00:04,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  62% 31/50 [00:07<00:04,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  64% 32/50 [00:07<00:04,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  66% 33/50 [00:08<00:04,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  68% 34/50 [00:08<00:03,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  70% 35/50 [00:08<00:03,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  72% 36/50 [00:08<00:03,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  74% 37/50 [00:09<00:03,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  76% 38/50 [00:09<00:02,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  78% 39/50 [00:09<00:02,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  80% 40/50 [00:09<00:02,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  82% 41/50 [00:09<00:02,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  84% 42/50 [00:10<00:01,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  86% 43/50 [00:10<00:01,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  88% 44/50 [00:10<00:01,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  90% 45/50 [00:10<00:01,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  92% 46/50 [00:11<00:00,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  94% 47/50 [00:11<00:00,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  96% 48/50 [00:11<00:00,  4.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  98% 49/50 [00:11<00:00,  4.24it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler: 100% 50/50 [00:12<00:00,  4.13it/s]
save to results/demo/general/14_0.png
spacedSampler :  make_schedule
timesteps used in spaced sampler: 
	[0, 20, 41, 61, 82, 102, 122, 143, 163, 183, 204, 224, 245, 265, 285, 306, 326, 347, 367, 387, 408, 428, 449, 469, 489, 510, 530, 550, 571, 591, 612, 632, 652, 673, 693, 714, 734, 754, 775, 795, 816, 836, 856, 877, 897, 917, 938, 958, 979, 999]
Spaced Sampler:   0% 0/50 [00:00<?, ?it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   2% 1/50 [00:00<00:11,  4.41it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   6% 3/50 [00:00<00:06,  7.67it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  10% 5/50 [00:00<00:05,  8.85it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  14% 7/50 [00:00<00:04,  9.43it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  18% 9/50 [00:00<00:04,  9.76it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  22% 11/50 [00:01<00:03,  9.97it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  26% 13/50 [00:01<00:03, 10.10it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  30% 15/50 [00:01<00:03, 10.18it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  34% 17/50 [00:01<00:03, 10.24it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  38% 19/50 [00:01<00:03, 10.27it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  42% 21/50 [00:02<00:02, 10.30it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  46% 23/50 [00:02<00:02, 10.32it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  50% 25/50 [00:02<00:02, 10.34it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  54% 27/50 [00:02<00:02, 10.35it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  58% 29/50 [00:02<00:02, 10.36it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  62% 31/50 [00:03<00:01, 10.36it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  66% 33/50 [00:03<00:01, 10.36it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  70% 35/50 [00:03<00:01, 10.36it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  74% 37/50 [00:03<00:01, 10.36it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  78% 39/50 [00:03<00:01, 10.35it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  82% 41/50 [00:04<00:00, 10.36it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  86% 43/50 [00:04<00:00, 10.34it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  90% 45/50 [00:04<00:00, 10.35it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  94% 47/50 [00:04<00:00, 10.35it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  98% 49/50 [00:04<00:00, 10.22it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler: 100% 50/50 [00:04<00:00, 10.07it/s]
save to results/demo/general/53_0.png
spacedSampler :  make_schedule
timesteps used in spaced sampler: 
	[0, 20, 41, 61, 82, 102, 122, 143, 163, 183, 204, 224, 245, 265, 285, 306, 326, 347, 367, 387, 408, 428, 449, 469, 489, 510, 530, 550, 571, 591, 612, 632, 652, 673, 693, 714, 734, 754, 775, 795, 816, 836, 856, 877, 897, 917, 938, 958, 979, 999]
Spaced Sampler:   0% 0/50 [00:00<?, ?it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   2% 1/50 [00:00<00:10,  4.48it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:   6% 3/50 [00:00<00:05,  7.92it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  10% 5/50 [00:00<00:04,  9.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  14% 7/50 [00:00<00:04,  9.88it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  18% 9/50 [00:00<00:04, 10.23it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  22% 11/50 [00:01<00:03, 10.44it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  26% 13/50 [00:01<00:03, 10.51it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  30% 15/50 [00:01<00:03, 10.54it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  34% 17/50 [00:01<00:03, 10.65it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  38% 19/50 [00:01<00:02, 10.68it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  42% 21/50 [00:02<00:02, 10.65it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  46% 23/50 [00:02<00:02, 10.56it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  50% 25/50 [00:02<00:02, 10.38it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  54% 27/50 [00:02<00:02, 10.46it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  58% 29/50 [00:02<00:01, 10.60it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  62% 31/50 [00:03<00:01, 10.69it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  66% 33/50 [00:03<00:01, 10.72it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  70% 35/50 [00:03<00:01, 10.77it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  74% 37/50 [00:03<00:01, 10.61it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  78% 39/50 [00:03<00:01, 10.69it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  82% 41/50 [00:03<00:00, 10.78it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  86% 43/50 [00:04<00:00, 10.84it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  90% 45/50 [00:04<00:00, 10.85it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  94% 47/50 [00:04<00:00, 10.83it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler:  98% 49/50 [00:04<00:00, 10.81it/s]spacesampler  :  p_sample
_extract_into_tensor_
spaced_smapler: predict_noise
controlLDM :  apply Model 
ControlNet :  forward 
_extract_into_tensor_
_extract_into_tensor_
q_posterior_mean_variance
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
_extract_into_tensor_
Spaced Sampler: 100% 50/50 [00:04<00:00, 1