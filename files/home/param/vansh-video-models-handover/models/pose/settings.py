from easydict import EasyDict

# Wan2.2 5B model architecture
wan_5b = EasyDict()
wan_5b.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
wan_5b.t5_tokenizer = 'google/umt5-xxl'
wan_5b.text_len = 512
wan_5b.vae_checkpoint = 'Wan2.2_VAE.pth'
wan_5b.patch_size = (1, 2, 2)
wan_5b.in_dim = 48
wan_5b.dim = 3072
wan_5b.ffn_dim = 14336
wan_5b.freq_dim = 256
wan_5b.text_dim = 4096
wan_5b.out_dim = 48
wan_5b.num_heads = 24
wan_5b.num_layers = 30
wan_5b.window_size = (-1, -1)
wan_5b.qk_norm = True
wan_5b.cross_attn_norm = True
wan_5b.eps = 1e-6

# Width-expanded wan2.2
wan_5b_wide = EasyDict(dict(wan_5b))
wan_5b_wide.dim = 3840
wan_5b_wide.ffn_dim = 17920
wan_5b_wide.num_heads = 30

# Wan2.2 VAE compression + clip constants
FPS = 25.0
VAE_TEMPORAL_STRIDE = 4
VAE_S_SCALE = 16
NUM_LATENT_FRAMES = 20
