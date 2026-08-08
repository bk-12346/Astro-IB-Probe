from astropt.model_utils import load_astropt

model = load_astropt(
    repo_id="smith42/astropt_v2.0",
    path="astropt/095M",  # placeholder — we'll confirm the real available sizes next
    weights_filename="ckpt.pt",
)
model = model.to("cpu")
model.eval()

print(model)
print(sum(p.numel() for p in model.parameters()), "parameters")


# ***** OUTPUT *****
# model loaded successfully from smith42/astropt_v2.0/astropt/095M
# args: {'n_layer': 12, 'n_head': 12, 'n_embd': 768, 'n_chan': 3, 'block_size': 1024, 'dropout': 0.0, 'modalities': [ModalityConfig(name='images', input_size=768, pos_input_size=1, patch_size=16, embed_pos=True, loss_weight=1.0)], 'attn_type': 'causal'}
# GPT(
#   (transformer): ModuleDict(
#     (wte): ModuleDict(
#       (images): Encoder(
#         (c_fc): Linear(in_features=768, out_features=3072, bias=False)
#         (c_proj): Linear(in_features=3072, out_features=768, bias=False)
#       )
#     )
#     (wpe): ModuleDict(
#       (images): Embedder(
#         (wpe): Embedding(1024, 768)
#       )
#     )
#     (drop): Dropout(p=0.0, inplace=False)
#     (h): ModuleList(
#       (0-11): 12 x Block(
#         (ln_1): LayerNorm()
#         (attn): SelfAttention(
#           (c_attn): Linear(in_features=768, out_features=2304, bias=False)
#           (c_proj): Linear(in_features=768, out_features=768, bias=False)
#           (attn_dropout): Dropout(p=0.0, inplace=False)
#           (resid_dropout): Dropout(p=0.0, inplace=False)
#         )
#         (ln_2): LayerNorm()
#         (mlp): MLP(
#           (c_fc): Linear(in_features=768, out_features=3072, bias=False)
#           (c_proj): Linear(in_features=3072, out_features=768, bias=False)
#           (dropout): Dropout(p=0.0, inplace=False)
#         )
#       )
#     )
#     (ln_f): LayerNorm()
#   )
#   (lm_head): ModuleDict(
#     (images): Decoder(
#       (c_fc): Linear(in_features=768, out_features=3072, bias=False)
#       (c_proj): Linear(in_features=3072, out_features=768, bias=False)
#     )
#   )
# )
# 95177472 parameters
