from huggingface_hub import list_repo_files

files = list_repo_files("smith42/astroPT")
for f in files:
    print(f)

# ***** OUTPUT *****
# .gitattributes
# README.md
# embeddings/300M_spiralized/idss_64t.npy
# embeddings/300M_spiralized/xss_64t.npy
# embeddings/300M_spiralized/zss_64t.npy
# models/fully_trained/0001M_params/005000_ckpt.pt
# models/fully_trained/0001M_params/010000_ckpt.pt
# models/fully_trained/0001M_params/015000_ckpt.pt
# models/fully_trained/0001M_params/020000_ckpt.pt
# models/fully_trained/0001M_params/025000_ckpt.pt
# models/fully_trained/0001M_params/030000_ckpt.pt
# models/fully_trained/0005M_params/005000_ckpt.pt
# models/fully_trained/0005M_params/010000_ckpt.pt
# models/fully_trained/0005M_params/015000_ckpt.pt
# models/fully_trained/0005M_params/020000_ckpt.pt
# models/fully_trained/0005M_params/025000_ckpt.pt
# models/fully_trained/0005M_params/030000_ckpt.pt
# models/fully_trained/0012M_params/005000_ckpt.pt
# models/fully_trained/0012M_params/010000_ckpt.pt
# models/fully_trained/0012M_params/015000_ckpt.pt
# models/fully_trained/0012M_params/020000_ckpt.pt
# models/fully_trained/0012M_params/025000_ckpt.pt
# models/fully_trained/0012M_params/030000_ckpt.pt
# models/fully_trained/0021M_params/005000_ckpt.pt
# models/fully_trained/0021M_params/010000_ckpt.pt
# models/fully_trained/0021M_params/015000_ckpt.pt
# models/fully_trained/0021M_params/020000_ckpt.pt
# models/fully_trained/0021M_params/025000_ckpt.pt
# models/fully_trained/0021M_params/030000_ckpt.pt
# models/fully_trained/0089M_params/005000_ckpt.pt
# models/fully_trained/0089M_params/010000_ckpt.pt
# models/fully_trained/0089M_params/015000_ckpt.pt
# models/fully_trained/0089M_params/020000_ckpt.pt
# models/fully_trained/0089M_params/025000_ckpt.pt
# models/fully_trained/0089M_params/030000_ckpt.pt
# models/fully_trained/0309M_params/005000_ckpt.pt
# models/fully_trained/0309M_params/010000_ckpt.pt
# models/fully_trained/0309M_params/015000_ckpt.pt
# models/fully_trained/0309M_params/020000_ckpt.pt
# models/fully_trained/0309M_params/025000_ckpt.pt
# models/fully_trained/0309M_params/030000_ckpt.pt
# models/fully_trained/0830M_params/005000_ckpt.pt
# models/fully_trained/0830M_params/010000_ckpt.pt
# models/fully_trained/0830M_params/015000_ckpt.pt
# models/fully_trained/0830M_params/020000_ckpt.pt
# models/fully_trained/0830M_params/025000_ckpt.pt
# models/fully_trained/0830M_params/030000_ckpt.pt
# models/fully_trained/2100M_params/040000_ckpt.pt
# models/trained_to_1M_gals/01Mgal_001Mparam_best.pt
# models/trained_to_1M_gals/01Mgal_010Mparam_best.pt
# models/trained_to_1M_gals/01Mgal_300Mparam_best.pt
# models/trained_to_1M_gals/01Mgal_700Mparam_best.pt
# models/trained_to_1M_gals/spiral_300M.pt
