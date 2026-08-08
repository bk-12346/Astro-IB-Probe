from huggingface_hub import list_repo_files

files = list_repo_files("smith42/astropt_v2.0")
sizes = sorted(set(f.split("/")[1] for f in files if f.startswith("astropt/") and "/" in f[len("astropt/"):]))
print("Available AstroPT v2.0 sizes:")
for s in sizes:
    print(" -", s)


# ***** OUTPUT *****
# Available AstroPT v2.0 sizes:
#  - 015M
#  - 095M
#  - 850M