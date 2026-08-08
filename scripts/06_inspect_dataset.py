from datasets import load_dataset

ds = load_dataset("Smith42/galaxies", split="validation", streaming=True)
example = next(iter(ds))
print("Available fields:")
for k, v in example.items():
    print(f"  {k}: {type(v)}")

# ***** OUTPUT *****

# Available fields:
#   image: <class 'PIL.JpegImagePlugin.JpegImageFile'>
#   image_crop: <class 'PIL.PngImagePlugin.PngImageFile'>
#   dr8_id: <class 'str'>
#   galaxy_size: <class 'int'>
