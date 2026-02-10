import os
from PIL import Image

def resize_images(input_dir, output_dir, size=(32, 32)):
    """
    Resize all images in the input directory to the specified size and save them
    in the output directory, maintaining the folder structure.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory to save resized images.
        size (tuple): Target size for resizing (width, height).
    """
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding directories in the output folder
        relative_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(root, file)
                output_path = os.path.join(target_dir, f"{os.path.splitext(file)[0]}_32x32{os.path.splitext(file)[1]}")

                try:
                    with Image.open(input_path) as img:
                        img_resized = img.resize(size, Image.Resampling.LANCZOS)
                        img_resized.save(output_path)
                        print(f"Resized and saved: {output_path}")
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")

if __name__ == "__main__":
    input_directory = "architecture_dataset"
    output_directory = "architecture_dataset_32x32"

    resize_images(input_directory, output_directory)