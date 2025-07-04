import os
import pandas as pd

def count_images_per_class(data_dir, subsets=('train', 'val', 'test'), extensions={'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}):
    data = {}

    for subset in subsets:
        subset_path = os.path.join(data_dir, subset)
        if not os.path.exists(subset_path):
            continue
        for class_name in os.listdir(subset_path):
            class_path = os.path.join(subset_path, class_name)
            if not os.path.isdir(class_path):
                continue
            count = sum(
                1 for f in os.listdir(class_path)
                if os.path.splitext(f)[1].lower() in extensions
            )
            if class_name not in data:
                data[class_name] = {}
            data[class_name][subset] = count

    df = pd.DataFrame.from_dict(data, orient='index').fillna(0).astype(int)

    # Ensure consistent column order
    for subset in subsets:
        if subset not in df.columns:
            df[subset] = 0
    df = df[list(subsets)]


    # Add a Total row
    total_row = df.sum().to_frame().T
    total_row.index = ['Total']
    df = pd.concat([df, total_row])

    return df

# Example usage
dataset_path = r"/home/endodl/PHASE-1/mln/anatomical/anatomical_stomach/anat_yolo_split/data"
df = count_images_per_class(dataset_path)
df.to_csv("image_counts.csv")

print(df)

