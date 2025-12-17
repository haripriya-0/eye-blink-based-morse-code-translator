import os
import cv2
import numpy as np

IMG_SIZE = 64  # Width and height
DATASET_PATH = 'dataset'
CATEGORIES = ['Closed_Eyes', 'Open_Eyes']

def load_data(split):
    data = []
    labels = []
    for category in CATEGORIES:
        label = CATEGORIES.index(category)  # 0 for closed, 1 for open
        folder_path = os.path.join(DATASET_PATH, split, category)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img.flatten())
            labels.append(label)
    return np.array(data), np.array(labels)

# Load train and test data
X_train, y_train = load_data('train')
X_test, y_test = load_data('test')

# Save to .npy files
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
print("Dataset saved as .npy files.")
