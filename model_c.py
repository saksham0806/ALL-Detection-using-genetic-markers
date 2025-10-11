# Leukemia Subtype Detection using Blood Smear Images
# Complete Pipeline: Load, Preprocess, Train, and Test

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import cv2
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, ResNet50, DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ============================================================================
# 2. CONFIGURATION
# ============================================================================

class Config:
    # Paths - MODIFY THESE ACCORDING TO YOUR DATASET STRUCTURE
    BASE_DIR = '/kaggle/input/leukemia-dataset'  # Change this to your dataset path
    TRAIN_DIR = '/kaggle/input/leukemia-dataset/train'
    TEST_DIR = '/kaggle/input/leukemia-dataset/test'
    
    # Image parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # Classes
    CLASSES = ['ALL', 'AML', 'CLL', 'CML', 'Healthy']
    NUM_CLASSES = len(CLASSES)
    
    # Validation split (only for train data)
    VALIDATION_SPLIT = 0.2

config = Config()

# ============================================================================
# 3. DATA LOADING AND EXPLORATION
# ============================================================================

def load_image_paths(base_dir, subset_name):
    """Load all image paths and their labels from a specific directory"""
    image_paths = []
    labels = []
    
    for class_name in config.CLASSES:
        class_dir = os.path.join(base_dir, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(class_name)
    
    return np.array(image_paths), np.array(labels)

# Load training data
print("Loading training image paths...")
train_image_paths, train_labels = load_image_paths(config.TRAIN_DIR, 'train')

# Load test data
print("Loading test image paths...")
test_image_paths, test_labels = load_image_paths(config.TEST_DIR, 'test')

# Create dataframes
train_full_df = pd.DataFrame({'image_path': train_image_paths, 'label': train_labels})
test_df = pd.DataFrame({'image_path': test_image_paths, 'label': test_labels})

print(f"\nTotal training images: {len(train_full_df)}")
print(f"Total test images: {len(test_df)}")

print("\nTraining set class distribution:")
print(train_full_df['label'].value_counts())

print("\nTest set class distribution:")
print(test_df['label'].value_counts())

# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Training set
train_full_df['label'].value_counts().plot(kind='bar', color='skyblue', ax=axes[0])
axes[0].set_title('Training Set - Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Test set
test_df['label'].value_counts().plot(kind='bar', color='lightcoral', ax=axes[1])
axes[1].set_title('Test Set - Class Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ============================================================================
# 4. IMAGE PREPROCESSING FUNCTIONS
# ============================================================================

def normalize_stain(img):
    """
    Normalize color/stain variations in blood smear images
    Uses histogram equalization and color normalization
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels
    lab = cv2.merge([l, a, b])
    
    # Convert back to RGB
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return normalized

def remove_background(img):
    """Remove background and enhance cells"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Otsu's thresholding
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    result = cv2.bitwise_and(img, mask_3channel)
    
    return result

def preprocess_image(img_path, apply_stain_norm=True, apply_bg_removal=False):
    """Complete preprocessing pipeline"""
    # Read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
    
    # Apply stain normalization
    if apply_stain_norm:
        img = normalize_stain(img)
    
    # Apply background removal (optional, may not be needed for all cases)
    if apply_bg_removal:
        img = remove_background(img)
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

# Visualize preprocessing effects
def visualize_preprocessing(img_paths, n_samples=3):
    """Visualize original vs preprocessed images"""
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    
    for i in range(n_samples):
        # Original
        img_original = cv2.imread(img_paths[i])
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_original = cv2.resize(img_original, (config.IMG_WIDTH, config.IMG_HEIGHT))
        
        # With stain normalization
        img_norm = preprocess_image(img_paths[i], apply_stain_norm=True, apply_bg_removal=False)
        
        # With stain norm + background removal
        img_full = preprocess_image(img_paths[i], apply_stain_norm=True, apply_bg_removal=True)
        
        axes[i, 0].imshow(img_original)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img_norm)
        axes[i, 1].set_title('Stain Normalized')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(img_full)
        axes[i, 2].set_title('Stain Norm + BG Removal')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Sample images from each class for visualization
sample_indices = []
for class_name in config.CLASSES:
    class_indices = train_full_df[train_full_df['label'] == class_name].index
    if len(class_indices) > 0:
        sample_indices.append(class_indices[0])

if len(sample_indices) >= 3:
    visualize_preprocessing(train_full_df.loc[sample_indices[:3], 'image_path'].values)

# ============================================================================
# 5. DATA GENERATORS
# ============================================================================

def create_data_generators():
    """Create train, validation, and test data generators with augmentation"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        preprocessing_function=lambda x: x  # Already preprocessed
    )
    
    # No augmentation for validation and test
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x
    )
    
    return train_datagen, val_test_datagen

# Split data
from sklearn.model_selection import train_test_split

# Split training data into train and validation
train_df, val_df = train_test_split(
    train_full_df, 
    test_size=config.VALIDATION_SPLIT, 
    stratify=train_full_df['label'], 
    random_state=42
)

# Test data is already separate
print(f"\nTrain samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

print("\n" + "="*70)
print("Train set distribution:")
print(train_df['label'].value_counts())
print("\nValidation set distribution:")
print(val_df['label'].value_counts())
print("\nTest set distribution:")
print(test_df['label'].value_counts())
print("="*70)

# Custom data generator
class LeukemiaDataGenerator(keras.utils.Sequence):
    def __init__(self, dataframe, batch_size, img_size, n_classes, 
                 augment=False, shuffle=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_classes = n_classes
        self.augment = augment
        self.shuffle = shuffle
        self.label_to_index = {label: idx for idx, label in enumerate(config.CLASSES)}
        
        if self.augment:
            self.datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )
        
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = self.dataframe.iloc[indices]
        
        X, y = self.__data_generation(batch_data)
        return X, y
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __data_generation(self, batch_data):
        X = np.empty((len(batch_data), *self.img_size, 3))
        y = np.empty((len(batch_data), self.n_classes))
        
        for i, (_, row) in enumerate(batch_data.iterrows()):
            # Load and preprocess image
            img = preprocess_image(row['image_path'], apply_stain_norm=True)
            
            # Apply augmentation if needed
            if self.augment:
                img = self.datagen.random_transform(img)
            
            X[i,] = img
            
            # One-hot encode label
            label_idx = self.label_to_index[row['label']]
            y[i,] = keras.utils.to_categorical(label_idx, self.n_classes)
        
        return X, y

# Create generators
train_generator = LeukemiaDataGenerator(
    train_df, 
    batch_size=config.BATCH_SIZE,
    img_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
    n_classes=config.NUM_CLASSES,
    augment=True,
    shuffle=True
)

val_generator = LeukemiaDataGenerator(
    val_df,
    batch_size=config.BATCH_SIZE,
    img_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
    n_classes=config.NUM_CLASSES,
    augment=False,
    shuffle=False
)

test_generator = LeukemiaDataGenerator(
    test_df,
    batch_size=config.BATCH_SIZE,
    img_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
    n_classes=config.NUM_CLASSES,
    augment=False,
    shuffle=False
)

# ============================================================================
# 6. MODEL ARCHITECTURE
# ============================================================================

def build_model(model_name='EfficientNetB0'):
    """
    Build transfer learning model
    Options: 'EfficientNetB0', 'ResNet50', 'DenseNet121'
    """
    
    # Input layer
    inputs = layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS))
    
    # Base model
    if model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
    elif model_name == 'ResNet50':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
    elif model_name == 'DenseNet121':
        base_model = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model, base_model

# Build model
print("\nBuilding model...")
model, base_model = build_model('EfficientNetB0')
model.summary()

# ============================================================================
# 7. COMPILE MODEL
# ============================================================================

# Compute class weights for imbalanced dataset
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)
class_weight_dict = dict(enumerate(class_weights))
print("\nClass weights:", class_weight_dict)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

# ============================================================================
# 8. CALLBACKS
# ============================================================================

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_leukemia_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ============================================================================
# 9. TRAINING
# ============================================================================

print("\n" + "="*70)
print("STARTING TRAINING - PHASE 1: Feature Extraction")
print("="*70)

# Phase 1: Train with frozen base model
history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Phase 2: Fine-tuning - Unfreeze some layers of base model
print("\n" + "="*70)
print("STARTING TRAINING - PHASE 2: Fine-tuning")
print("="*70)

# Unfreeze the last few layers
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30  # Unfreeze last 30 layers

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE/10),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

# Continue training
history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=config.EPOCHS,
    initial_epoch=len(history_phase1.history['loss']),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Combine histories
history = {
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
}

# ============================================================================
# 10. VISUALIZATION OF TRAINING
# ============================================================================

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# ============================================================================
# 11. MODEL EVALUATION
# ============================================================================

print("\n" + "="*70)
print("EVALUATING MODEL ON TEST SET")
print("="*70)

# Evaluate on test set
test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(
    test_generator,
    verbose=1
)

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Get predictions
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(np.vstack([y for _, y in test_generator]), axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=config.CLASSES))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=config.CLASSES, 
            yticklabels=config.CLASSES,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# Normalized confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=config.CLASSES,
            yticklabels=config.CLASSES,
            cbar_kws={'label': 'Percentage'})
plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================================
# 12. PREDICTION ON NEW IMAGES
# ============================================================================

def predict_single_image(img_path, model):
    """Predict leukemia subtype for a single image"""
    # Preprocess image
    img = preprocess_image(img_path, apply_stain_norm=True)
    img = np.expand_dims(img, axis=0)
    
    # Predict
    predictions = model.predict(img, verbose=0)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = config.CLASSES[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    return predicted_class, confidence, predictions

def visualize_predictions(img_paths, true_labels, model, n_images=9):
    """Visualize predictions for multiple images"""
    n_cols = 3
    n_rows = int(np.ceil(n_images / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i, (img_path, true_label) in enumerate(zip(img_paths[:n_images], true_labels[:n_images])):
        # Load original image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get prediction
        pred_class, confidence, all_probs = predict_single_image(img_path, model)
        
        # Display
        axes[i].imshow(img)
        color = 'green' if pred_class == true_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_class} ({confidence:.2%})',
                         fontsize=10, color=color, fontweight='bold')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Sample predictions from test set
sample_test_data = test_df.sample(min(9, len(test_df)))
sample_test_paths = sample_test_data['image_path'].values
sample_test_labels = sample_test_data['label'].values
visualize_predictions(sample_test_paths, sample_test_labels, model, n_images=9)

# ============================================================================
# 13. SAVE MODEL
# ============================================================================

# Save the model
model.save('leukemia_classifier_final.h5')
print("\nModel saved as 'leukemia_classifier_final.h5'")

# Save model in TensorFlow SavedModel format
model.save('leukemia_classifier_savedmodel', save_format='tf')
print("Model saved in SavedModel format")

print("\n" + "="*70)
print("TRAINING AND EVALUATION COMPLETE!")
print("="*70)