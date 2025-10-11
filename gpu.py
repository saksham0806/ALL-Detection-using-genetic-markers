import tensorflow as tf

# List all GPUs visible to TF
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available to TensorFlow:", gpus)

# Detailed info (name, memory, etc.)
from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
for d in devices:
    if d.device_type == "GPU":
        print(f"TensorFlow is using: {d.physical_device_desc}")
