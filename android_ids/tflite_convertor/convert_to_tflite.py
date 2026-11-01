import tensorflow as tf

from tfltransfer import bases, heads, optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

# ---- IDS adaptation (the only changed file on the Python side, besides server settings) ----
# Same structure as the original Flower example, only the head model is replaced by the
# autoencoder of src/task.py (class Autoencoder) and the loss is MSE instead of cross-entropy.
INPUT_DIM = 11  # = INPUT_DIM in dataset_load_tpot.py, = TransferLearningModelWrapper.INPUT_DIM

# Base model: identity layer, passes the input unchanged to the head (as in the original).
base = tf.keras.Sequential(
    [tf.keras.Input(shape=(INPUT_DIM,)), tf.keras.layers.Lambda(lambda x: x)]
)
base.compile(loss="mse", optimizer="sgd")
base.save("identity_model", save_format="tf")

# Head model = autoencoder (11 -> 48 -> 24 -> 6 -> 24 -> 48 -> 11).
# Differences to the PyTorch model: no BatchNorm / Dropout (their non-trainable state does not
# fit the TFLite on-device training API); AdamW -> plain SGD.
head = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(INPUT_DIM,)),
        tf.keras.layers.Dense(48, activation="relu"),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(6),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(48, activation="relu"),
        tf.keras.layers.Dense(INPUT_DIM),
    ]
)
head.compile(loss="mse", optimizer="sgd")

# The "number of classes" argument is the size of the target vector. For an autoencoder
# target = input, so it is INPUT_DIM. SGD (the Java client only supports stateless optimizers; Adam needs optimizer state that the example's LiteOptimizerModel.performStep does not pass) with lr = 1e-2 and batch size 32 (smallest device has only 48 training rows).
# This generates a directory tflite_model with five tflite models.
# Copy them to client/app/src/main/assets/model.
base_path = bases.saved_model_base.SavedModelBase("identity_model")
converter = TFLiteTransferConverter(
    INPUT_DIM, base_path, heads.KerasModelHead(head), optimizers.SGD(1e-2), train_batch_size=32
)
converter.convert_and_save("tflite_model")
