import json
import numpy as np
import tensorflow as tf
import pickle

from keras.callbacks import ModelCheckpoint

from img_process_server_connect import ImageProcessServerConnect
from tensorflow.keras import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D, GRU, TimeDistributed, Input, Concatenate

from collect_data import IMAGE_SIZE, DATA_FOLDER
from prep_data import SEQUENCE_SIZE


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

class CustomBatchLoadGenerator(Sequence):

    def __init__(self, image_paths: list, seq_metadata: np.ndarray, y: np.ndarray, batch_size : int, img_server : ImageProcessServerConnect):
        self.image_paths = image_paths
        self.seq_metadata = seq_metadata
        self.y = y
        self.batch_size = batch_size
        self.img_server = img_server

    # Loads all images in batch dynamically from disk, to avoid taking up to much RAM.
    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_paths_x = self.image_paths[start_index:end_index]
        batch_metadata_x = self.seq_metadata[start_index:end_index]
        batch_y = self.y[start_index:end_index]

        all_image_paths = [path for sequence_paths in batch_paths_x for path in sequence_paths]
        all_images = self.img_server.ask_for_images(all_image_paths, IMAGE_SIZE, IMAGE_SIZE)
        x = []
        for i in range(0, len(all_images), SEQUENCE_SIZE):
            sequence_images = all_images[i:i+SEQUENCE_SIZE]
            sequence_x = np.array(sequence_images) / 255.0
            x.append(sequence_x)

        x = np.array(x)
        y = np.array(batch_y)
        # Shuffle for good measure
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        return [x[indices], batch_metadata_x[indices]], y[indices]

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))


def get_model() -> Model:
    metadata_input = Input(shape=(None, 1,))

    seq_model = tf.keras.Sequential()
    seq_model.add(Input(shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3)))
    seq_model.add(Dropout(0.55))

    time_distributed_layers = [
        #Conv2D(32, 3, activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        #Conv2D(32, 3, activation="relu"),
        #MaxPool2D(pool_size=(2, 2)),

        Conv2D(16, 3, activation="relu"),
        Conv2D(16, 3, activation="relu"),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(8, 3, activation="relu"),
        Conv2D(8, 3, activation="relu"),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(4, 3, activation="relu"),
        Conv2D(4, 3, activation="relu"),
        MaxPool2D(pool_size=(2, 2)),

        Flatten(),
    ]

    # Wrap every layer before GRU in TimeDistributed, because Conv2D cannot deal with 5 dimensions
    for layer in time_distributed_layers:
        seq_model.add(TimeDistributed(layer))

    combined = Concatenate()([seq_model.output, metadata_input])
    x = TimeDistributed(Dense(64, activation="relu"))(combined)
    x = Dropout(0.5)(x)
    x = GRU(32)(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[seq_model.input, metadata_input], outputs=output)
    model.build(input_shape=((None, None, IMAGE_SIZE, IMAGE_SIZE, 3), (None, None, 1)))
    model.summary()
    return model


def load_data(path: str) -> (list, np.ndarray, np.ndarray):
    with open(path, mode="rb") as f:
        data : dict = pickle.load(f)
    return data["x_image_paths"], data["x_metadata"], data["y"]

def save_model_metadata(model_name: str):
    with open("dataset_meta.json", mode="r", encoding="utf-8") as f:
        dataset_metadata = json.load(f)

    model_metadata = {
        "train_sequence_size" : SEQUENCE_SIZE,
        "image_size": (IMAGE_SIZE, IMAGE_SIZE),
        "dataset_meta": dataset_metadata
    }
    with open(f"models/{model_name}.meta.json", mode="w", encoding="utf-8") as f:
        json.dump(model_metadata, f)


def main():
    # Load data
    validation_image_paths, validation_seq_metadata, validation_y = load_data("validation.pickle")
    train_image_paths, train_seq_metadata, train_y = load_data("train.pickle")

    # Get and compile model
    model = get_model()
    model.compile(optimizer="adam",
                  loss="mean_squared_error") # Maybe should be binary_crossentropy?
    #model = tf.keras.models.load_model("models/v2_smol_less_gru_var_less_dense_end.model")
    # Setup generators
    batch_size = 32
    img_server = ImageProcessServerConnect("E:\Download\Data\ml\image_cache", True, working_dir=DATA_FOLDER)
    custom_training_batch_generator = CustomBatchLoadGenerator(train_image_paths, train_seq_metadata, train_y, batch_size, img_server)
    custom_validation_batch_generator = CustomBatchLoadGenerator(validation_image_paths, validation_seq_metadata, validation_y, batch_size, img_server)

    # Train
    print("[*] Training model")
    model_name = "v2_smol_less_gru_med_dense_more_drop_max_duration_more_before"
    save_model_metadata(model_name)

    model.fit(
        x=custom_training_batch_generator,
        validation_data=custom_validation_batch_generator,
        epochs=30,
        steps_per_epoch=len(train_image_paths) // batch_size,
        validation_steps=len(validation_image_paths) // batch_size,
        verbose=1,
        callbacks=[
            ModelCheckpoint(f"models/{model_name}_best.model", monitor='val_loss', save_best_only=True, verbose=0, save_format="h5"),
            TensorBoard(log_dir=f"./logs/{model_name}")
        ]
    )
    print("[*] Saving model")
    model.save(f"models/{model_name}_end.model")


if __name__ == "__main__":
    main()