from src.ia.util import *
from config import *


def train(model_name):

    # Loading data
    images, labels = load_data(DATA_FOLDER, CATEGORIES, IMAGE_WIDTH, IMAGE_HEIGTH)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(labels), test_size=TEST_SIZE
            )

    # Get a compiled model
    model = get_model(IMAGE_WIDTH, IMAGE_HEIGTH, CATEGORIES)

    # Fitting the model using training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluating model performance
    model.evaluate(x_test, y_test, verbose=2)

    # Saving model
    model_file_path = os.path.join(MODEL_FOLDER, MODEL_NAME if model_name == None else f"{model_name}.keras")
    model.save(model_file_path)
