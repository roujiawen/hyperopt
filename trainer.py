def load_data():
    """
    Ref: https://victorzhou.com/blog/keras-neural-network-tutorial/
    """
    import numpy as np
    import mnist
    from tensorflow.keras.utils import to_categorical

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    # Normalize the images.
    train_images = (train_images / 255) - 0.5
    test_images = (test_images / 255) - 0.5

    # Flatten the images.
    train_images = train_images.reshape((-1, 784))
    test_images = test_images.reshape((-1, 784))

    # One-hot encoding
    train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)

    return (train_images, train_labels, test_images, test_labels)

def train_mnist(hpset):
    """
    Ref: https://victorzhou.com/blog/keras-neural-network-tutorial/
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    train_images, train_labels, test_images, test_labels = load_data()


    model = Sequential([
      Dense(hpset["layer1_nodes"], activation='relu', input_shape=(784,)),
      Dense(hpset["layer2_nodes"], activation='relu'),
      Dense(10, activation='softmax'),
    ])

    model.compile(
      optimizer=hpset["optimizer"],
      loss='categorical_crossentropy',
      metrics=['accuracy'],
    )

    model.fit(
      train_images,
      train_labels,
      epochs=1,
      batch_size=32,
      verbose=0
    )

    logs = model.evaluate(
      test_images,
      test_labels,
      verbose=0
    )

    accuracy = logs[1]

    return (hpset, accuracy, logs)
