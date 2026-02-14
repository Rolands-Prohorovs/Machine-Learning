## 0) Imports

```python
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_circles 
```
- TensorFlow: build/train neural networks
- NumPy: arrays ops, masks, reshape
- Matplotlib: plots, images, training curves
- Sklearn: datasets and spliting to train/test

## 1. Load dataset
**Goal:** get (X, y) for training and testing

### Example A: Fashion-MNIST - multiclass images
```python
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```

### Example B: CIFAR-10 - color images
```python
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()
```
**Note**
by many keras loss functions and metrics it is required to have labes stored in shape (N,). So using .flatten() function we convert from (N, 1) to (N, ).

### Example C: Circles - bianry, non-linear
```python
    X, y = make_circles(n_samples=1000, noise=0.03, random_state=42)
```

## 2. Explore/ check shapes 
```python
    print(x_train.shape, y_train.shape)
    print(x_train.dtype, x_train.min(), x_train.max())
```
**For Fashion-MNIST you expect:**
- x_train: (60000, 28, 28)
- y_train: (60000,)
- dtype often uint8, range 0..255


## 3. Mormalize/preprocess data
**Goal:** make training stable and faster 

### Images: scale pixels to [0, 1]
```python
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
```

### CNN: needs channel dimension
Fashino-MNIST is (28, 28), CNN expects (28, 28, 1):
```python
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
```

## 4. Split data (train/validation/test)
**Goal:** train, track overfitting, and evaluate honestly

### Option A (Keras): use validation_split
```python
    history = model.fit(x_train, y_train, epoch=10, validation_split=0.1)
```

### Option B (sklearn): train_test_split
```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```
- train: learn
- validation: tune/ detect overfitting
- test: finale evaluation 

## 5. Create the model (architecture depends on task)

### A: Binary classification (circles) - Dense network
```python
    model= tf.keras.Sequential([
        tf.keras.layer.Input(shape=(2,)),
        tf.keras.layer.Dense(16, activation="relu"),
        tf.keras.layer.Dense(16, activation="relu"),
        tf.keras.layer.Dense(1, activation="sigmoid"),
    ])
```
- output: `sigmoid`
- units: `1`

### B: Multiclass classification (Fashion-MNIST) - Dense baseline
```python
    model = tf.keras.Sequential([
        tf.keral.layer.Input(shape(28, 28)),
        tf.keras.layer.Flatten(),
        tf.keras.layer.Dense(128, activation="relu"),
        tf.keras.layer.Dense(1, "softmax")
    ])
```
- output: `softmax`
- units: `#classes` (10)


### C: CNN for images (Fashion-MNIST)
```python
    model = tf.keras.Sequential([
        tf.keral.layer.Input(shape=(28,28,1)),
        tf.keras.layer.Conv2D(32,3, activation="relu", padding="same"),
        tf.keras.layer.MaxPool2D(),
        tf.keras.layer.Conv2D(64, 3, activation="relu", padding="same")
        tf.keral.layer.MaxPool(),
        tf.keras.layer.Flatten(),
        tf.keras.layer.Dense(128,activation="relu")
        tf.keras.layer.Dropout(0.3),
        tf.keras.layer.Dense(10, activation="softmax")
    ])
```
**CNN**
- keeps spatial info
- learns edges - shapes - objects

### D: Transfer learning (feature extraction) -- pretrained base + new head
```python
    IMG_SIZE = 96

    preprocess = tf.keras.Sequential([
        tf.keras.layer.Resizing(IMG_SIZE, IMG_SIZE),
        tf.keras.layer.Lambda(tf.keras.application.mobilenet_v2.preprocess_input)
    ])

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_trainable = False

    model = tf.keras.Sequential([
        tf.keras.layer.Inputs(shape=(32,32,3)),
        preprocess,
        base,
        tf.keras.layer.GlobalAveragePooling2D(),
        tf.keras.layer.Dense(3, activation="softmax")
    ])
```
- freeze base, train only head


## 6. Compile (loss + optimazer + metrics)
**Goal:** tell TF how to learn
**Loss function**(must match output + lables)

### Binary classification
- output: `softmax`
- loss: `sparce_categorical_crossentory
```python
    loss="sparce_categorical_crossentory"
```

### Multiclass (one-hot labels)
- output: `softmax`
- loss: `categorical_crossentropy`
```python
    loss="categorical_crossentropy"
```

### Optimazer
**Adam**(default best for most tasks)
```python
    optimazer="adam"
```
With learning rate:
```python
    optimazer=tf.keras.optimazer.Adam(learning_rate=1e-3)
```

**SGD**(classic,soemtimes used in exersises)
```python
    optimazer=tf.keras.optimazer.SGD(learning_rate=0.01)
```

**Metrics**
- calssification: `accuracy`
- regresion: `mae`, `mse`

### Full compile examples:
**Binary**
```python
    model.compile(optimizer="adam", loss="bianry_crossentropy", metrics=["accurasy"])
```

**Multiclass**
```python
    model.compile(optimazer="adam", loss="sparse_categorical_crossentropy", metrics=["accurase"])
```

## 7. Train(Fit)
**Goal:** run learning
```python
    history= model.fit(
        x_train, y_train,
        eposch=10,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
```
- `epochs`: more training
- `batch_size`: speed vs memory
- `validation_split`: track generalization

## 8. Evaluate on test set
**Goal:** honest finale score
```python
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print("test accurasy:", acc)
```

## 9. Predict and interpret outputs

### Binary: probability -> class by threshold
```python
        probs = model.predict(x_test[:5])
        preds = (probs > 0.5).astype(int)
```

### Multiclass: softmax -> argmax
```python
    probs = model.predict(x_test[:8])
    preds = model.argmax(axis=1)
```

## 10. Visualize result

### Plot training curves
```python
    plt.plot(history.history["accuracy"], lable="training acc")
    plt.plot(history.history["val_accuracy"], label="vall acc")
    plt.legend(); plt.show()
```


### Show predicted images(Fashion-MNIST)
```python
    plt.plot(x_test[0], cmap="gray")
    plt.plot(f"Pred: {preds[0]}")
    plt.axis("off")
    plt.show()
```
