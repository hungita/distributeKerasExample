# keras 多主机分布式训练，mnist为例

## 1.概述

由于一般GPU的显存只有11G左右，（土豪误入），采用多主机分布式训练是非常有必要的；折腾了几天，按照谷歌的教程，终于搞清楚了，给大家梳理一下：

参考：https://tensorflow.google.cn/tutorials/distribute/multi_worker_with_keras?hl=be

## 2.配置

首先，设置 TensorFlow 和必要的导入。

```
import os
from tensorflow import keras  #tensorflow-gpu==2.0.0
import tensorflow as tf
import json
```

## 3.准备数据集

这里数据采用的是`tf.data.Dataset.from_tensor_slices`将数据转换成需要的格式，由于分割数据问题，需要添加`.repeat()`；

```
def get_dataset():
    num_val_samples = 10000

    # Return the MNIST dataset in the form of a `tf.data.Dataset`.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).repeat(),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).repeat(),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).repeat(),
        x_train.shape[0], x_val.shape[0], x_test.shape[0]
    )
```

## 4.构建 Keras 模型

在这里，我们使用[`tf.keras.Sequential`](https://tensorflow.google.cn/api_docs/python/tf/keras/Sequential) API来构建和编译一个简单的卷积神经网络 Keras 模型，用我们的 MNIST 数据集进行训练。

注意：有关构建 Keras 模型的详细训练说明，请参阅[TensorFlow Keras 指南](https://tensorflow.google.cn/guide/keras#sequential_model)。

```
def get_compiled_model():
    # Make a simple 2-layer densely-connected neural network.
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
```

在单个工作器（worker）中观察结果，以确保一切正常。 随着训练的迭代，您应该会看到损失（loss）下降和准确度（accuracy）接近1.0。

```
Train for 1562 steps, validate for 2 steps
Epoch 1/2
1562/1562 [==============================] - 2s 1ms/step - loss: 0.2260 - sparse_categorical_accuracy: 0.9320 - val_loss: 0.1047 - val_sparse_categorical_accuracy: 0.9844
Epoch 2/2
1562/1562 [==============================] - 2s 1ms/step - loss: 0.0940 - sparse_categorical_accuracy: 0.9718 - val_loss: 0.0313 - val_sparse_categorical_accuracy: 0.9844
```

详细的见，`keras_one_gpu.py`

## 5.多工作器（worker）配置

`TF_CONFIG` 有两个组件：`cluster` 和 `task` 。 `cluster` 提供有关训练集群的信息，这是一个由不同类型的工作组成的字典，例如 `worker` 。在多工作器（worker）培训中，除了常规的“工作器”之外，通常还有一个“工人”承担更多责任，比如保存检查点和为 TensorBoard 编写摘要文件。这样的工作器（worker）被称为“主要”工作者，习惯上`worker` 中 `index` 0被指定为主要的 `worker`（事实上这就是[`tf.distribute.Strategy`](https://tensorflow.google.cn/api_docs/python/tf/distribute/Strategy)的实现方式）。 另一方面，`task` 提供当前任务的信息。

在这个例子中，我们将任务 `type` 设置为 `"worker"` 并将任务 `index` 设置为 `0` 。这意味着具有这种设置的机器是第一个工作器，它将被指定为主要工作器并且要比其他工作器做更多的工作。请注意，其他机器也需要设置 `TF_CONFIG` 环境变量，它应该具有相同的 `cluster` 字典，但是不同的任务`type` 或 `index` 取决于这些机器的角色。

为了便于说明，本教程展示了如何在 `localhost` 上设置一个带有2个工作器的`TF_CONFIG`。 实际上，用户会在外部IP地址/端口上创建多个工作器，并在每个工作器上适当地设置`TF_CONFIG`。

```
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})
```

示例：

假如有2个台主机，在对应的程序里的配置如下：

```
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["IP1:12345", "IP2:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})
```

```
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["IP1:12345", "IP2:23456"]
    },
    'task': {'type': 'worker', 'index': 1}
})
```

## 6.选择正确的策略

在 TensorFlow 中，分布式训练包括同步训练（其中训练步骤跨工作器和副本同步）、异步训练（训练步骤未严格同步）。

`MultiWorkerMirroredStrategy` 是同步多工作器训练的推荐策略，将在本指南中进行演示。

要训练模型，请使用 [`tf.distribute.experimental.MultiWorkerMirroredStrategy`](https://tensorflow.google.cn/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy) 的实例。 `MultiWorkerMirroredStrategy` 在所有工作器的每台设备上创建模型层中所有变量的副本。 它使用 `CollectiveOps` ，一个用于集体通信的 TensorFlow 操作，来聚合梯度并使变量保持同步。 [`tf.distribute.Strategy`指南](https://tensorflow.google.cn/guide/distribute_strategy)有关于此策略的更多详细信息。

```python
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
```

这句话要写在`TF_CONFIG`的后面，不能放太后了，不然会报错；

## 7.使用 MultiWorkerMirroredStrategy 训练模型

通过将 [`tf.distribute.Strategy`](https://tensorflow.google.cn/api_docs/python/tf/distribute/Strategy) API集成到 [`tf.keras`](https://tensorflow.google.cn/api_docs/python/tf/keras) 中，将训练分发给多人的唯一更改就是将模型进行构建和 `model.compile()` 调用封装在 `strategy.scope()` 内部。 分发策略的范围决定了如何创建变量以及在何处创建变量，对于 MultiWorkerMirroredStrategy 而言，创建的变量为 MirroredVariable ，并且将它们复制到每个工作器上。

```
with strategy.scope():
    model = make_or_restore_model()
model.fit(train_dataset, epochs=2, validation_data=val_dataset, validation_steps=2,
          steps_per_epoch=train_shape // batch_size)
```

## 8.容错能力

在同步训练中，如果其中一个工作器出现故障并且不存在故障恢复机制，则集群将失败。 在工作器退出或不稳定的情况下，将 Keras 与 [`tf.distribute.Strategy`](https://tensorflow.google.cn/api_docs/python/tf/distribute/Strategy) 一起使用会具有容错的优势。 我们通过在您选择的分布式文件系统中保留训练状态来做到这一点，以便在重新启动先前失败或被抢占的实例后，将恢复训练状态。

由于所有工作器在训练 epochs 和 steps 方面保持同步，因此其他工作器将需要等待失败或被抢占的工作器重新启动才能继续。

### ModelCheckpoint 回调

要在多工作器训练中利用容错功能，请在调用 [`tf.keras.Model.fit()`](https://tensorflow.google.cn/api_docs/python/tf/keras/Model#fit) 时提供一个 [`tf.keras.callbacks.ModelCheckpoint`](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks/ModelCheckpoint) 实例。 回调会将检查点和训练状态存储在与 `ModelCheckpoint` 的 `filepath` 参数相对应的目录中。

```
with strategy.scope():
    model = make_or_restore_model()

callbacks = [
    # This callback saves a SavedModel every 100 batches
    keras.callbacks.ModelCheckpoint(filepath='path/to/cloud/location/ckpt',
                                    save_freq=100),
    keras.callbacks.TensorBoard('path/to/cloud/location/tb/')
]
model.fit(train_dataset, epochs=2, validation_data=val_dataset, validation_steps=2,
          steps_per_epoch=train_shape // batch_size)
```

如果某个工作线程被抢占，则整个集群将暂停，直到重新启动被抢占的工作线程为止。工作器重新加入集群后，其他工作器也将重新启动。 现在，每个工作器都将读取先前保存的检查点文件，并获取其以前的状态，从而使群集能够恢复同步，然后继续训练。

github:  https://github.com/hungita/distributeKerasExample.git

# keras真香！！！