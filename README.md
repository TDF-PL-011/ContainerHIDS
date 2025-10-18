# CHIDS

| CHIDS is an unsupervised anomaly-based host intrusion detection system for containers. CHIDS relies on monitoring heterogeneous properties of system calls (syscalls). The development of CHIDS is based on the premise that malicious activities can be accurately uncovered when various syscall properties (e.g., frequency, arguments) are inspected jointly within their context. In detail, CHIDS learns container "normal" behavior and flags deviations in production. | ![ContainerHIDS](https://i0.wp.com/foxutech.com/wp-content/uploads/2017/03/Docker-Security.png?fit=820%2C407&ssl=1 "ContainerHIDS") |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------:|

## Research paper

We present our approach and the findings of this work in the following research paper:

**Contextualizing System Calls in Containers for Anomaly-Based Intrusion Detection** 
[[PDF]](https://conand.me/publications/elkhairi-ccsw-2022.pdf)  
Asbat El Khairi, Marco Caselli, Christian Knierim, Andreas Peter, Andrea Continella.  
*Proceedings of the ACM Cloud Computing Security Workshop (CCSW), 2022*

## CHIDS Architecture
<img src="figures/architecture.png" width="100%">

- **Syscalls Chunking.**
We divide the ongoing flow of system calls into short captures *scaps*, defined as *syscall sequences*. 

- **Syscalls Encoding.** 
We transform each syscall sequence into a *syscall sequence graph* *(SSG)*, from which we extract three features forming the *anomaly vector* *(AV)*. 

- **Model Training.**  We feed the anomaly vector into an unsupervised auto-encoder neural network for training. This training aims to minimize the reconstruction loss and generate a behavioral baseline that represents a given container's normal behavior.

- **Anomaly Detection.**
We classify an anomaly vector based on the trained model and a selected threshold in production.

## Get Training Elements

The training elements are the following:
   - previously seen syscalls
   - previously seen args
   - the max SSG training size or (max len of training sequences)
   - the thresholds list
   - the trained model
```
$> python3 main.py baseline  --td training_dir --od output_dir
```
### Example: Brute-Force Login (CWE-307) Training Summary

<img src="figures/screenshots/training_results.png" width="100%">

## Get Evaluation Results 

The evaluation script takes as inputs the following elements: 
   - previously seen syscalls
   - previously seen args
   - the max SSG training size or (max len of training sequences)
   - the thresholds list
   - the trained model
```
$> python3 main.py evaluate --ss output_dir/seen_syscalls.pkl --sa output_dir/seen_args.pkl --fm  output_dir/max_freq.pkl --tm  output_dir/model.h5 --tl output_dir/thresh_list.pkl --ns normal_scaps --ms malicious_scaps 
```

### Example: Brute-Force Login (CWE-307) Evaluation Summary

<img src="figures/screenshots/evaluation_results.png" width="40%">


## Alternative models:

### Dense autoencoder that respects sequence shape

```
    def _autoencoder_model(self, vectors):
        T, F = vectors.shape[1], vectors.shape[2]
        inputs = L.Input(shape=(T, F))

        # Flatten sequence
        x = L.Flatten()(inputs)                                 # shape: (T*F,)

        # Encoder
        x = L.Dense(ENCODING_DIM, kernel_regularizer=R.l2(1e-4))(x)
        x = L.BatchNormalization()(x)
        x = L.LeakyReLU()(x)
        z = L.Dense(BOTTLENECK,
                    activation="linear",
                    activity_regularizer=R.l1(REG_RATE))(x)     # sparse bottleneck

        # Decoder
        x = L.Dense(ENCODING_DIM, kernel_regularizer=R.l2(1e-4))(z)
        x = L.BatchNormalization()(x)
        x = L.LeakyReLU()(x)
        x = L.Dense(T*F, activation="linear")(x)

        # Reshape back to (T, F)
        outputs = L.Reshape((T, F))(x)

        model = tf.keras.Model(inputs, outputs)
        return model
```


### Enhanced encoder with batch normalization and LeakyReLU

```
    def _autoencoder_model(self, vectors):
        inputs = Input(shape=(vectors.shape[1], vectors.shape[2]))

        # Enhanced encoder with batch normalization and LeakyReLU
        L1 = Dense(ENCODING_DIM * 2, activation="linear")(inputs)
        L1 = keras.layers.LeakyReLU(alpha=0.1)(L1)
        L1 = keras.layers.BatchNormalization()(L1)
        L1 = keras.layers.Dropout(0.2)(L1)

        L2 = Dense(ENCODING_DIM, activity_regularizer=regularizers.l1(REG_RATE))(L1)
        L2 = keras.layers.LeakyReLU(alpha=0.1)(L2)
        L2 = keras.layers.BatchNormalization()(L2)

        L3 = Dense(BOTTLENECK)(L2)
        L3 = keras.layers.LeakyReLU(alpha=0.1)(L3)

        # Enhanced decoder
        L4 = Dense(ENCODING_DIM)(L3)
        L4 = keras.layers.LeakyReLU(alpha=0.1)(L4)
        L4 = keras.layers.BatchNormalization()(L4)

        L5 = Dense(ENCODING_DIM * 2)(L4)
        L5 = keras.layers.LeakyReLU(alpha=0.1)(L5)

        output = Dense(vectors.shape[2], activation=ACTIVATION)(L5)
        model = Model(inputs=inputs, outputs=output)
        return model
```


### LTSM autoencoder

```
    def _autoencoder_model(self, vectors):
        T, F = vectors.shape[1], vectors.shape[2]
        inputs = L.Input(shape=(T, F))

        # Encoder
        x = L.Masking()(inputs)                                 # if you use padding
        x = L.LSTM(ENCODING_DIM, return_sequences=True)(x)
        x = L.LayerNormalization()(x)
        z = L.LSTM(BOTTLENECK, return_sequences=False,
                activation="tanh")(x)                        

        # Decoder
        x = L.RepeatVector(T)(z)
        x = L.LSTM(ENCODING_DIM, return_sequences=True)(x)
        x = L.TimeDistributed(L.Dense(F, activation="linear"))(x)

        model = tf.keras.Model(inputs, x)
        return model
```

### Temporal Conv (fast, scalable)

```
    def _autoencoder_model(self, vectors):
        T, F = vectors.shape[1], vectors.shape[2]
        inputs = L.Input(shape=(T, F))
        x = L.Conv1D(64, 3, padding="causal")(inputs)
        x = L.ReLU()(x)
        x = L.Conv1D(64, 3, dilation_rate=2, padding="causal")(x)
        x = L.ReLU()(x)
        x = L.Conv1D(BOTTLENECK, 3, dilation_rate=4, padding="causal")(x)
        z = L.GlobalAveragePooling1D()(x)

        x = L.RepeatVector(T)(z)
        x = L.Conv1DTranspose(64, 3, padding="same")(x)
        x = L.ReLU()(x)
        x = L.Conv1DTranspose(64, 3, padding="same")(x)
        outputs = L.TimeDistributed(L.Dense(F, activation="linear"))(x)

        model = tf.keras.Model(inputs, outputs)
        return model
```
