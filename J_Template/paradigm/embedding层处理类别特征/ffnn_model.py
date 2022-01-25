from tensorflow import keras


def MyFFNN(num_input_shape, hidden_units, input_dim, output_dim):
    """使用embedding层处理类别特征"""
    # Each instance will consist of two inputs: a single user id, and a single movie id
    stock_id_input = keras.Input(shape=(1,), name='stock_id')
    num_input = keras.Input(shape=(num_input_shape,), name='num_data')

    # embedding, flatenning and concatenating
    stock_embedded = keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim,
                                            input_length=1, name='stock_embedding')(stock_id_input)
    stock_flattened = keras.layers.Flatten()(stock_embedded)
    out = keras.layers.Concatenate()([stock_flattened, num_input])

    # Add one or more hidden layers
    for n_hidden in hidden_units:
        out = keras.layers.Dense(n_hidden, activation='swish')(out)

    # A single output: our predicted rating
    out = keras.layers.Dense(1, activation='linear', name='prediction')(out)

    model = keras.Model(
        inputs=[stock_id_input, num_input],
        outputs=out,
    )

    return model
