from typing import Dict
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp


# get data
features = list(pd.read_csv("train_data.csv", nrows=1))
X = pd.read_csv("train_data.csv", usecols=[col for col in features if col != "class"])
y = pd.read_csv("train_data.csv", usecols=["class"])
test_data_df = pd.read_csv("test_data.csv")

# split into train and validation sets
validation_frac = 0.20
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_frac)

# normalize dataset because features have greatly varying ranges
normalize = tf.keras.layers.Normalization()
normalize.adapt(X_train)

# create model
model = tf.keras.Sequential([
    normalize,
    tf.keras.layers.Dense(len(features), activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation="sigmoid"),  # sigmoid as we want a single label classification b/w 0 and 1
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

# train (keras shuffles the dataset for you)
model.fit(X_train, y_train, epochs=150, verbose=2)

# evaluate on validation set
print("\nEvaluation:")
model.evaluate(X_validation, y_validation, verbose=2)

# predict
preds = model.predict(test_data_df)
preds = tf.greater(preds, .5)  # convert probabilities to labels
preds = tf.cast(preds, tf.int8)  # convert bools to int
preds = tf.transpose(preds)  # easier to read 1D
print(preds)


#######
# below is code I used for various hyperparameter tuning
def tune_hyperparams() -> None:
    """
    Runs hyperparam tuning and logs results

    :return None
    """
    session_num = 0
    num_units_tuning_vals = [10, 20, 30, 40, 50, len(features)]
    dropout_tuning_vals = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    num_epochs_tuning_vals = [20, 50, 100, 150, 200]
    learning_rate_tuning_vals = [1e-2, 1e-3, 1e-4]

    for num_units in num_units_tuning_vals:
        for dropout_rate in dropout_tuning_vals:
            for num_epochs in num_epochs_tuning_vals:
                for learning_rate in learning_rate_tuning_vals:
                    hparams = {
                        'num_units': num_units,
                        'dropout': dropout_rate,
                        'num_epochs': num_epochs,
                        'learning_rate': learning_rate,
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h: hparams[h] for h in hparams})
                    train_test_model(hparams, 'logs/hparam_tuning3/' + run_name)
                    session_num += 1


def train_test_model(hparams: Dict, logdir: str) -> None:
    """
    Helper function used for tuning hyperparameters

    :param hparams: dict of hyper params name to value
    :param logdir: directory to save logs for hyperparam performance
    :return: None
    """
    model = tf.keras.Sequential([
        normalize,
        tf.keras.layers.Dense(len(features), activation="relu"),
        tf.keras.layers.Dropout(hparams['dropout']),
        tf.keras.layers.Dense(hparams['num_units'], activation="relu"),
        tf.keras.layers.Dropout(hparams['dropout']),
        tf.keras.layers.Dense(1, activation="sigmoid"),  # sigmoid as we want a single label classification b/w 0 and 1
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate']),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train, epochs=hparams['num_epochs'],
        callbacks=[
            tf.keras.callbacks.TensorBoard(logdir),  # log metrics
            hp.KerasCallback(logdir, hparams),  # log hparams
        ],
    )
    _, accuracy = model.evaluate(X_validation, y_validation)
    return accuracy

#tune_hyperparams()
