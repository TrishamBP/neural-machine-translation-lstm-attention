import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Setting this env variable prevents TF warnings from showing up
import tensorflow as tf
from utils import (sentences, train_data, val_data, english_vectorizer, portuguese_vectorizer,
                   masked_loss, masked_acc, tokens_to_text)
from translator import Translator

VOCAB_SIZE = 12000
UNITS = 256

def compile_and_train(model, epochs=20, steps_per_epoch=500):
    model.compile(optimizer="adam", loss=masked_loss, metrics=[masked_acc, masked_loss])

    history = model.fit(
        train_data.repeat(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
    )

    return model, history

translator = Translator(VOCAB_SIZE, UNITS)

trained_translator, history = compile_and_train(translator)
