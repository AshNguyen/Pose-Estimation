import os
import scipy.io
import pandas as pd
import numpy as np

class classification_model:
    def __init__(self, save_dir, nframe, n_class=15, n_latent_layer = 3, latent_unit = 100, dropout_rate = 0.2, lr = 0.001, early_stop=True):
        from keras.layers import Input, Dense, LSTM, Dropout, Bidirectional
        from keras.models import Model
        from keras.optimizers import Adam, RMSprop, Adadelta 
        try:
            from keras.models import load_model
            self.model = load_model(save_dir)
       	except:
	        self.early_stop = early_stop
	        self.save_dir = save_dir
	        inputs = Input(shape=(nframe, 13*3))
	        x = LSTM(latent_unit, return_sequences=True)(inputs)
	        for _ in range(n_latent_layer-1):
	            x = Dropout(dropout_rate)(x)
	            x = LSTM(latent_unit, return_sequences=True)(x)
	        x = Dropout(dropout_rate)(x)
	        x = LSTM(latent_unit)(x)
	        x = Dropout(dropout_rate)(x)
	        pred = Dense(n_class, activation='softmax')(x)

	        self.model = Model(inputs=inputs, outputs=pred)
	        self.optimizer = RMSprop(lr=lr)
	        self.model.compile(optimizer=self.optimizer,
	                      loss='categorical_crossentropy',
	                      metrics=['acc'])

    def training(self, train_input, train_label, val_input, val_label, n_epoch=50):
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        cp = ModelCheckpoint(self.save_dir, save_best_only=True)
        es = EarlyStopping(monitor="val_acc")

        if self.early_stop:
            self.history = self.model.fit(train_input, train_label, epochs=n_epoch, batch_size=128, validation_data=(val_input, val_label), callbacks=[es, cp])
        else: self.history = self.model.fit(train_input, train_label, epochs=n_epoch, batch_size=128, validation_data=(val_input, val_label), callbacks=[cp])
        return self.history

    def training_generator(self, train_generator, val_generator, n_step, n_step_val, n_epoch=50):
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        cp = ModelCheckpoint(self.save_dir, save_best_only=True)
        es = EarlyStopping(monitor="val_acc")
        if self.early_stop:
            self.history = self.model.fit_generator(generator=train_generator, epochs=n_epoch, steps_per_epoch=n_step, validation_data=val_generator, validation_steps=n_step_val, callbacks=[es, cp])
        else: self.history = self.model.fit_generator(generator=train_generator, epochs=n_epoch, steps_per_epoch=n_step, validation_data=val_generator, validation_steps=n_step_val, callbacks=[cp])
        return self.history

    def predict(self, _input):
        return self.model.predict(_input)


class classification_model_latent:
    def __init__(self, save_dir, n_class=15, n_latent_layer = 3, latent_unit = 100, dropout_rate = 0.2, lr = 0.001, early_stop=True):
        from keras.layers import Input, Dense, LSTM, Dropout, Bidirectional
        from keras.models import Model
        from keras.optimizers import Adam, RMSprop, Adadelta 
        try:
            from keras.models import load_model
            self.model = load_model(save_dir)
        except:
	        self.early_stop = early_stop
	        self.save_dir = save_dir
	        inputs = Input(shape=(80, ))
	        x = LSTM(latent_unit, return_sequences=True)(inputs)
	        for _ in range(n_latent_layer-1):
	            x = Dropout(dropout_rate)(x)
	            x = LSTM(latent_unit, return_sequences=True)(x)
	        x = Dropout(dropout_rate)(x)
	        x = LSTM(latent_unit)(x)
	        x = Dropout(dropout_rate)(x)
	        pred = Dense(n_class, activation='softmax')(x)

	        self.model = Model(inputs=inputs, outputs=pred)
	        self.optimizer = RMSprop(lr=lr)
	        self.model.compile(optimizer=self.optimizer,
	                      loss='categorical_crossentropy',
	                      metrics=['acc'])

    def training(self, train_input, train_label, val_input, val_label, n_epoch=50):
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        cp = ModelCheckpoint(self.save_dir, save_best_only=True)
        es = EarlyStopping(monitor="val_acc")

        if self.early_stop:
            self.history = self.model.fit(train_input, train_label, epochs=n_epoch, batch_size=128, validation_data=(val_input, val_label), callbacks=[es, cp])
        else: self.history = self.model.fit(train_input, train_label, epochs=n_epoch, batch_size=128, validation_data=(val_input, val_label), callbacks=[cp])
        return self.history

    def training_generator(self, train_generator, val_generator, n_step, n_step_val, n_epoch=50):
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        cp = ModelCheckpoint(self.save_dir, save_best_only=True)
        es = EarlyStopping(monitor="val_acc")
        if self.early_stop:
            self.history = self.model.fit_generator(generator=train_generator, epochs=n_epoch, steps_per_epoch=n_step, validation_data=val_generator, validation_steps=n_step_val, callbacks=[es, cp])
        else: self.history = self.model.fit_generator(generator=train_generator, epochs=n_epoch, steps_per_epoch=n_step, validation_data=val_generator, validation_steps=n_step_val, callbacks=[cp])
        return self.history

    def predict(self, _input):
        return self.model.predict(_input)