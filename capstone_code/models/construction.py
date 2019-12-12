import numpy as np

class Model_contract:
    def __init__(self, save_dir, latent_activation='relu', output_activation='sigmoid'):
        from keras.layers import Input, Dense, Dropout
        from keras.models import Model
        from keras.optimizers import Adam
        assert latent_activation in ['relu', 'sigmoid', 'elu', 'tanh']
        assert output_activation in ['relu', 'sigmoid', 'elu', 'tanh']
        if save_dir:
            from keras.models import load_model
            self.model = load_model(save_dir)

        self.save_dir = save_dir
        inputs = Input(shape=(13*3,))
        x = Dense(30, activation=latent_activation)(inputs)
        x = Dense(20, activation=latent_activation)(x)
        latent_outputs = Dense(10, activation=latent_activation)(x)
        self.encoder = Model(inputs=inputs, outputs=latent_outputs)

        latent_inputs = Input(shape=(10, ))
        y = Dense(20, activation=latent_activation)(latent_inputs)
        y = Dense(30, activation=latent_activation)(y)
        outputs = Dense(13*3, activation=output_activation)(y)
        self.decoder = Model(inputs=latent_inputs, outputs=outputs)

        outputs = self.decoder(self.encoder(inputs))
        self.model = Model(inputs=inputs, outputs=outputs)

        self.optimizer = Adam()
        self.model.compile(optimizer=self.optimizer,
                      loss='mse',
                      metrics=['mae'])

    def training(self, train_input, train_label, val_input, val_label, n_epoch = 50):
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        cp = ModelCheckpoint(self.save_dir, save_best_only=True)
        es = EarlyStopping(monitor="val_mean_absolute_error")

        self.history = self.model.fit(train_input, train_label, epochs=n_epoch, batch_size=128, validation_data=(val_input, val_label), callbacks=[es, cp])
        return self.history

    def training_generator(self, train_generator, val_generator, n_step, n_step_val, n_epoch = 50):
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        cp = ModelCheckpoint(self.save_dir, save_best_only=True)
        es = EarlyStopping(monitor="val_mean_absolute_error")

        self.history = self.model.fit_generator(generator=train_generator, epochs=n_epoch, steps_per_epoch=n_step, validation_data=val_generator, validation_steps=n_step_val, callbacks=[es, cp])
        return self.history

    def predict(self, _input):
        return self.model.predict(_input)


###########################################################################################################################################################################################################

class Model_expand:
    def __init__(self, save_dir, latent_dim = 120, n_latent_layer = 3, latent_activation='relu', output_activation='sigmoid', early_stop=True):
        from keras.layers import Input, Dense, Dropout
        from keras.models import Model
        from keras.optimizers import Adam
        d = int(np.floor(latent_dim/n_latent_layer))
        assert d > 13*3
        if save_dir:
            from keras.models import load_model
            self.model = load_model(save_dir)
        self.early_stop = early_stop
        self.save_dir = save_dir
        inputs = Input(shape=(13*3,))
        x = Dense(d, activation= latent_activation)(inputs)
        for _ in range(2, n_latent_layer):
            x = Dense(d*_, activation= latent_activation)(x)
        latent_outputs = Dense(latent_dim, activation= latent_activation)(x)
        self.encoder = Model(inputs=inputs, outputs=latent_outputs)

        latent_inputs = Input(shape=(latent_dim, ))
        y = Dense(d*(n_latent_layer-1), activation= latent_activation)(latent_inputs)
        for _ in range(n_latent_layer-2, 0, -1):
            y = Dense(d*_, activation= latent_activation)(y)
        outputs = Dense(13*3, activation= output_activation)(y)
        self.decoder = Model(inputs=latent_inputs, outputs=outputs)

        outputs = self.decoder(self.encoder(inputs))
        self.model = Model(inputs=inputs, outputs=outputs)

        self.optimizer = Adam()
        self.model.compile(optimizer=self.optimizer,
                      loss='mse',
                      metrics=['mae'])
    def training(self, train_input, train_label, val_input, val_label, n_epoch = 50):
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        cp = ModelCheckpoint(self.save_dir, save_best_only=True)
        es = EarlyStopping(monitor="val_mean_absolute_error")
        if self.early_stop:
            self.history = self.model.fit(train_input, train_label, epochs=n_epoch, batch_size=128, validation_data=(val_input, val_label), callbacks=[es, cp])
        else: self.history = self.model.fit(train_input, train_label, epochs=n_epoch, batch_size=128, validation_data=(val_input, val_label), callbacks=[cp])
        return self.history

    def training_generator(self, train_generator, val_generator, n_step, n_step_val, n_epoch = 50):
        from keras.callbacks import ModelCheckpoint, EarlyStopping
        cp = ModelCheckpoint(self.save_dir, save_best_only=True)
        es = EarlyStopping(monitor="val_mean_absolute_error")
        if self.early_stop:
            self.history = self.model.fit_generator(generator=train_generator, epochs=n_epoch, steps_per_epoch=n_step, validation_data=val_generator, validation_steps=n_step_val, callbacks=[es, cp])
        else: self.history = self.model.fit_generator(generator=train_generator, epochs=n_epoch, steps_per_epoch=n_step, validation_data=val_generator, validation_steps=n_step_val, callbacks=[cp])
        return self.history

    def predict(self, _input):
        return self.model.predict(_input)

   