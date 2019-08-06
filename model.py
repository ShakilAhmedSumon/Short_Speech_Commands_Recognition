from keras import *
from keras.layers import *
from keras.regularizers import *
from preprocess import Preprocess
import keras


class Model_KWS():
	def __init__(self):
		processed = Preprocess()
		self.train_set, self.test_set, self.y_train, self.y_test = processed.train_test_split()
		self.y_train = keras.utils.to_categorical(self.y_train)
		self.y_test = keras.utils.to_categorical(self.y_test)


	def cnn_model(self):



		cnn_input = Input(shape = (40,32,1))

		conv = Conv2D(32, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(cnn_input)
		# conv = Dropout(.2)(conv)
		conv = Conv2D(16, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dropout(.2)(conv)
		conv = Conv2D(8, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dropout(.2)(conv)
		conv = Flatten()(conv)
		conv = Dense(50, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(conv)
		# conv = Dense(100, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dense(12, activation = 'softmax')(conv)

		model = Model(cnn_input, conv)

		return model

	def residual_block(self,inp):
		conv = Conv2D(45, (3,3), padding='same', activation= 'relu', dilation_rate=(1, 1),use_bias=False,kernel_regularizer=regularizers.l2(0.01))(inp)
		bn = keras.layers.BatchNormalization()(conv)
		conv = Conv2D(45, (3,3), padding='same', activation= 'relu', dilation_rate=(1, 1), use_bias=False,kernel_regularizer=regularizers.l2(0.01))(conv)
		bn = keras.layers.BatchNormalization()(conv)

		conv_merged = Add()([inp, bn])

		return conv_merged



	def resnet_kws(self):

		cnn_input = Input(shape = (40,32,1))
		conv = Conv2D(45, (3,3), padding='same', activation= 'relu', dilation_rate=(1, 1), use_bias=False, kernel_regularizer=regularizers.l2(0.01))(cnn_input)

		for i in range(5):
			conv = self.residual_block(conv)

		conv = Conv2D(45, (3,3), padding='same', activation= 'relu', dilation_rate=(1, 1), use_bias=False, kernel_regularizer=regularizers.l2(0.01))(conv)
		bn = keras.layers.BatchNormalization()(conv)
		pool = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(bn)

		pool = Flatten()(pool)
		dense = Dense(12, activation = 'softmax')(pool)

		model = Model(cnn_input, dense)

		return model



	def resnet_model_prev(self):

		cnn_input = Input(shape = (40,32,1))

		conv = Conv2D(64, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(cnn_input)
		conv = Dropout(.2)(conv)
		conv = Conv2D(32, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dropout(.2)(conv)
		conv = Conv2D(8, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(conv)
		# conv = Dropout(.2)(conv)
		
		conv_shortcut = Conv2D(8, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(cnn_input)
		conv_merged = Add()([conv, conv_shortcut])
		conv_merged = Activation('relu')(conv_merged)

		conv = Flatten()(conv_merged)
		# conv = Dense(50, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(conv)
		# conv = Dense(100, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dense(12, activation = 'softmax')(conv)

		model = Model(cnn_input, conv)

		return model

	def lstm_model(self):

		lstm_input = Input(shape = (40,32))

		lstm = CuDNNLSTM(32, return_sequences = True,kernel_regularizer=regularizers.l2(0.01))(lstm_input)
		lstm = CuDNNLSTM(32, return_sequences = True,kernel_regularizer=regularizers.l2(0.01))(lstm)
		lstm = CuDNNLSTM(10)(lstm)
		dense = Dense(100, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(lstm)
		# dense = Dropout(.2)(dense)
		dense = Dense(100, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(dense)
		# dense = Dropout(.2)(dense)
		dense = Dense(100, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(dense)
		dense = Dense(12, activation = 'softmax')(dense)

		model = Model(lstm_input , dense)

		return model


	def reshape_data(self):
		self.train_set = self.train_set.reshape((self.train_set.shape[0],40,32,1))
		self.test_set = self.test_set.reshape((self.test_set.shape[0],40,32,1))

	def run_model(self):

		# self.train_set = self.train_set.reshape(self.train_set.shape[0],40,32,1)
		# self.test_set = self.test_set.reshape(self.test_set.shape[0],40,32,1)

		

		model = self.resnet_kws()
		self.reshape_data()
		model.summary()

		model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

		model.fit(self.train_set, self.y_train, batch_size = 100, epochs = 100, validation_data = (self.test_set, self.y_test), verbose = 1)
		# model.save_weights('kws.h5')


if __name__ == '__main__':
	model_kws = Model_KWS()
	model_kws.run_model()



		