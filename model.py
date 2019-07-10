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
		conv = Dropout(.2)(conv)
		conv = Conv2D(32, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dropout(.2)(conv)
		conv = Conv2D(32, (3,3), padding='same', activation= 'relu', kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dropout(.2)(conv)
		conv = Flatten()(conv)
		conv = Dense(50, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(conv)
		# conv = Dense(100, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dense(12, activation = 'softmax')(conv)

		model = Model(cnn_input, conv)

		return model

	def resnet_model(self):

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
		conv = Dense(50, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dense(100, activation = 'relu',kernel_regularizer=regularizers.l2(0.01))(conv)
		conv = Dense(12, activation = 'softmax')(conv)

		model = Model(cnn_input, conv)

		return model






	def run_model(self):

		self.train_set = self.train_set.reshape(self.train_set.shape[0],40,32,1)
		self.test_set = self.test_set.reshape(self.test_set.shape[0],40,32,1)

		

		model = self.cnn_model()
		model.summary()

		model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

		model.fit(self.train_set, self.y_train, batch_size = 32, epochs = 100, validation_data = (self.test_set, self.y_test), verbose = 1)
		# model.save_weights('kws.h5')


if __name__ == '__main__':
	model_kws = Model_KWS()
	model_kws.run_model()



		