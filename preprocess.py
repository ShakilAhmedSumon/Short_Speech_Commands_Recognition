import numpy as np 
import os
import librosa


class Preprocess():
	def __init__(self):
		self.folder_list = ['yes', 'no', 'go', 'up', 'down', 'one', 'two', 'three','bed' , 'cat', 'dog', 'happy']
		self.DATA_PATH = 'speech_commands/'
		self.max_length = 40
		self.numpy_files = os.listdir('numpy_files/')
		self.numpy_path = 'numpy_files/'

	def read_audio(self):
		for folder in self.folder_list:
			files = os.listdir(self.DATA_PATH  + folder + '/')
			audios = []
			for file in files:
				audio, sr = librosa.load(self.DATA_PATH + folder + '/' + file, mono = True, sr = 16000)
				mfcc = librosa.feature.mfcc(audio, sr = 16000, n_mfcc = 40 )

				if mfcc.shape[1] > 32:
					mfcc = mfcc[:,:32]
				elif mfcc.shape[1] < 32:
					pad_length = 32 - mfcc.shape[1]
					mfcc = np.pad(mfcc, ((0,0),(0,pad_length)), mode = 'constant')
				


				audios.append(mfcc)
			vugichugi = np.asarray(audios)
			np.save(folder + '.npy', vugichugi)

	def train_test_split(self):
		train_set = np.zeros((5,40,32))
		test_set = np.zeros((5,40,32))
		y_train = np.zeros(5)
		y_test = np.zeros(5)
		i = 0

		for file in self.numpy_files:
			numpy_file = np.load(self.numpy_path + file)
			train_set = np.concatenate((train_set,numpy_file[:1500]),axis = 0 )
			test_set = np.concatenate((test_set, numpy_file[1500:1600]), axis = 0)

			labels_train = np.ones(1500) * i
			labels_test = np.ones(100) * i

			y_train = np.concatenate((y_train,labels_train))
			y_test = np.concatenate((y_test, labels_test))

			i = i + 1

		# print(train_set.shape)
		# print(test_set.shape)
		# print(y_train.shape)
		# print(y_test.shape)

		return train_set[5:], test_set[5:], y_train[5:], y_test[5:]




if __name__ == '__main__':
	pre = Preprocess()

	pre.train_test_split()