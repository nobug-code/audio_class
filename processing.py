import librosa
import argparse
import pandas as pd
import numpy as np
import glob
import torch
import torchaudio
import torchvision
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--sampling_rate", default=44100, type=int)

def extract_spectrogram(name, values, clip, target):

	num_channels = 3
	window_sizes = [25, 50, 100]
	hop_sizes = [10, 25, 50]
	centre_sec = 2.5
	specs = []

	for i in range(num_channels):
		window_length = int(round(window_sizes[i] * args.sampling_rate / 1000))
		hop_length = int(round(hop_sizes[i] * args.sampling_rate / 1000))

		clip = torch.Tensor(clip)

		#  사람이 들을 수 있는 경계값으로 scale 해주는 작업.
		spec = torchaudio.transforms.MelSpectrogram(sample_rate=args.sampling_rate, n_fft=4410,
													win_length=window_length, hop_length=hop_length, n_mels=128)(clip)
		eps = 1e-6
		spec = spec.numpy()
		spec = np.log(spec + eps)
		spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
		specs.append(spec)

	new_entry = {}
	audio_array = np.array(specs)
	np_name = '/Users/selectstar_namkill/PycharmProjects/Audio_classification/data/audios/save_np/{}.npy'.format(name)
	np.save(np_name, audio_array)
	print("np_name", np_name, audio_array.shape)
	new_entry["audio"] = clip.numpy()
	new_entry["values"] = np_name
	new_entry["target"] = target
	#print(new_entry)
	values.append(new_entry)


	return values

def extract_features():

	values = []

	paths = glob.glob("./data/audios/*")

	for path in paths:

		label = path.split('/')[-1]

		if label == 'bellypain':
			target = 0
		elif label == 'burping':
			target = 1
		elif label == 'save_np':
			continue
		else:
			print("Not find label ")
			exit()

		file_list = glob.glob(path + '/*.wav')

		for file in file_list:
			# librosa.load return value -1 ~ 1 로 정규화 돼서 값이 나온다.
			clip, sr = librosa.load(file, sr=args.sampling_rate)
			extract_spectrogram(file.split('/')[-1], values, clip, target)

	df = pd.DataFrame(values)

	df.to_csv("/Users/selectstar_namkill/PycharmProjects/Audio_classification/data/files/total_audio_list.csv")
	print("end processing")

if __name__=="__main__":

	args = parser.parse_args()
	extract_features()