def train_test_split(features_df, target_df, prediction_days):
	features_train = features_df[:len(features_df)-prediction_days]
	features_test = features_df[len(features_df)-prediction_days:]

	target_train = target_df[:len(target_df)-prediction_days]
	target_test = target_df[len(target_df)-prediction_days:]
	return features_train, features_test, target_train, target_test

def window_data(features_df, target_df, window_len=7):
	import numpy as np
	window_data = []
	for i in range(len(features_df) - window_len):
		tmp = features_df[i: (i + window_len)].copy()
		window_data.append(tmp.values)
		feature_data = np.array(window_data)
		target_data = target_df[window_len:].values
	return feature_data, target_data


