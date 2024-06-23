# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import scipy.io.wavfile as wavfile
# # # import seaborn as sns
# # # from sklearn.ensemble import RandomForestClassifier
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.metrics import accuracy_score

# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.decomposition import PCA


# # # dataset = r"C:\Users\azatv\Jupyter\JupyterProjects\ML and PC\development.csv"
# # # dataset1 = r"C:\Users\azatv\Jupyter\JupyterProjects\ML and PC\idx_to_feature_name.csv"
# # # timestamp_dataset = np.load(r"C:\Users\azatv\Jupyter\JupyterProjects\ML and PC\development.npy")
# # # development = pd.read_csv(dataset)
# # # idx_to_feature_name = pd.read_csv(dataset1)


# # # timestamp_dataset = np.swapaxes(timestamp_dataset, 2, 1)

# # # x, y, z = np.shape(timestamp_dataset) # shape is 45296 44 175
# # # print(x,y,z)
# # # timestamp_dataset_xyz = timestamp_dataset.reshape((x * y, z))

# # # np.shape(timestamp_dataset_xyz)

# # # df = pd.DataFrame(timestamp_dataset_xyz)
# # # df.columns = list(idx_to_feature_name['feature_name'])

# # # correlation_matrix = df.corr()

# # # plt.figure(figsize=(20, 15))
# # # sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".1f")
# # # plt.title('Feature Correlation Matrix')
# # # plt.show()

# # # Define the components of the CEN
# # conditions = {'r', 's', 't'}
# # events = {'a', 'b', 'c'}
# # inputs = {'a': {'r'}, 'b': {'s'}, 'c': {'t'}}  # Adjust as per the diagram
# # outputs = {'a': {'s'}, 'b': {'t'}, 'c': {'r'}}  # Adjust as per the diagram

# # # Define the network as a tuple
# # N = (conditions, inputs, events, outputs)

# # # Calculate the number of possible markings
# # number_of_markings = 2 ** len(conditions)

# # # Define a function to determine which events can fire given a marking
# # def events_that_can_fire(marking, inputs):
# #     return {event for event, needed_conditions in inputs.items() if needed_conditions.issubset(marking)}

# # # Define a function to get the new marking after an event fires
# # def fire_event(marking, event, inputs, outputs):
# #     if not inputs[event].issubset(marking):
# #         raise ValueError(f"Event {event} cannot fire because the conditions {inputs[event]} are not met.")
# #     # Remove the inputs and add the outputs
# #     new_marking = (marking - inputs[event]) | outputs[event]
# #     return new_marking

# # # Define a function to determine possible markings after events fire
# # def possible_markings_after_fire(marking, inputs, outputs):
# #     possible_markings = {}
# #     for event in events_that_can_fire(marking, inputs):
# #         new_marking = fire_event(marking, event, inputs, outputs)
# #         possible_markings[event] = new_marking
# #     return possible_markings

# # # Now we can calculate for specific markings
# # marking_r_s = {'r', 's'}
# # marking_t = {'t'}
# # marking_r = {'r'}

# # # Use the defined functions to calculate the answers for parts c to f
# # possible_events_r_s = events_that_can_fire(marking_r_s, inputs)
# # new_marking_b_fires = fire_event(marking_r_s, 'b', inputs, outputs)
# # possible_markings_t = possible_markings_after_fire(marking_t, inputs, outputs)
# # possible_markings_r = possible_markings_after_fire(marking_r, inputs, outputs)

# # # Display the results
# # print(f"Number of possible markings: {number_of_markings}")
# # print(f"Possible events from marking {marking_r_s}: {possible_events_r_s}")
# # print(f"New marking when 'b' fires from {marking_r_s}: {new_marking_b_fires}")
# # print(f"Possible events and markings from {marking_t}: {possible_markings_t}")
# # print(f"Possible events and markings from {marking_r}: {possible_markings_r}")

# # import librosa
# # import librosa.display
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import cv2
# # import pandas as pd

# # # Path to the WAV file
# # wav_path = "C:\\Users\\azatv\\Jupyter\\JupyterProjects\\ML and PC\\development_scenes\\wav\\scenes\\3_speech_true_Radio_an.wav"

# # # Load the audio file
# # y, sr = librosa.load(wav_path, sr=None)

# # # Compute the spectrogram
# # S = np.abs(librosa.stft(y))
# # #(1025, 867)
# # segment = S[671:724,:]
# # print(np.shape(segment))
# # # Convert to decibels
# # S_db = librosa.amplitude_to_db(S, ref=np.max)

# # # Normalize the spectrogram to the range 0-255
# # S_db_normalized = cv2.normalize(S_db, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# # # Convert to uint8
# # image = S_db_normalized.astype(np.uint8)

# # # Plot the spectrogram using matplotlib
# # plt.figure(figsize=(10, 4))
# # librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
# # plt.colorbar(format='%+2.0f dB')
# # plt.title('Spectrogram')
# # plt.tight_layout()
# # plt.savefig('spectrogram_image.png')  # Save the spectrogram as an image
# # plt.show()


# # import os
# # import pandas as pd
# # import numpy as np
# # import librosa
# # import cv2
# # from matplotlib import pyplot as plt
# # import sounddevice as sd

# # # Load CSV file
# # csv_file = r'C:\Users\azatv\Jupyter\JupyterProjects\ML and PC\development_scenes\development_scene_annotations.csv'
# # labels_df = pd.read_csv(csv_file)

# # # Directory paths
# # wav_dir = r'C:\Users\azatv\Jupyter\JupyterProjects\ML and PC\development_scenes\wav\scenes'
# # output_dir = r'C:\Users\azatv\Jupyter\JupyterProjects\ML and PC\development_scenes\tests'

# # # Ensure output directory exists
# # os.makedirs(output_dir, exist_ok=True)

# # # Function to process each entry
# # def process_entry(filename, command, start, end):
# #     wav_path = os.path.join(wav_dir, filename + '.wav')
    
# #     # Load the audio file
# #     y, sr = librosa.load(wav_path, sr=None)
    
# #     # Compute the spectrogram
# #     n_fft = 2048
# #     hop_length = 512
# #     S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
# #     S_db = librosa.amplitude_to_db(S, ref=np.max)
    
# #     # Normalize spectrogram for image conversion
# #     S_db_normalized = cv2.normalize(S_db, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
# #     image = S_db_normalized.astype(np.uint8)
    
# #     # Number of frames and duration
# #     num_frames = S.shape[1]
# #     duration = (num_frames * hop_length) / sr
    
# #     # Calculate middle point of the window
# #     middle_time = (start + end) / 2
# #     middle_frame = int((middle_time / duration) * num_frames)
    
# #     # Get window dimensions in frames
# #     start_frame = int((start / duration) * num_frames)
# #     end_frame = int((end / duration) * num_frames)
# #     window_width = end_frame - start_frame
    
# #     # Segment the audio and compute word boundaries
# #     transcript = command.split()
# #     num_words = len(transcript)
# #     word_boundaries = np.linspace(start_frame, end_frame, num_words + 1, dtype=int)
    
# #     # Save spectrogram image
# #     plt.imsave(os.path.join(output_dir, filename + '.png'), image, cmap='gray')
    
# #     # Write to text file
# #     txt_filename = os.path.join(output_dir, filename + '.txt')
# #     with open(txt_filename, 'a') as f:
# #         for i in range(len(word_boundaries) - 1):
# #             word_start = word_boundaries[i]
# #             word_end = word_boundaries[i + 1]
# #             word_middle = (word_start + word_end) // 2
            
# #             x = word_middle / num_frames
# #             y1 = 0.5  # Assuming the y-coordinate in the middle for simplicity
# #             width = (word_end - word_start) / num_frames
# #             height = 0.5  # Assuming a fixed height for simplicity
            
# #             f.write(f'{transcript[i]} {x:.6f} {y1:.6f} {width:.6f} {height:.6f}\n')
            
# #             # Convert frame indices to sample indices
# #             word_start_sample = word_start * hop_length
# #             word_end_sample = word_end * hop_length
# #             print(f"word start: {word_start_sample}")
# #             print(f"word end: {word_end_sample}")
            
# #             # Play each word segment
# #             segment = y[int(word_start_sample):int(word_end_sample)]
# #             print(f"Playing {transcript[i]} from sample {word_start_sample} to {word_end_sample} "
# #                   f"(time frame {word_start_sample / sr:.2f}s to {word_end_sample / sr:.2f}s)...")
# #             sd.play(segment, sr)
# #             sd.wait()
# #             input("Press Enter to continue...")

# # # Process each entry in the CSV file
# # for index, row in labels_df.iterrows():
# #     process_entry(row['filename'], row['command'], row['start'], row['end'])

# import os
# import random
# import librosa
# import numpy as np
# import soundfile as sf
# import cv2
# import matplotlib.pyplot as plt
# import sounddevice as sd

# # Define paths
# base_path = "C:\\Users\\azatv\\Jupyter\\JupyterProjects\\ML and PC\\mlpc_dataset_with just_words"
# output_path = "C:\\Users\\azatv\\Jupyter\\JupyterProjects\\ML and PC\\dataset_test"
# os.makedirs(output_path, exist_ok=True)
# os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
# os.makedirs(os.path.join(output_path, 'labels'), exist_ok=True)

# # Define directories
# word_groups = {
#     "group1": ["Alarm", "Fernseher", "Heizung", "Licht", "Luftung", "Ofen", "Radio", "Staubsauger"],
#     "group2": ["Brotchen", "Haus", "kann", "Leitung", "nicht", "offen", "Schraube", "Spiegel", "warm", "wunderbar"],
#     "actions": ["an", "aus"]
# }
# noise_dir = os.path.join(base_path, "other")

# # Helper function to load audio files
# def load_audio(file_path):
#     try:
#         audio, sr = librosa.load(file_path)
#         return audio, sr
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None, None

# # Helper function to add noise to audio
# def add_noise(noise_files, sr, duration):
#     noise_file = random.choice(noise_files)
#     print(f"Adding noise from file: {noise_file}")
#     noise_audio, sr = load_audio(noise_file)
#     if noise_audio is not None:
#         noise_audio = librosa.util.fix_length(noise_audio, size=int(duration * sr))
#         return noise_audio  # Adjust noise level
#     return np.zeros(int(duration * sr))  # Return silence if noise file fails

# # Helper function to save audio and labels
# def save_audio_and_labels(audio, sr, labels, file_name):
#     output_audio_path = os.path.join(output_path, 'images', file_name + ".wav")
#     output_label_path = os.path.join(output_path, 'labels', file_name + ".txt")
#     print(f"Saving audio to {output_audio_path}")
#     print(f"Saving labels to {output_label_path}")
#     sf.write(output_audio_path, audio, sr)
#     with open(output_label_path, 'w') as f:
#         f.write("\n".join(labels))

# # Helper function to convert audio to spectrogram and save
# def save_spectrogram(audio, sr, file_name, n_fft=2048, hop_length=512):
#     # Compute the spectrogram
#     S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
#     print(np.shape(S))
    
#     # Compute to dB
#     S_db = librosa.amplitude_to_db(S, ref=np.max)

#     # Normalize spectrogram
#     S_db_normalized = cv2.normalize(S_db, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#     image = S_db_normalized.astype(np.uint8)

#     # Save the image
#     output_image_path = os.path.join(output_path, 'images', file_name + ".png")
#     cv2.imwrite(output_image_path, image)
#     print(f"Spectrogram saved to {output_image_path}")

# # Get all noise files
# noise_files = [os.path.join(noise_dir, str(f).replace('._', '')) for f in os.listdir(noise_dir) if f.endswith('.wav')]
# print(f"Found noise files: {noise_files}")

# # Function to create random combinations and mix words with noise
# def create_combination(word_group, action_group, noise_files, sr, include_additional_command, n_fft=2048, hop_length=512):
#     selected_word = random.choice(word_group)
#     selected_action = random.choice(action_group)
#     combined_audio = np.array([])
#     labels = []
#     words_included = [selected_word, selected_action]

#     total_duration = 0
#     num_frames = 0

#     # Add random noise at the beginning
#     noise_duration = random.uniform(1, 3)
#     combined_audio = np.concatenate((combined_audio, add_noise(noise_files, sr, noise_duration)))
#     total_duration += noise_duration

#     # Function to calculate label parameters
#     def calculate_label_params(word_start, word_end, total_duration, n_fft, hop_length):
#         word_center = (word_start + word_end) / 2
#         x = word_center / total_duration
#         width = (word_end - word_start) / total_duration

#         num_frames = int(total_duration * sr / hop_length) + 1
#         y = 0.5  # Assuming centered vertically
#         height = 1  # Assuming fixed height for simplicity

#         return x, y, width, height, num_frames

#     # Add selected word
#     word_dir = os.path.join(base_path, selected_word)
#     word_files = [os.path.join(word_dir, str(f).replace('._', '')) for f in os.listdir(word_dir) if f.endswith('.wav')]
#     word_file = random.choice(word_files)
#     print(f"Adding word from file: {word_file}")
#     audio, _ = load_audio(word_file)
#     if audio is not None:
#         word_start = total_duration
#         combined_audio = np.concatenate((combined_audio, audio))
#         word_end = total_duration + len(audio) / sr
#         total_duration += len(audio) / sr

#         x, y, width, height, num_frames = calculate_label_params(word_start, word_end, total_duration, n_fft, hop_length)
#         labels.append(f"{selected_word} {x} {y} {width} {height}")

#     # Add action word
#     action_dir = os.path.join(base_path, selected_action)
#     action_files = [os.path.join(action_dir, str(f).replace('._', '')) for f in os.listdir(action_dir) if f.endswith('.wav')]
#     action_file = random.choice(action_files)
#     print(f"Adding action from file: {action_file}")
#     action_audio, _ = load_audio(action_file)
#     if action_audio is not None:
#         action_start = total_duration
#         combined_audio = np.concatenate((combined_audio, action_audio))
#         action_end = total_duration + len(action_audio) / sr
#         total_duration += len(action_audio) / sr

#         x, y, width, height, num_frames = calculate_label_params(action_start, action_end, total_duration, n_fft, hop_length)
#         labels.append(f"{selected_action} {x} {y} {width} {height}")

#     # Optionally add an additional command
#     if include_additional_command:
#         additional_word = random.choice(word_group)
#         additional_word_dir = os.path.join(base_path, additional_word)
#         additional_word_files = [os.path.join(additional_word_dir, str(f).replace('._', '')) for f in os.listdir(additional_word_dir) if f.endswith('.wav')]
#         additional_word_file = random.choice(additional_word_files)
#         print(f"Adding additional word from file: {additional_word_file}")
#         additional_audio, _ = load_audio(additional_word_file)
#         if additional_audio is not None:
#             additional_start = total_duration
#             combined_audio = np.concatenate((combined_audio, additional_audio))
#             additional_end = total_duration + len(additional_audio) / sr
#             total_duration += len(additional_audio) / sr

#             x, y, width, height, num_frames = calculate_label_params(additional_start, additional_end, total_duration, n_fft, hop_length)
#             labels.append(f"{additional_word} {x} {y} {width} {height}")
#             words_included.append(additional_word)

#     # Add random noise at the end
#     noise_duration = random.uniform(1, 3)
#     combined_audio = np.concatenate((combined_audio, add_noise(noise_files, sr, noise_duration)))
#     total_duration += noise_duration

#     # Ensure the final length is between 12 and 30 seconds
#     target_duration = random.uniform(12, 30)
#     if total_duration < target_duration:
#         additional_noise_duration = target_duration - total_duration
#         combined_audio = np.concatenate((combined_audio, add_noise(noise_files, sr, additional_noise_duration)))
#         total_duration += additional_noise_duration

#     # Trim the audio to the target duration if necessary
#     if total_duration > target_duration:
#         combined_audio = combined_audio[:int(target_duration * sr)]

#     return combined_audio, labels, total_duration, words_included, num_frames

# # Process each word directory
# for i in range(10):  # Generate 10 samples
#     # Select a random word group and action group
#     if random.random() < 0.5:
#         word_group = word_groups["group1"]
#     else:
#         word_group = word_groups["group2"]
#     action_group = word_groups["actions"]

#     # Determine if an additional command should be included
#     include_additional_command = random.random() < 0.3  # 10-30% probability
#     sr = 16000
#     combined_audio, labels, total_duration, words_included, num_frames = create_combination(word_group, action_group, noise_files, sr, include_additional_command)
    
#     if len(combined_audio) == 0:
#         print("Skipping sample due to invalid audio.")
#         continue  # Skip if no valid audio files were found

#     # Create the file name
#     file_name = f"{i}_{'_'.join(words_included)}"

#     # Save combined audio and labels
#     save_audio_and_labels(combined_audio, sr, labels, file_name)

#     # Save the spectrogram image
#     #save_spectrogram(combined_audio, sr, file_name, n_fft=2048, hop_length=512)

# # # Function to load and parse the labels
# # def load_labels(label_path):
# #     with open(label_path, 'r') as f:
# #         lines = f.readlines()
# #     labels = []
# #     for line in lines:
# #         parts = line.strip().split()
# #         if len(parts) == 5:
# #             label = {
# #                 'word': parts[0],
# #                 'x': float(parts[1]),
# #                 'y': float(parts[2]),
# #                 'width': float(parts[3]),
# #                 'height': float(parts[4])
# #             }
# #             labels.append(label)
# #     return labels

# # # Function to play specific word
# # def play_word(audio, sr, label, total_duration):
# #     start_time = (label['x'] - (label['width'] / 2)) * total_duration
# #     end_time = (label['x'] + (label['width'] / 2)) * total_duration
# #     start_sample = int(start_time * sr)
# #     end_sample = int(end_time * sr)
    
# #     print(f"Playing word '{label['word']}' from {start_time:.2f}s to {end_time:.2f}s ({start_sample} to {end_sample} samples)")
    
# #     word_audio = audio[start_sample:end_sample]
    
# #     # Ensure the audio segment has a minimum volume
# #     word_audio = word_audio / np.max(np.abs(word_audio))  # Normalize to -1 to 1
# #     word_audio = word_audio * 0.5  # Adjust volume if needed
    
# #     sd.play(word_audio, sr)
# #     sd.wait()

# # # List all generated files
# # wav_files = [f for f in os.listdir(os.path.join(output_path, 'images')) if f.endswith('.wav')]
# # txt_files = [f for f in os.listdir(os.path.join(output_path, 'labels')) if f.endswith('.txt')]
# # png_files = [f for f in os.listdir(os.path.join(output_path, 'images')) if f.endswith('.png')]

# # # Ensure there are files to process
# # if not wav_files or not txt_files or not png_files:
# #     print("No files found to process.")
# # else:
# #     # Select a file to test
# #     test_index = 1
# #     wav_file = os.path.join(output_path, 'images', wav_files[test_index])
# #     txt_file = os.path.join(output_path, 'labels', txt_files[test_index])
# #     png_file = os.path.join(output_path, 'images', png_files[test_index])

# #     print(f"Testing file: {wav_file}")
    
# #     # Load the audio file
# #     audio, sr = librosa.load(wav_file)

# #     # Load the labels
# #     labels = load_labels(txt_file)

# #     # Compute the total duration of the audio
# #     total_duration = len(audio) / sr

# #     # Display the spectrogram image
# #     plt.imshow(plt.imread(png_file))
# #     plt.title(f"Spectrogram: {wav_files[test_index]}")
# #     plt.show()

# #     # Play the specific words based on the labels
# #     for label in labels:
# #         print(f"Playing word: {label['word']}")
# #         play_word(audio, sr, label, total_duration)


# import os
# import random
# import librosa
# import numpy as np
# import soundfile as sf
# import cv2
# import matplotlib.pyplot as plt
# import sounddevice as sd

# # Define paths
# base_path = "C:\\Users\\azatv\\Jupyter\\JupyterProjects\\ML and PC\\mlpc_dataset_with just_words"
# output_path = "C:\\Users\\azatv\\Jupyter\\JupyterProjects\\ML and PC\\dataset_test"
# os.makedirs(output_path, exist_ok=True)
# os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
# os.makedirs(os.path.join(output_path, 'labels'), exist_ok=True)

# # Define directories
# word_groups = {
#     "group1": ["Alarm", "Fernseher", "Heizung", "Licht", "Luftung", "Ofen", "Radio", "Staubsauger"],
#     "group2": ["Brotchen", "Haus", "kann", "Leitung", "nicht", "offen", "Schraube", "Spiegel", "warm", "wunderbar"],
#     "actions": ["an", "aus"]
# }
# noise_dir = os.path.join(base_path, "other")

# # Helper function to load audio files
# def load_audio(file_path):
#     try:
#         audio, sr = librosa.load(file_path)
#         return audio, sr
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return None, None

# # Helper function to add noise to audio
# def add_noise(noise_files, sr, duration):
#     noise_file = random.choice(noise_files)
#     print(f"Adding noise from file: {noise_file}")
#     noise_audio, sr = load_audio(noise_file)
#     if noise_audio is not None:
#         noise_audio = librosa.util.fix_length(noise_audio, size=int(duration * sr))
#         return noise_audio * 0.05  # Adjust noise level
#     return np.zeros(int(duration * sr))  # Return silence if noise file fails

# # Helper function to save audio and labels
# def save_audio_and_labels(audio, sr, labels, file_name):
#     output_audio_path = os.path.join(output_path, 'images', file_name + ".wav")
#     output_label_path = os.path.join(output_path, 'labels', file_name + ".txt")
#     print(f"Saving audio to {output_audio_path}")
#     print(f"Saving labels to {output_label_path}")
#     #sf.write(output_audio_path, audio, sr)
#     #with open(output_label_path, 'w') as f:
#         #f.write("\n".join(labels))

# # Helper function to convert audio to spectrogram and save
# def save_spectrogram(audio, sr, file_name, n_fft=2048, hop_length=512):
#     # Compute the spectrogram
#     S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
#     print(np.shape(S))
    
#     # Compute to dB
#     S_db = librosa.amplitude_to_db(S, ref=np.max)

#     # Normalize spectrogram
#     S_db_normalized = cv2.normalize(S_db, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#     image = S_db_normalized.astype(np.uint8)

#     # Save the image
#     output_image_path = os.path.join(output_path, 'images', file_name + ".png")
#     #cv2.imwrite(output_image_path, image)
#     print(f"Spectrogram saved to {output_image_path}")

# # Get all noise files
# noise_files = [os.path.join(noise_dir, str(f).replace('._', '')) for f in os.listdir(noise_dir) if f.endswith('.wav')]
# print(f"Found noise files: {noise_files}")

# # Function to create random combinations and mix words with noise
# def create_combination(word_group, action_group, noise_files, sr, include_additional_command, n_fft=2048, hop_length=512):
#     selected_word = random.choice(word_group)
#     selected_action = random.choice(action_group)
#     combined_audio = np.array([])
#     labels = []
#     words_included = [selected_word, selected_action]

#     total_duration = 0
#     num_frames = 0

#     # Add random noise at the beginning
#     noise_duration = random.uniform(1.1, 3)
#     combined_audio = np.concatenate((combined_audio, add_noise(noise_files, sr, noise_duration)))
#     total_duration += noise_duration

#     # Function to calculate label parameters
#     def calculate_label_params(word_start, word_end, total_duration, n_fft, hop_length):
#         print(word_start)
#         print(word_end)
#         word_center = np.mean([word_start,word_end])
#         print(word_center)
#         x = word_center / total_duration
#         width = (word_end - word_start) / total_duration

#         num_frames = int(total_duration * sr / hop_length) + 1
#         y = 0.5  # Assuming centered vertically
#         height = 1  # Assuming fixed height for simplicity

#         return x, y, width, height, num_frames

#     # Add selected word
#     word_dir = os.path.join(base_path, selected_word)
#     word_files = [os.path.join(word_dir, str(f).replace('._', '')) for f in os.listdir(word_dir) if f.endswith('.wav')]
#     word_file = random.choice(word_files)
#     print(f"Adding word from file: {word_file}")
#     audio, st = load_audio(word_file)
#     if audio is not None:
#         word_start = total_duration
#         combined_audio = np.concatenate((combined_audio, audio))
#         word_end = total_duration + len(audio) / sr
#         total_duration += len(audio) / sr

#         x, y, width, height, num_frames = calculate_label_params(word_start, word_end, total_duration, n_fft, hop_length)
#         labels.append(f"{selected_word} {x} {y} {width} {height}")

#     # Add action word
#     action_dir = os.path.join(base_path, selected_action)
#     action_files = [os.path.join(action_dir, str(f).replace('._', '')) for f in os.listdir(action_dir) if f.endswith('.wav')]
#     action_file = random.choice(action_files)
#     print(f"Adding action from file: {action_file}")
#     action_audio, sr = load_audio(action_file)
#     if action_audio is not None:
#         action_start = total_duration
#         combined_audio = np.concatenate((combined_audio, action_audio))
#         action_end = total_duration + len(action_audio) / sr
#         total_duration += len(action_audio) / sr

#         x, y, width, height, num_frames = calculate_label_params(action_start, action_end, total_duration, n_fft, hop_length)
#         labels.append(f"{selected_action} {x} {y} {width} {height}")

#     # Optionally add an additional command
#     if include_additional_command:
#         additional_word = random.choice(word_group)
#         additional_word_dir = os.path.join(base_path, additional_word)
#         additional_word_files = [os.path.join(additional_word_dir, str(f).replace('._', '')) for f in os.listdir(additional_word_dir) if f.endswith('.wav')]
#         additional_word_file = random.choice(additional_word_files)
#         print(f"Adding additional word from file: {additional_word_file}")
#         additional_audio, sr = load_audio(additional_word_file)
#         if additional_audio is not None:
#             additional_start = total_duration
#             combined_audio = np.concatenate((combined_audio, additional_audio))
#             additional_end = total_duration + len(additional_audio) / sr
#             total_duration += len(additional_audio) / sr

#             x, y, width, height, num_frames = calculate_label_params(additional_start, additional_end, total_duration, n_fft, hop_length)
#             labels.append(f"{additional_word} {x} {y} {width} {height}")
#             words_included.append(additional_word)

#     # Add random noise at the end
#     noise_duration = random.uniform(1, 3)
#     combined_audio = np.concatenate((combined_audio, add_noise(noise_files, sr, noise_duration)))
#     total_duration += noise_duration

#     # Ensure the final length is between 12 and 30 seconds
#     target_duration = random.uniform(12, 30)
#     if total_duration < target_duration:
#         additional_noise_duration = target_duration - total_duration
#         combined_audio = np.concatenate((combined_audio, add_noise(noise_files, sr, additional_noise_duration)))
#         total_duration += additional_noise_duration

#     # Trim the audio to the target duration if necessary
#     if total_duration > target_duration:
#         combined_audio = combined_audio[:int(target_duration * sr)]

#     return combined_audio, labels, total_duration, words_included, num_frames

# # Process each word directory
# for i in range(1):  # Generate 10 samples
#     # Select a random word group and action group
#     if random.random() < 0.5:
#         word_group = word_groups["group1"]
#     else:
#         word_group = word_groups["group2"]
#     action_group = word_groups["actions"]

#     # Determine if an additional command should be included
#     include_additional_command = random.random() < 0.3  # 10-30% probability

#     combined_audio, labels, total_duration, words_included, num_frames = create_combination(word_group, action_group, noise_files, sr, include_additional_command)
    
#     if len(combined_audio) == 0:
#         print("Skipping sample due to invalid audio.")
#         continue  # Skip if no valid audio files were found

#     # Create the file name
#     file_name = f"{i}_{'_'.join(words_included)}"

#     # Save combined audio and labels
#     save_audio_and_labels(combined_audio, sr, labels, file_name)

#     # Save the spectrogram image
#     save_spectrogram(combined_audio, sr, file_name, n_fft=2048, hop_length=512)

import torch
from pathlib import Path
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.plots import plot_one_box

# Load model
weights = 'runs/train/custom_yolov5_results5/weights/best.pt'
model = attempt_load(weights, map_location='cuda' if torch.cuda.is_available() else 'cpu')  # load FP32 model

# Load images
source = 'path/to/your/image_or_folder'
dataset = LoadImages(source, img_size=640)

# Inference
for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(model.device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        p, s, im0 = path, '', im0s

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=(255, 0, 0), line_thickness=2)

        # Save results
        save_path = str(Path('runs/detect/exp') / Path(p).name)
        cv2.imwrite(save_path, im0)
