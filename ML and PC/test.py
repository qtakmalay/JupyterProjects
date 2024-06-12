# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import scipy.io.wavfile as wavfile
# # import seaborn as sns
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score

# # from sklearn.preprocessing import StandardScaler
# # from sklearn.decomposition import PCA


# # dataset = r"C:\Users\azatv\Jupyter\JupyterProjects\ML and PC\development.csv"
# # dataset1 = r"C:\Users\azatv\Jupyter\JupyterProjects\ML and PC\idx_to_feature_name.csv"
# # timestamp_dataset = np.load(r"C:\Users\azatv\Jupyter\JupyterProjects\ML and PC\development.npy")
# # development = pd.read_csv(dataset)
# # idx_to_feature_name = pd.read_csv(dataset1)


# # timestamp_dataset = np.swapaxes(timestamp_dataset, 2, 1)

# # x, y, z = np.shape(timestamp_dataset) # shape is 45296 44 175
# # print(x,y,z)
# # timestamp_dataset_xyz = timestamp_dataset.reshape((x * y, z))

# # np.shape(timestamp_dataset_xyz)

# # df = pd.DataFrame(timestamp_dataset_xyz)
# # df.columns = list(idx_to_feature_name['feature_name'])

# # correlation_matrix = df.corr()

# # plt.figure(figsize=(20, 15))
# # sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".1f")
# # plt.title('Feature Correlation Matrix')
# # plt.show()

# # Define the components of the CEN
# conditions = {'r', 's', 't'}
# events = {'a', 'b', 'c'}
# inputs = {'a': {'r'}, 'b': {'s'}, 'c': {'t'}}  # Adjust as per the diagram
# outputs = {'a': {'s'}, 'b': {'t'}, 'c': {'r'}}  # Adjust as per the diagram

# # Define the network as a tuple
# N = (conditions, inputs, events, outputs)

# # Calculate the number of possible markings
# number_of_markings = 2 ** len(conditions)

# # Define a function to determine which events can fire given a marking
# def events_that_can_fire(marking, inputs):
#     return {event for event, needed_conditions in inputs.items() if needed_conditions.issubset(marking)}

# # Define a function to get the new marking after an event fires
# def fire_event(marking, event, inputs, outputs):
#     if not inputs[event].issubset(marking):
#         raise ValueError(f"Event {event} cannot fire because the conditions {inputs[event]} are not met.")
#     # Remove the inputs and add the outputs
#     new_marking = (marking - inputs[event]) | outputs[event]
#     return new_marking

# # Define a function to determine possible markings after events fire
# def possible_markings_after_fire(marking, inputs, outputs):
#     possible_markings = {}
#     for event in events_that_can_fire(marking, inputs):
#         new_marking = fire_event(marking, event, inputs, outputs)
#         possible_markings[event] = new_marking
#     return possible_markings

# # Now we can calculate for specific markings
# marking_r_s = {'r', 's'}
# marking_t = {'t'}
# marking_r = {'r'}

# # Use the defined functions to calculate the answers for parts c to f
# possible_events_r_s = events_that_can_fire(marking_r_s, inputs)
# new_marking_b_fires = fire_event(marking_r_s, 'b', inputs, outputs)
# possible_markings_t = possible_markings_after_fire(marking_t, inputs, outputs)
# possible_markings_r = possible_markings_after_fire(marking_r, inputs, outputs)

# # Display the results
# print(f"Number of possible markings: {number_of_markings}")
# print(f"Possible events from marking {marking_r_s}: {possible_events_r_s}")
# print(f"New marking when 'b' fires from {marking_r_s}: {new_marking_b_fires}")
# print(f"Possible events and markings from {marking_t}: {possible_markings_t}")
# print(f"Possible events and markings from {marking_r}: {possible_markings_r}")

import numpy as np
file_path = r"C:\Users\azatv\Jupyter\JupyterProjects\ML and PC\development_scenes\development_scenes\3_speech_true_Radio_an.npy"
file_np = np.load(file_path)
print(np.shape(file_np))