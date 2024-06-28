


import tensorflow as tf
import tensorflow_io as tfio
import librosa

Fernseher = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Fernseher'
Heizung = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Heizung'
Licht = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Licht'
Luftung = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Luftung'
Ofen = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Ofen'
Alarm = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Alarm'
Radio = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Radio'
Staubsauger = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Staubsauger'
an = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\an'
aus = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\aus'

Brotchen = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Brotchen'
Haus = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Haus'
kann = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\kann'
Leitung = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Leitung'
nicht = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\nicht'
offen = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\offen'
Schraube = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Schraube'
Spiegel = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\Spiegel'
warm = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\warm'
wunderbar = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\wunderbar'

other = r'C:\Users\azatv\Jupyter\JupyterProjects\ML_and_PC\mlpc_dataset_with just_words\other'

label_map = {
    'Fernseher': int(0),
    'Heizung': int(1),
    'Licht': int(2),
    'Luftung': int(3),
    'Ofen': int(4),
    'Alarm': int(5),
    'Radio': int(6),
    'Staubsauger': int(7),
    'an': int(8),
    'aus': int(9),
    'Brotchen': int(10),
    'Haus': int(11),
    'kann': int(12),
    'Leitung': int(13),
    'nicht': int(14),
    'offen': int(15),
    'Schraube': int(16),
    'Spiegel': int(17),
    'warm': int(18),
    'wunderbar': int(19),
    'other': int(20)
}

# Create datasets for each category
ferseher_tf = tf.data.Dataset.list_files(Fernseher + r'\*.wav')
heizung_tf = tf.data.Dataset.list_files(Heizung + r'\*.wav')
licht_tf = tf.data.Dataset.list_files(Licht + r'\*.wav')
luftung_tf = tf.data.Dataset.list_files(Luftung + r'\*.wav')
ofen_tf = tf.data.Dataset.list_files(Ofen + r'\*.wav')
alarm_tf = tf.data.Dataset.list_files(Alarm + r'\*.wav')
radio_tf = tf.data.Dataset.list_files(Radio + r'\*.wav')
staubsauger_tf = tf.data.Dataset.list_files(Staubsauger + r'\*.wav')
an_tf = tf.data.Dataset.list_files(an + r'\*.wav')
aus_tf = tf.data.Dataset.list_files(aus + r'\*.wav')

Brotchen_tf = tf.data.Dataset.list_files(Brotchen + r'\*.wav')
Haus_tf = tf.data.Dataset.list_files(Haus + r'\*.wav')
kann_tf = tf.data.Dataset.list_files(kann + r'\*.wav')
Leitung_tf = tf.data.Dataset.list_files(Leitung + r'\*.wav')
nicht_tf = tf.data.Dataset.list_files(nicht + r'\*.wav')
offen_tf = tf.data.Dataset.list_files(offen + r'\*.wav')
Schraube_tf = tf.data.Dataset.list_files(Schraube + r'\*.wav')
Spiegel_tf = tf.data.Dataset.list_files(Spiegel + r'\*.wav')
warm_tf = tf.data.Dataset.list_files(warm + r'\*.wav')
wunderbar_tf = tf.data.Dataset.list_files(wunderbar + r'\*.wav')

other_tf = tf.data.Dataset.list_files(other + r'\*.wav')

def load_wav_16k_mono(filename):
    filename = filename.numpy().decode('utf-8')
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return wav

# Determine the number of elements in each dataset
num_elements_ferseher = len(list(ferseher_tf.as_numpy_iterator()))
num_elements_heizung = len(list(heizung_tf.as_numpy_iterator()))
num_elements_licht = len(list(licht_tf.as_numpy_iterator()))
num_elements_luftung = len(list(luftung_tf.as_numpy_iterator()))
num_elements_ofen = len(list(ofen_tf.as_numpy_iterator()))
num_elements_alarm = len(list(alarm_tf.as_numpy_iterator()))
num_elements_radio = len(list(radio_tf.as_numpy_iterator()))
num_elements_staubsauger = len(list(staubsauger_tf.as_numpy_iterator()))
num_elements_an = len(list(an_tf.as_numpy_iterator()))
num_elements_aus = len(list(aus_tf.as_numpy_iterator()))

num_elements_Brotchen = len(list(Brotchen_tf.as_numpy_iterator()))
num_elements_Haus = len(list(Haus_tf.as_numpy_iterator()))
num_elements_kann = len(list(kann_tf.as_numpy_iterator()))
num_elements_Leitung = len(list(Leitung_tf.as_numpy_iterator()))
num_elements_nicht = len(list(nicht_tf.as_numpy_iterator()))
num_elements_offen = len(list(offen_tf.as_numpy_iterator()))
num_elements_Schraube = len(list(Schraube_tf.as_numpy_iterator()))
num_elements_Spiegel = len(list(Spiegel_tf.as_numpy_iterator()))
num_elements_warm = len(list(warm_tf.as_numpy_iterator()))
num_elements_wunderbar = len(list(wunderbar_tf.as_numpy_iterator()))

num_elements_other = len(list(other_tf.as_numpy_iterator()))


labels_ferseher = [label_map['Fernseher']] * num_elements_ferseher
labels_heizung = [label_map['Heizung']] * num_elements_heizung
labels_licht = [label_map['Licht']] * num_elements_licht
labels_luftung = [label_map['Luftung']] * num_elements_luftung
labels_ofen = [label_map['Ofen']] * num_elements_ofen
labels_alarm = [label_map['Alarm']] * num_elements_alarm
labels_radio = [label_map['Radio']] * num_elements_radio
labels_staubsauger = [label_map['Staubsauger']] * num_elements_staubsauger
labels_an = [label_map['an']] * num_elements_an
labels_aus = [label_map['aus']] * num_elements_aus

labels_Brotchen = [label_map['Brotchen']] * num_elements_Brotchen
labels_Haus = [label_map['Haus']] * num_elements_Haus
labels_kann = [label_map['kann']] * num_elements_kann
labels_Leitung = [label_map['Leitung']] * num_elements_Leitung
labels_nicht = [label_map['nicht']] * num_elements_nicht
labels_offen = [label_map['offen']] * num_elements_offen
labels_Schraube = [label_map['Schraube']] * num_elements_Schraube
labels_Spiegel = [label_map['Spiegel']] * num_elements_Spiegel
labels_warm = [label_map['warm']] * num_elements_warm
labels_wunderbar = [label_map['wunderbar']] * num_elements_wunderbar

labels_other = [label_map['other']] * num_elements_other


ferseher_tf = tf.data.Dataset.zip((ferseher_tf, tf.data.Dataset.from_tensor_slices(labels_ferseher)))
heizung_tf = tf.data.Dataset.zip((heizung_tf, tf.data.Dataset.from_tensor_slices(labels_heizung)))
licht_tf = tf.data.Dataset.zip((licht_tf, tf.data.Dataset.from_tensor_slices(labels_licht)))
luftung_tf = tf.data.Dataset.zip((luftung_tf, tf.data.Dataset.from_tensor_slices(labels_luftung)))
ofen_tf = tf.data.Dataset.zip((ofen_tf, tf.data.Dataset.from_tensor_slices(labels_ofen)))
alarm_tf = tf.data.Dataset.zip((alarm_tf, tf.data.Dataset.from_tensor_slices(labels_alarm)))
radio_tf = tf.data.Dataset.zip((radio_tf, tf.data.Dataset.from_tensor_slices(labels_radio)))
staubsauger_tf = tf.data.Dataset.zip((staubsauger_tf, tf.data.Dataset.from_tensor_slices(labels_staubsauger)))
an_tf = tf.data.Dataset.zip((an_tf, tf.data.Dataset.from_tensor_slices(labels_an)))
aus_tf = tf.data.Dataset.zip((aus_tf, tf.data.Dataset.from_tensor_slices(labels_aus)))

Brotchen_tf = tf.data.Dataset.zip((Brotchen_tf, tf.data.Dataset.from_tensor_slices(labels_Brotchen)))
Haus_tf = tf.data.Dataset.zip((Haus_tf, tf.data.Dataset.from_tensor_slices(labels_Haus)))
kann_tf = tf.data.Dataset.zip((kann_tf, tf.data.Dataset.from_tensor_slices(labels_kann)))
Leitung_tf = tf.data.Dataset.zip((Leitung_tf, tf.data.Dataset.from_tensor_slices(labels_Leitung)))
nicht_tf = tf.data.Dataset.zip((nicht_tf, tf.data.Dataset.from_tensor_slices(labels_nicht)))
offen_tf = tf.data.Dataset.zip((offen_tf, tf.data.Dataset.from_tensor_slices(labels_offen)))
Schraube_tf = tf.data.Dataset.zip((Schraube_tf, tf.data.Dataset.from_tensor_slices(labels_Schraube)))
Spiegel_tf = tf.data.Dataset.zip((Spiegel_tf, tf.data.Dataset.from_tensor_slices(labels_Spiegel)))
warm_tf = tf.data.Dataset.zip((warm_tf, tf.data.Dataset.from_tensor_slices(labels_warm)))
wunderbar_tf = tf.data.Dataset.zip((wunderbar_tf, tf.data.Dataset.from_tensor_slices(labels_wunderbar)))

other_tf = tf.data.Dataset.zip((other_tf, tf.data.Dataset.from_tensor_slices(labels_other)))


data = ferseher_tf.concatenate(heizung_tf).concatenate(licht_tf).concatenate(luftung_tf).concatenate(ofen_tf).concatenate(alarm_tf).concatenate(radio_tf).concatenate(staubsauger_tf).concatenate(an_tf).concatenate(aus_tf).concatenate(Brotchen_tf).concatenate(Haus_tf).concatenate(kann_tf).concatenate(Leitung_tf).concatenate(nicht_tf).concatenate(offen_tf).concatenate(Schraube_tf).concatenate(Spiegel_tf).concatenate(warm_tf).concatenate(wunderbar_tf).concatenate(other_tf)




