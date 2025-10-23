import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from music21 import converter, instrument, note, chord, stream
import glob
import pickle
import os

# 🎶 Step 1: Load MIDI files and extract notes
def get_notes():
    notes = []
    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes.pkl', 'wb') as f:
        pickle.dump(notes, f)
    return notes

# 🎼 Step 2: Prepare sequences for training
def prepare_sequences(notes, n_vocab):
    pitchnames = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    sequence_length = 100
    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[ch] for ch in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)
    network_output = tf.keras.utils.to_categorical(network_output)

    return network_input, network_output

# 🎹 Step 3: Build the LSTM model
def create_model(network_input, n_vocab):
    model = Sequential([
        LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(512, return_sequences=True),
        Dropout(0.3),
        LSTM(512),
        Dense(256),
        Dropout(0.3),
        Dense(n_vocab, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

# 🎧 Step 4: Generate music from trained model
def generate_notes(model, network_input, pitchnames, n_vocab):
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    prediction_output = []

    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = pitchnames[index]
        prediction_output.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

# 🎶 Step 5: Convert generated notes into a MIDI file
def create_midi(prediction_output, filename="output.mid"):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = [note.Note(int(n)) for n in pattern.split('.')]
            for n in notes_in_chord:
                n.storedInstrument = instrument.Piano()
            chord_notes = chord.Chord(notes_in_chord)
            chord_notes.offset = offset
            output_notes.append(chord_notes)
        else:
            single_note = note.Note(pattern)
            single_note.offset = offset
            single_note.storedInstrument = instrument.Piano()
            output_notes.append(single_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)

# 🚀 Step 6: Train and generate
def main():
    notes = get_notes()
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_model(network_input, n_vocab)
    model.fit(network_input, network_output, epochs=20, batch_size=64)
    pitchnames = sorted(set(notes))
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("midi_songs", exist_ok=True)
    main()
