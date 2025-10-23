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
    # Check if notes are already cached
    if os.path.exists('data/notes.pkl'):
        print("📦 Loading cached notes...")
        with open('data/notes.pkl', 'rb') as f:
            return pickle.load(f)
    
    notes = []
    midi_files = glob.glob("midi_songs/*.mid")
    
    if not midi_files:
        raise FileNotFoundError("⚠️ No MIDI files found in 'midi_songs/' folder. Please add .mid files to train the model.")
    
    print(f"🎵 Processing {len(midi_files)} MIDI files...")
    for file in midi_files:
        try:
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
                    # Use pitches instead of normalOrder for proper note representation
                    notes.append('.'.join(str(n) for n in element.pitches))
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")
            continue
    
    if not notes:
        raise ValueError("⚠️ No notes extracted from MIDI files. Check if files are valid.")

    print(f"✅ Extracted {len(notes)} notes from MIDI files")
    with open('data/notes.pkl', 'wb') as f:
        pickle.dump(notes, f)
    return notes

# 🎼 Step 2: Prepare sequences for training
def prepare_sequences(notes, n_vocab):
    sequence_length = 100
    
    if len(notes) < sequence_length + 1:
        raise ValueError(f"⚠️ Not enough notes. Need at least {sequence_length + 1} notes, but got {len(notes)}. Add more MIDI files.")
    
    pitchnames = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    
    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[ch] for ch in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    print(f"📊 Created {n_patterns} training patterns")
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)
    network_output = tf.keras.utils.to_categorical(network_output)

    return network_input, network_output

# 🎹 Step 3: Build the LSTM model
def create_model(network_input, n_vocab):
    print("🏗️ Building LSTM model...")
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
    print("✅ Model built successfully")
    return model

# 🎧 Step 4: Generate music from trained model
def generate_notes(model, network_input, pitchnames, n_vocab):
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    prediction_output = []

    print("🎼 Generating music...")
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        # Ensure index is within bounds
        index = min(index, len(pitchnames) - 1)
        result = pitchnames[index]
        prediction_output.append(result)

        pattern = np.append(pattern, index / float(n_vocab))
        pattern = pattern[1:len(pattern)]

    print(f"✅ Generated {len(prediction_output)} notes")
    return prediction_output

# 🎶 Step 5: Convert generated notes into a MIDI file
def create_midi(prediction_output, filename="output.mid"):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            # Handle chord notation
            try:
                notes_in_chord = [note.Note(int(n)) for n in pattern.split('.')]
            except ValueError:
                # If not all integers, treat as pitch names
                notes_in_chord = [note.Note(n) for n in pattern.split('.')]
            
            for n in notes_in_chord:
                n.storedInstrument = instrument.Piano()
            chord_notes = chord.Chord(notes_in_chord)
            chord_notes.offset = offset
            output_notes.append(chord_notes)
        else:
            # Single note
            single_note = note.Note(pattern)
            single_note.offset = offset
            single_note.storedInstrument = instrument.Piano()
            output_notes.append(single_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)
    print(f"🎵 MIDI file saved as '{filename}'")

# 🚀 Step 6: Train and generate
def main():
    print("🎵 AI Music Generator Starting...")
    print("=" * 50)
    
    try:
        notes = get_notes()
        n_vocab = len(set(notes))
        print(f"🎼 Vocabulary size: {n_vocab} unique notes")
        
        network_input, network_output = prepare_sequences(notes, n_vocab)
        model = create_model(network_input, n_vocab)
        
        print("🏋️ Training model (this may take a while)...")
        model.fit(network_input, network_output, epochs=20, batch_size=64, verbose=1)
        
        print("\n🎹 Training complete! Generating music...")
        pitchnames = sorted(set(notes))
        prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
        create_midi(prediction_output)
        
        print("=" * 50)
        print("✅ Music generation complete! Check 'output.mid'")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("midi_songs", exist_ok=True)
    main()
