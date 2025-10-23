# 🎵 AI Music Generator using LSTM

This project generates **music sequences** using a **Recurrent Neural Network (LSTM)** trained on MIDI files.

## 🧠 Features
- Reads and processes MIDI files using `music21`
- Trains an LSTM neural network on musical note sequences
- Generates new, original note sequences
- Exports generated music as a `.mid` file

## 🛠️ Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
# ▶️ How to Run

Place your MIDI files inside the midi_songs/ folder.

Run the project:

python music_generator.py


After training, your generated music will be saved as output.mid.

# 🎧 Output

You can open the generated MIDI file in any DAW (like FL Studio, GarageBand, or LMMS).