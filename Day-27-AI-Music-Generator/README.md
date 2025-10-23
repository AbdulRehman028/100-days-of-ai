# 🎵 AI Music Generator using LSTM

This project generates **music sequences** using a **Recurrent Neural Network (LSTM)** trained on MIDI files.

## 🧠 Features
- Reads and processes MIDI files using `music21`
- Trains an LSTM neural network on musical note sequences
- Generates new, original note sequences
- Exports generated music as a `.mid` file
- Caches extracted notes for faster subsequent runs

## 🛠️ Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Setup
1. Create a `midi_songs` folder in the project directory
2. Add at least 5-10 MIDI files (.mid) to train the model effectively
3. The script will automatically create a `data` folder for caching

## ▶️ How to Run

1. Place your MIDI files inside the `midi_songs/` folder.

2. Run the project:
   ```bash
   python music_generator.py
   ```

3. After training, your generated music will be saved as `output.mid`.

## 🎧 Output

You can open the generated MIDI file in any DAW (like FL Studio, GarageBand, or LMMS) or MIDI player.

## 🐛 Troubleshooting

- **No MIDI files found**: Make sure you have `.mid` files in the `midi_songs/` folder
- **Not enough notes**: Add more MIDI files (need at least 101 notes total)
- **Training too slow**: Reduce epochs in `main()` or use a GPU-enabled TensorFlow installation

## 📝 Notes

- First run will take longer as it processes MIDI files
- Subsequent runs will be faster using cached notes from `data/notes.pkl`
- Training time depends on dataset size and hardware (typically 10-30 minutes on CPU)