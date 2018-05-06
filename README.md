# Guitar Chord Recognizer
Feedforward neural network for live guitar chord recognition. Supported chords include A, A7, Am, B, Bm, C, C7, Cm, D, D7, Dm, E, E7, Em, F, Fm, G, G7, Gm.

# Dependencies
Make sure you have Python 3 installed, as well as the following libraries:
- [TensorFlow](https://www.tensorflow.org/install/)
- [Numpy](http://www.numpy.org/)
- [Sklearn](http://scikit-learn.org/stable/)
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
  * Note: PyAudio requires the prerequisite portaudio library to be installed. On Mac, this can be installed using homebrew.

# Usage
Simply clone this repository, and run live_demo.py.
```
git clone https://github.com/connormcl/chord_recognizer.git
cd chord_recognizer
python3 live_demo.py
```
# Data
Anyone at Yale can download the data used to train this neural network [here](https://yale.box.com/s/t1dqx6aumsejs171gme4sr56085p5q94). To generate additional WAV files, a utility script has been provided to help.

In generate_wav_files.py, simply modify the call to generate_wavs() according to which files you would like to generate. For example, generate_wavs('e7', 1,100) would record and create 100 wav files named e7_1.wav through e7_100.wav in chords/e7. Note that if you are adding a new chord to the database, you must create the corresponding directory within the chords directory. See generate_wav_files.py for more information.

# References
- HPCP.py modified from [this repository](https://github.com/jvbalen/hpcp_demo)
- Some chord data was provided by a link found in [this paper](http://jim.afim-asso.org/jim12/pdf/jim2012_08_p_osmalskyj.pdf)
