# This is a language AI model
 
Feel free to use this, as this is basically following this [video](https://www.youtube.com/watch?v=kCc8FmEb1nY). 
I express my greatest gratitude to Andrej.

However, I have added plenty of my own notes in there, hopefully this helps you to understand the video and code better.

## What is the difference between bigram.py and v2.py?
The bigram.py file contains the simple bigram language model with some more basic notes,
while v2.py contains the full code and more advanced notes.

## Play around with the hyperparams too! 
- normally the n_embd should be 384 
- the n_layer should be 6 
- I have a bad gpu let me be

### Prerequisites
* Python 3.9
* Pytorch

### Installation
1. Clone the repository
```sh
git clone https://github.com/Elbert-Ainstein/nanoGPT.git
```
2. Change directory into the folder
```sh
cd nanoGPT
```
3. Run the program
* For Windows:
```sh 
python v2.py
```
* For Mac:
```sh
python3 v2.py
```