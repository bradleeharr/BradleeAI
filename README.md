**Chess AI based on my Lichess game database.** 
# 1 | Introduction
Inspired by [Maia Chess](https://maiachess.com/). Objective to use deep learning to mimic personal chess styles, emphasizing capturing human decision-making in rapid games.

# 2 | Data
Focused personal game datasets to ensure unique results. The dataset (~2,000 games) is optimized for an AMD Ryzen CPU without GPU acceleration. 
It comprises 1,633 Bullet, 165 Blitz, and 244 Rapid games. Emphasizing Bullet games captures impulsive decisions, providing a comprehensive play style view. All games were split into train, validation, and test sets (75-15-15).

# 3 | Methods
**3.0 Board Features**:

 8x8x12 map representation, where each 8x8 channel denotes a piece (12 total). A '1' indicates the presence of a piece, and '0' its absence
**3.1 Move Features**:

 Moves were translated from the Universal Chess Interface to a numerical system, leading to 4096 potential classes.

**3.2 Random Valid Move Model**:

A baseline model, achieving 3.1-3.6% accuracy through random move selection.
# 4 | Base Convolutional Model:

![image](https://github.com/bradleeharr/BradleeAI/assets/56418392/ec95dcc9-ee64-4d30-9167-0b18f78e52ca)

The base convolutional model uses a convolutional neural network with a variable number of layers.

* Utilizes a convolutional neural network with variable layer counts.
* Each layer maintains consistent input-output channels with a 3x3 kernel and padding.
* Output is flattened and connected to two dense layers, culminating in a 4096-long channel.
* Model trained by comparing cross-entropy loss between output and move classifications.

The best model had losses of 0.9274 (training), 4.616 (validation), and 4.696 (test). This led to a move prediction accuracy of 14.24%.

# 5 | Residual Model
![image](https://github.com/bradleeharr/BradleeAI/assets/56418392/81102fdc-193e-4ccc-a161-fffa3956efb1)

After hyperparameter adjustments, this model had a top move prediction accuracy of 10.21%. Though inferior to the convolutional model, it outperforms the random baseline. Its intricate structure, combined with a smaller dataset, poses training challenges.

