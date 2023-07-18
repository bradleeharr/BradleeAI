**Chess AI based on my Lichess game database.** Final Project for ECE 5973 Artificial Neural Networks [WIP]

# 1 | Introduction
Computer chess engines are computer systems designed to generate solutions for the problem of 
playing a game of chess. The most powerful engines have already achieved powerful performance using 
human-like learning through neural networks and reinforcement learning. However, while chess engines are capable of making the best moves, many chess engines will suggest moves that 
a human would never play because of the computer’s ability to calculate with precision and accuracy. This may make it more of a challenge for humans to learn from the results of a chess engine when 
using it to train or better understand the game of chess. 

One example of an existing attempt to create a specifically human-like chess AI is [Maia Chess](https://maiachess.com/) [1] [2], an engine trained on 12 million human games to 
perform more human-like movement prediction.

The goal of my project was to explore using deep learning approaches to capture personal human style 
from several hundreds of chess games played by an individual person. One differing approach I used in comparison to Maia was in using a much smaller set of personal games as well as including shorter-time games in my data. This was necessary because of the lack of games I had in my personal database, but it makes an important distinction, considering the problem of capturing human
decision-making style in shorter time frames.

# 2 | Data
The first step to addressing the problem was to acquire individual game datasets. Many databases of
thousands to millions of games are available, but for my project I wanted to try to use personal games. 
This both keeps the results highly individual and limits the dataset to a small amount (~2,000 games), 
capable for processing and training on only an AMD Ryzen CPU with no GPU acceleration.
Other attempts to create AI to predict human move predictions in chess and other games typically filter 
their data to remove very short time format chess games. For example, in Maia, Hyper-Bullet (30 
seconds or less per player) and Bullet (1-2 minutes per player) format games were filtered out [1], and in 
the training of the original AlphaGo model, only professional games from master-level players were used 
[3]. To use the most data available to me, I decided to include all my personal games, which includes 
1,633 Bullet games, 165 Blitz (3-8 minutes per player) games, and 244 Rapid (10-15 minutes per player) 
games. 

Heavily including many Bullet games applies more focus on the impulsive reactions and 
emotions that arise in low-time scenarios and can be useful to capture and model human behavior in 
short-time scenarios. The diversity of including the different time formats also has a potential benefit 
that helps focus on capturing the full scope of play. Given the highly complex nature of predicting chess 
moves in a time-sensitive context, the data was quite limited. Nevertheless, I strived to obtain 
meaningful results by trying different architectures.

From every game, each board position and move choice made by the specified player was converted 
data inputs for each model. The data was split into train, validation, and test sets with a 75-15-15 split.

# 3 | Methods
**3.0 Board Features**:

Each position of every game was transformed into an 8x8x12 map. The 8x8 area corresponds to the 
chess board itself, and each channel of area 8x8 corresponds to one piece. Since there are 12 different 
pieces, 12 channels are needed. A 1 is placed in a square if a piece is present and a 0 if a piece is not 
present

**3.1 Move Features**:

Each move of every game was converted from the Universal Chess Interface format to a number based 
on the square that the move is from (1-64) and the square that the move is going to (1-64). This results 
in 64 x 64 combinations, or 4096 classes.

**3.2 Random Valid Move Model**:

For a baseline metric to compare results for move prediction accuracy, this model selects a random 
move for each position. The random valid move choices resulted in an accuracy of 3.1-3.6%.

# 4 | Base Convolutional Model:

![image](https://github.com/bradleeharr/BradleeAI/assets/56418392/ec95dcc9-ee64-4d30-9167-0b18f78e52ca)

The base convolutional model uses a convolutional neural network with a variable number of layers.

* Each convolutional layer has the same number of input and output channels and uses a 3x3 kernel with padding 1 to 
maintain the same output shape. 

* For an 8x8x12 single position input, the number of channels will be 12. 

* The convolutional network output is flattened and followed by two fully connected layers with the 
output layer being a 4096-long channel. 

* The model is trained and optimized by comparing the cross-entropy loss between the output classification and the real classification of the move. 

* The model used 7 convolutional layers with kernel size 3 and padding 1. The number of previous input 
positions varied from 1 to 5. Each convolutional layer used had the same amount of input channels as 
output channels and was followed by a batch normalization step and a ReLU activation function.

I applied the convolutional model to the full data set of games, ran a Bayesian hyperparameter sweep using 
Weights and Biases with parameters: learning rate, steplr step size, steplr gamma, momentum, and weight decay, number of previous input positions, and linear hidden layer size.

The best model reached a training loss of 0.9274, validation loss of 4.616, and a test loss of 4.696. 

Running move prediction on this set resulted in a move prediction accuracy of 14.24%, which indicates that a decent improvement has been made from a random choice baseline.

However, this is still a fairly low move prediction accuracy. Further improvements can be made.

One change that I tested was changing the model to a residual convolutional network with squeeze-and-excitation blocks, similar to the model in the Maia [1] paper.

# 5 | Residual Model
![image](https://github.com/bradleeharr/BradleeAI/assets/56418392/81102fdc-193e-4ccc-a161-fffa3956efb1)

After training many of the residual models with varying hyperparameters, running move prediction on the best test set resulted in a lower move prediction accuracy of 10.21%. Though 
it is lower than the convolutional network, this is still better the random move as a baseline. Due to 
the increased complexity of this network and the lower prediction accuracy, it seems to be more 
difficult to train with the smaller dataset as compared to the standard convolutional model.

# References
[1] R. McIlroy-Young, S. Sen, J. Kleinberg, and A. Anderson, ‘Aligning Superhuman AI with Human 
Behavior’, in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge 
Discovery & Data Mining, 2020.

[2] R. McIlroy-Young, R. Wang, S. Sen, J. Kleinberg, and A. Anderson, ‘Learning Models of Individual 
Behavior in Chess’, in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery
and Data Mining, 2022

[3] D. Silver et al., ‘Mastering the game of Go with deep neural networks and tree search’, Nature, vol. 
529, pp. 484–489, 01 2016
