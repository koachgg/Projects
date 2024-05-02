# Projects
A repository of all the small and personal projects created by Belo Abhigyan


## Project : Dicee Game

Welcome to the Dicee Game, inspired by Angela Yu's Web Development course - Boss Level Challenge 1.

## Description

This project is a web-based Dicee Game that allows two players to roll virtual dice and determine the winner based on the roll results. It's a simple and fun game that you can play with a friend or by yourself.

## How to Play

1. Open the game in a web browser.
2. Click the "Roll Dice" button to roll the dice for Player 1 and Player 2.
3. The player with the higher dice roll wins the round.
4. The game will announce the winner and update the score.
5. Click "Play Again" to start a new round.

## Preview

![Dicee Game Screenshot](dicee_game_ss.png)

## Technologies Used

- HTML
- CSS
- JavaScript

## Credits

- This project was created as part of the "Web Development" course by Angela Yu on Udemy.

## Acknowledgments


- Special thanks to Angela Yu for providing the inspiration for this project through her course.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--- 

## Project 2 :   Suicidal Intention Detection in Text Using BERT-Based Transformers
### [Click Here](https://github.com/koachgg/Projects/blob/main/Suicide_Intent_Detection%20(1).ipynb)
## Objective: 
Developed a suite of BERT-based transformer models to detect suicidal intention in textual data, contributing to efforts to prevent suicide by analyzing social media posts for signs of ideation.

## Dataset:
The dataset was sourced from the "SuicideWatch" subreddit on Reddit, consisting of 232,074 posts categorized into suicide and non-suicide classes. It was stratified into training (162,451) and testing (69,623) sets using stratified random sampling.

## Preprocessing:

- Text preprocessing included:
- Converting capital letters to lowercase.
- Removing broken Unicode, URLs, and extra spaces.
- Expanding contractions and correcting special characters.
- Filtering out HTML tags, punctuation, and emoticons.
- Implemented the BERT preprocessing module for additional processing and tokenization.

  ## Models Implemented:

- BERT-Based Models: Utilized various pre-trained transformer models to classify text based on suicidal intention.
- BERT-Base
- ALBERT
- BERT Experts
- BERT with Talking-Heads Attention and Gated GELU
- ELECTRA
Each model was fine-tuned on the dataset with a one-cycle learning rate policy and specific configurations such as maximum length, batch size, and transformer parameters.

## Results:

- The BERT with Talking-Heads Attention and Gated GELU model achieved the highest performance with 90.64% training accuracy and 90.27% testing accuracy, along with a response time of 12 seconds for detecting suicidal intention in text.
- Other models also performed well:
1. BERT-Base: 89.75% training accuracy and 89.49% testing accuracy.
2. ALBERT: 88.34% training accuracy and 87.85% testing accuracy.
3. BERT Experts: 88.83% training accuracy and 87.58% testing accuracy.
4. ELECTRA: 87.17% training accuracy and 87.06% testing accuracy.

Model performance was evaluated using metrics such as accuracy, precision, recall, and F1-score.
 
 ## Tools & Technologies Used:

- Languages: Python
- Libraries: TensorFlow, PyTorch, Hugging Face Transformers
- Platforms: Google Colab Pro with T4 GPU and 51GB RAM

---

## Project 3 : MILK QUALITY DETECTION USING MACHINE LEARNING ALGORITHMS
### [Click Here](https://github.com/koachgg/Projects/blob/main/Milk_Quality_Detection.ipynb)
## Objective:
 Developed a machine learning model to predict and prevent milk spoilage, reducing financial losses and mitigating health risks.

## Dataset :
 Utilized a dataset from Kaggle with 1059 rows and 8 columns, consisting of seven independent features: pH, temperature, taste, odor, fat content, turbidity, and color. The target variable was the milk grade, classified into low, medium, and high.

 ## ML Techniques :
Implemented multiple machine learning algorithms for milk quality classification, including:
- AdaBoost
- Artificial Neural Networks (ANN)
- Support Vector Machines (SVM)
- Random Forest (RF)
- K-Nearest Neighbors (KNN)
- XGBoost
- Gradient Boosting (GBM)
- Decision Trees (DT)

Data pre-processing involved:
- Label encoding of categorical data.
- Feature scaling using Min-Max scaling and z-score scaling.
- Addressed skewness with PowerTransformer.
- Achieved data normalization through feature-wise scaling to ensure model convergence and prevent feature bias

## Results:

- AdaBoost outperformed other models with a classification accuracy of 99.9%.
- ANN achieved a classification accuracy of 95.4%.
- Random Forest, XGBoost, and KNN also showed high accuracy scores of 98.58%.
- Utilized confusion matrices to visualize and compare results among different algorithms.

## Tools & Technologies Used:

- Languages: Python
- Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn, xgboost
- Platforms: Kaggle
