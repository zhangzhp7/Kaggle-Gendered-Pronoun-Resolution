# Kaggle Competition: Gendered Pronoun Resolution
# Method 1: LSTM model with glove embeddings

## Introduction
In this competition, you must identify the target of a pronoun within a text passage. The source text is taken from Wikipedia articles. You are provided with the pronoun and two candidate names to which the pronoun could refer. You must create an algorithm capable of deciding whether the pronoun refers to name A, name B, or neither.
## Data Description
- ID - Unique identifier for an example (Matches to Id in output file format)  
- Text - Text containing the ambiguous pronoun and two candidate names (about a paragraph in length)  
- Pronoun - The target pronoun (text)  
- Pronoun-offset The character offset of Pronoun in Text  
- A - The first name candidate (text)  
- A-offset - The character offset of name A in Text  
- B - The second name candidate  
- B-offset - The character offset of name B in Text  
- URL - The URL of the source Wikipedia page for the example  
## Preprocessing
- text cleaning
- tokenizer
- pad the sentence
- label preprocess
## Glove embeddings

## LSTM model
- 1st layer: Glove embedding layer
- 2nd layer: SpatialDropout1D (0.4)
- 3rd layer: Bidirectional LSTM
- 4th layer: GlobalMaxPool1D
- 5th layer: Dense(32)
- 6th layer: Dropout(0.6)
- 7th layer: Output dense(softmax)
## Code

## Result
- Trainning set: 2454 samples  
- Validation set: 2000 samples  
Validation dataset:  
Accuracy: 0.53  
Log loss: 0.93  

# Todo
- Attention   
- Spacy embeddings  
- Bert embeddings  

