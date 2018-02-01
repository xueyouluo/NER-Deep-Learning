# NER-Deep-Learning
Using BiLSTM-CRF model for Chinese NER

## Char + Word Segment
This model is almost the same as [ChineseNER](https://github.com/zjy-ucas/ChineseNER), I implement it to learn the CRF model in tensorflow. I borrowed a lot data processing and evaluation code from [ChineseNER](https://github.com/zjy-ucas/ChineseNER), thanks to the author for sharing the code and the training data.

I used the config setting same as ChineseNER and got the best f1 scores of dev and test set are:
- dev: 89.35
- test: 88.53, PER = 90~91

I tried the original ChineseNER code, and got:
- dev: 90.65
- test: 91.02, PER = 93.89

## Char + Word Segment + More training data
I add training data from People's Daily 98(about 30K sentences) and Boson NLP(about 10K sentences). 