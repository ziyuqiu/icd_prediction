# icd_prediction
Predicting Medical Billing Codes (ICD9) from Clinical Notes (in MIMIC-III datasets) using Deep Learning

## Introduction
This is the code for our final project for cs5787 Deep Learning at Cornell University. In this project, we investigate how to approximate the mapping from the medical notes to predict the medical codes in the revenue cycle. This can be achieved through multi-label classification, hierarchical classification, and information retrieval. In our work, We will implement 5 models: (i) Hierarchical Self Attention Network (ii) Self-Attention Long-Short Term Memory (SLSTM) (iii) Codes Attentive Long-Short Term Memory (CLSTM) (iv) Hierarchical Attention Network (HAN) (v) Fine Tuning on Pre-trained Clinical BERT.
Particularly, we try two types of attention mechanisms for different model architectures. All our models are validated on the publicly available MIMIC-IIIdataset using macro-F1, micro-F1 and precision@N metrics.

## Data
We will train and evaluate our approach on [MIMIC-III](https://www.nature.com/articles/sdata201635) (Medical Information Mart for Intensive Care). It is an open-access dataset comprising de-identified medical records from Beth Israel Deaconess Medical Center from 2001 to 2012. The data is associated with 58,976 distinct hospital admissions from 46,520 patients. Each record describes the diagnoses and procedures during a patient’s stay, including basic structured information, free-text clinical notes, and ICD-9 codes tagged by humans. A detailed description of the dataset could be found [here](https://mimic.physionet.org/). For the purpose of this project, we only used the tables below:
![Data Schema](./imgs/Schema.png)

## Preprocessing
[Data Processing](./Preprocess/data_processing.ipynb) file contains methods to extract, transform and load the medical notes and codes from corresponding tables. 
[Note Sections](./Preprocess/note_sections.ipynb) file slice the medical notes into different sections, and then further sliced each section to individual sentences. By consulting literatures and medical professionals, the preprocess discarded meaningless portions, converted text to lowercase, removed stopwords and special characters, and peformed lemmatization and tokenization.
[High Level Codes](./Preprocess/high_lvl_codes.ipynb) file mapps the low level ICD codes to higher levels and group them into categories.

## Models
[TBD]
### LSTM (Long-Short Term Memory Networks)
Self-Attention Long-Short Term Memory (SLSTM) and Codes Attentive Long-Short Term Memory (CLSTM)
![LSTM](./imgs/LSTM.png)

### HAN (Hierarchical Attention Network)
<img src="https://github.com/ziyuqiu/icd_prediction/imgs/HAN/png" width="400">
![HAN](./imgs/HAN.png | width=400)

### BERT (Bidirectional Encoder Representations from Transformers)



## Authors

- Ziyu(Andrea) Qiu @ziyuqiu zq64@cornell.edu
- Yezhou(Yeats) Ma @YeatsMar ym462@cornell.edu
- Jingxuan(Kelly) Sun @shakingkelly js3422@cornell.edu
- Ta-Wei(David) Mao @davidmao8419 tm592@cornell.edu
- Yixue Wang yw2224@cornell.edu

