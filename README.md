# Challenges in Regulating Nanoparticles in Consumer Products
Author:Megha Roy Chowdhury  
Roll Number:25SCS1003004051    
Institution:IILM University, Greater Noida  Supervisor: Dr. Saurabh Shanu


## Project Overview
This project addresses the environmental and health risks associated with the unregulated use of nanoparticles in consumer goods. Due to their small size and high reactivity, these particles pose toxicity risks that traditional labeling often fails to explain in layman's terms.

The proposed system is a comprehensive toxicity analysis pipeline that allows consumers to scan product labels to determine safety levels and toxicity risks using AI and Machine Learning.

# Key Features
1) Automated Text Extraction: Uses EasyOCR to identify compounds (e.g., ZnO) from scanned images.
2) Toxicology Prediction:Employs Machine Learning (KNN & Regression)to predict long-term health effects based on dosage and exposure.
3) NLP Processing:Utilizes NLTK for tokenization and normalization of chemical descriptions.
4) Interactive Dashboard:A Tkinter-based interface for easy image uploading, result visualization, and CSV data export.

## Technical Stack
>Language:Python
>AI/ML Libraries:PyTorch,Scikit-learn, Pandas,NumPy
>OCR & NLP:EasyOCR, NLTK
>Visualization:Matplotlib,Tkinter
>Backend/Storage:MySQL,AWS Cloud Storage 
Performance Results
The model was tested on real-world datasets and achieved the following metrics:
Classification Accuracy:1.000 (Logistic Regression) — perfectly distinguishes between toxic and non-toxic entries.
Regression MSE:0.000 (Linear Regression) — provides precise dosage-based toxicity scores.
KNN Similarity:Successfully matched "ZnO" (Zinc Oxide) with a 0.000 cosine distance (perfect match).

