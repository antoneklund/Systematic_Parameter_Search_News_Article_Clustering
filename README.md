# Cluster loop for parameter search

## Clustering loop instructions
1. Install the requirements with pip install -r requirements.txt
2. Copy a raw dataset to the folder datasets. The raw dataset should be a pandas.DataFrame pickle-file created with pandas.DataFrame.to_pickle() with the columns [id, text, label].
3. Pre-process datasets for doc2vec and bert with 
    python create_vectors.py -po -c <corpus_path> -d <cleaned_dataset_prefix>
    * The dataset prefix will be combined into filenames like: <cleaned_dataset_prefix>_doc2vec_clean.pkl and <cleaned_dataset_prefix>_bert_clean.pkl after preprocessing.

4. Create doc2vec vectors with 
    python create_vectors.py -o d2v -c datasets/<dataset>_doc2vec_clean.pkl -s dim_vectors/<dataset>_doc2vec_vectors.pkl
5. Create BERT vectors with (TAKES A WHILE ON CPU)
    python create_vectors.py -o bert-base -c datasets/<dataset>_bert_clean.pkl -s dim_vectors/<dataset>_bert_vectors.pkl
    * For simplicity, the default is vectors from bert-base-uncased used without further training.
6. Run a loop with 
    python clustering_loop.py -v dim_vectors/<dataset>_doc2vec_vectors.pkl  -s scores/scores_<dataset>_d2v_230328.pkl -d snack -nbc -r pca
    * If your dataset has a different number of labels than 6, then the easiest way is to use the flag “-d snack” and change the settings in SNACK_MEANS_SETTINGS instead of trying to change all the code.


## Plotting
The dimensionality_reduction/plotting.py is used for graphs in papers. 
(TODO: Make an argument parser that can run these properly.)
