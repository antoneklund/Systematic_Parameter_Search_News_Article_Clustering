import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize



def main():
    df = pd.read_pickle("datasets/huffpost_final_220221.pkl")
    df["doc_vec"] = None
    
    model = Doc2Vec.load("models/doc2vec.model")
    
    for i, row in df.iterrows():
        df["doc_vec"].iloc[i] = model.infer_vector(word_tokenize(row["text"]))

    print(df)
    df.to_pickle("dim_vectors/huffpost_d2v_full_trained_220222.pkl")
    


if __name__ == "__main__":
    main()