python clustering_loop.py -v dim_vectors/bert_snack.pkl -s scores/scores_bert_snack_230111_pca.pkl -d snack -nbc -r pca
python clustering_loop.py -v dim_vectors/bert_snack.pkl -s scores/scores_bert_snack_230111.pkl -d snack -nbc -r umap
python clustering_loop.py -v dim_vectors/bert_reuters.pkl -s scores/scores_bert_r_230111_pca.pkl -d r -nbc -r pca
python clustering_loop.py -v dim_vectors/bert_reuters.pkl -s scores/scores_bert_r_230111.pkl -d r -nbc -r umap

python clustering_loop.py -v dim_vectors/doc2vec_snack.pkl -s scores/scores_d2v_snack_230111_pca.pkl -d snack -nbc -r pca
python clustering_loop.py -v dim_vectors/doc2vec_snack.pkl -s scores/scores_d2v_snack_230111.pkl -d snack -nbc -r umap
python clustering_loop.py -v dim_vectors/doc2vec_reuters.pkl -s scores/scores_d2v_r_230111_pca.pkl -d r -nbc -r pca
python clustering_loop.py -v dim_vectors/doc2vec_reuters.pkl -s scores/scores_d2v_r_230111.pkl -d r -nbc -r umap

python clustering_loop.py -v dim_vectors/doc2vec_ag.pkl -s scores/scores_d2v_ag_230111_pca.pkl -d ag -nbc -r pca
python clustering_loop.py -v dim_vectors/doc2vec_ag.pkl -s scores/scores_d2v_ag_230111.pkl -d ag -nbc -r umap
python clustering_loop.py -v dim_vectors/bert_ag.pkl -s scores/scores_bert_ag_230111_pca.pkl -d ag -nbc -r pca
python clustering_loop.py -v dim_vectors/bert_ag.pkl -s scores/scores_bert_ag_230111.pkl -d ag -nbc -r umap
