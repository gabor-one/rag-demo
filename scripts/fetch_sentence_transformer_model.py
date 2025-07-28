from rag_solution.singleton_worker_pool import test_sentence_transformer_model

# This is extremely ugly.
#  I should have disected the sentece_transformer for the proper call to download.
#  Sadly they don't expose a interface to download the model.
#  For the sake of time-management I will just run a dummy call to the model.
test_sentence_transformer_model()
print("Sentence Transformer model downloaded successfully.")