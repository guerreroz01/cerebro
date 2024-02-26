import chromadb
chroma_client = chromadb.Client()

""" Create a collection:

    Collections are where you'll store your embeddings, documents,
    and any additional metadata. You can create a collection with a name:
"""

collection = chroma_client.create_collection(name="my_collection")



########################### Add Documents to collection #########################
""" Add some text documents to the collection

    Chroma will store your text, and handle tokenization, embedding, and
    indexing automatically
"""

collection.add(
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source_2"}],
    ids=["id1", "id2"] # Estos son los indices de los documentos
)

""" If you have already generated embeddings yourself, yoy can load them
    directly in
"""

collection.add(
    embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]],
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)

########################## Query the collection #################################
""" You can query the collection with a list of query texts, and Chroma will 
return the n most similar results.  It's that easy
"""

results = collection.query(
    query_texts=["This is a query document"],
    n_results=2
)

""" By default data stored in Chroma is ephemeral making it easy to prototype
scripts. It's easy to make Chroma persistent so you can reuse every collection 
you create and add more documents to it later. It will load your data automatically
when you start the client, and save it automatically when you close it.
"""