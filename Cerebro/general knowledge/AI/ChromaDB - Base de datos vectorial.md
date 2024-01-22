---
sticker: emoji//1f6a7
---

```python
import chromadb
chroma_client = chromadb.Client()
```

## Create a collection:

Collections are where you'll store your embeddings, documents,
and any additional metadata. You can create a collection with a name

```python 
collection = chroma_client.create_collection(name="my_collection")
```



## Add Documents to collection
Add some text documents to the collection, Chroma will store your text, and handle tokenization, embedding, and indexing automatically

```python
collection.add(
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source_2"}],
    ids=["id1", "id2"] # Estos son los indices de los documentos
)
```
If you have already generated embeddings yourself, you can load them directly in
```python
collection.add(
    embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]],
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)
```
## Query the collection
You can query the collection with a list of query texts, and Chroma will 
return the n most similar results.  It's that easy

```python
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2
)
```

By default data stored in Chroma is ephemeral making it easy to prototype
scripts. It's easy to make Chroma persistent so you can reuse every collection 
you create and add more documents to it later. It will load your data automatically

when you start the client, and save it automatically when you close it.

## Initiating a persistent Chroma client
```python
import chromadb
```

You can configure Chroma to save and load from your local machine. Data will be persisted automatically and loaded on start (if it exists).

```python
client = chromadb.PersistentClient(path='/path/to/save/to')
```

The `path` is where Chroma will store is database files on disk, and load them to start.

The client object has a few useful convenience methods.
```python
client.heartbeat() # returns a nanosecond heartbeat, Usefull for making sure the client remains connected.
client.reset() # Empties and completly resets the database. ⚠️ This is destructive and not reversible.
``` 

## Running Chroma in client/server mode
Chroma can also be configured to run client/server mode. In this mode, the Chroma client connects to a Chroma server running in a separate process.

To start the Chroma server, run the following command:
```bash
chroma run --path /db_path
```

Then use Chroma HTTP client to connect to the server:
```python
import chromadb
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
```

That's it! Chroma's API will run in `client-server` mode with just this change.
