---
sticker: emoji//1f6a7
---
Docs: https://docs.trychroma.com/usage-guide

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
return the `n` most similar results.  It's that easy

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

### Using the python http-only client
If you are running chroma in client-server mode, you may not need the full Chroma library. Instead, you can use the lightweight client-only library. In this case you can install the `chromadb-client`package. This package is a lightweight HTTP client for the server with a minimal dependency footprint.

```bash
pip install chromadb-client
```

```python
import chromadb
#Example setup of the client to connect to your chroma server
client = chromadb.HttpClient(host='localhost', port=8000)
```

Note that the `cromadb-client` package is a subset of the full Chroma library and does not include all the dependencies. If you want to use full Chroma library, you can install the `chromadb`package instead.

Most importantly, there is no default embedding function. If you add() documents without embeddings, you most have manually specified an embedding function and installed the dependencies for it.

## Using collections
Chroma lets you manage collections of embeddings, using the `collection` primitive.

### Creating, inspecting, and deleting Collections
Chroma uses collection names in the url, so there are a few restrictions on naming them:
- The length of the name must be between 3 and 63 characters.
- The name must start and end with a lowercase letter or a digit, and it can contain dots, dashes, and underscores in between.
- The name must not contain two consecutive dots.
- The name must not be a valid IP address.

Chroma collections are created with a name and an optional embedding function. If you supply an embedding function, you must supply it every time you get the collection.

```python
collection = client.create_collection(name="my_collection", embedding_function=emb_fn)

collection = client.get_collection(name="my_collection", embedding_function=emb_fn)
```

>[!Caution] 
>If you later wish to `get_collection`, you MUST do so with the embedding function you supplied while creating the collection

The embedding function takes text as input, and performs tokenization and embedding. If no embedding function is supplied, Chroma will use **sentence transformer** as a default.

You can learn more about [[Embedding functions|Embedding functions]] and how to create your own.

Existing collections can be retrieved by name with `.get_collection`, and deleted with `.delete_collection`. You can also use `.get_or_create_collectio` to get a collection if it exists, or create it if it doesn't.

```python
collection = client.get_collection(name="test") # Get collection obj from an existing collection, by name. Will raise an exception if it's not found.

collection = client.get_or_create_collection(name="test") # Get a collection obj from an existing collection, by name. If it doen't exist, create it.

client.delete_collection(name="my_collection") # Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destrucctive and not reversible.

```

Collection have a few useful convenience methods.

```python
collection.peek() # returns a list of the first 10 items in the collection

collection.count() # returns the number of items in the collection

collection.modify(name="new_name") # Rename the collection
```

### Changing the distance function

`create_collection` also takes an optional `metadata` argument which can be used to customize the distance method of the embedding space by setting the value of `hnsw:space`.
```python
collection = client.create_collection(
    name="collection_name",
    metadata={"hnsw:space": "cosine"} # l2 is the default
)
```

Valid options for `hnsw:space` are "l2", "ip, "or "cosine". The **default** is "l2" which is the squared L2 norm.

|Distance|parameter|Equation|
|---|:-:|--:|
|Squared L2|'l2'|$d = \sum\left(A_i-B_i\right)^2$|
|Inner product|'ip'|$d = 1.0 - \sum\left(A_i \times B_i\right) $|
|Cosine similarity|'cosine'|$d = 1.0 - \frac{\sum\left(A_i \times B_i\right)}{\sqrt{\sum\left(A_i^2\right)} \cdot \sqrt{\sum\left(B_i^2\right)}}$|

## Adding data to a Collection
Add data to Chroma with `.add`.

Raw documents:
```python
collection.add(
	documents=["lorem ipsum....", "doc2", "doc3", ...],
	metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
	ids=["id1", "id2", "id3", ...]
)
```

If Chroma is passed a list of `documents`, it will automatically tokenize and embed them with the collection's embedding function (the default will be used if none was supplied at collection creation).

Chroma will also store the `documents`themselves. If the documents are too large to embed using the chosen embedding function, an exception wwill be raised.

Each document must have a unique associated `id` .  Trying to `.add` the same ID twice will result in only the initial value being stored. An optional list of `metadata` dictionaries can be supplied for each document, to store additional information and enable filtering.

Alternatively, you can supply a list of document-associated `embeddings` directly, and Chroma will store the associated documents without embedding them itself.

```python
collection.add(
    documents=["doc1", "doc2", "doc3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    ids=["id1", "id2", "id3", ...]
)
```

If the supplied `embeddings` are not the same dimension as the collection, an exception will be raised.

You can also store documents elsewhere, and just supply a list of `embeddings`and `metadata` to Chroma.

You can use the `ids` to associate the embeddings with your documents stored elsewhere.

```python
collection.add(
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    ids=["id1", "id2", "id3", ...]
)
```

## Querying a Collection
Chroma collections can be queried in a variety of ways, using the `.query` method.

You can query by a set of `query_embeddings`.
```python
collection.query(
	query_embeddings=[[11.1, 12.1, 13.1], [1.1, 2.3, 3.2], ...],
	n_results=10,
	where={"metadata_field": "is_equal_to_this"},
	where_document={"$contains": "search_string"}
)
```

The query will return  the `n_results` closest matches to each `query_embedding` ,  in order. An optional `where` filter dictionary can be supplied to filter by the `metadata` associated with each document.
Additionally, an optional `where_document` filter dictionary can be supplied to filter by contents of the document.

If the supplied `query_embeddings` are not the same dimension as the collection, an exception will be raised.

You can also query by a set of `query_texts`. Chroma will first embed each `query_text` with the collection's embedding function, and then perform the query with the generated embedding.

```python
collection.query(
    query_texts=["doc10", "thus spake zarathustra", ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":"search_string"}
)
```

You can also retrieve items from a collection by `id` using `.get` .

```python
collection.get(
    ids=["id1", "id2", "id3", ...],
    where={"style": "style1"}
)
```

`.get` also supports the `where` and `where_document` filters. If no `ids` are supplied, it will return all items in the collection that match the `where` and `where_document` filters.

### Choosing which data is returned
When using get or query you can use the include parameter to specify which data you want returned - any of `embeddings` , `documents` , `metadatas` , and for query, `distances` . By default, Chroma will return the `documents`, `metadatas` and in the case of query, the `distances` of the results. `embeddings` are excluded by default for performance and the `ids` are always returned. You can specify which of there you want returned by passing an array of included field names to the includes parameter of the query or get method.

```python

# Only get documents and ids
collection.get({
    include: [ "documents" ]
})

collection.query({
    queryEmbeddings: [[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
    include: [ "documents" ]
})

```

### Using Where filters
Chroma supports filtering queries by `metadata` and `document` contents. The `where` filter is used to filter by `metadata` , and the `where_document` filter is used to filter by `document` contents.

#### Filtering by metadata
In order to filter on metadata, you must supply a `where` filter dictionary to the query. The dictionary must have the following structure:

```python
{
    "metadata_field": {
        <Operator>: <Value>
    }
}
```

Filtering metadata supports the following operators:

- `$eq` - equal to (string, int, float)
- `$ne` - not equal to (string, int, float)
- `$gt` - greater than (int, float)
- `$gte` - greater than or equal to (int, float)
- `$lt` - less than (int, float)
- `$lte` - less than or equal to (int, float)

Using the `$eq` operator is equivalent to using the `where` filter.

```python
{
    "metadata_field": "search_string"
}

# is equivalent to

{
    "metadata_field": {
        "$eq": "search_string"
    }
}

```

>[!note]
>Where filters only search embeddings where the key exists. If you search `collection.get(where={"version": {"$ne": 1}})`. Metadata that does not have the key `version` will not be returned.

Continuación https://docs.trychroma.com/usage-guide#filtering-by-document-contents

