import chromadb

from spotify import get_top_100_songs
def build_client():
    client = chromadb.Client()
    collection = client.get_or_create_collection('songs')
    id  = []
    document = []
    idx = 1
    top_tracks = get_top_100_songs()
    for track in top_tracks:
        document.append(track['name'])
        id.append(str(idx))
        idx = idx + 1

    collection.add(
        
    ids=id,
    documents=document
    )
    #sorted_emotions, n_top_values = main()

    return collection
    #print(collection.query(query_texts= "beautiful", n_results=10)['documents'])


