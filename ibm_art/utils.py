import gzip
import pathlib
import pickle

def save_zip(object, path, protocol=0):
    """
    Saves a compressed object to disk
    """
    # Create the folder, if necessary
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    file = gzip.GzipFile(path, 'wb')
    pickled = pickle.dumps(object, protocol)
    file.write(pickled)
    file.close()



def load_zip(path):
    """
    Loads a compressed object from disk
    """
    file = gzip.GzipFile(path, 'rb')
    buffer = b""
    while True:
        data = file.read()
        if data == b"":
            break
        buffer += data
    obj = pickle.loads(buffer)
    file.close()
    return obj