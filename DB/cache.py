import atexit
import os
import pickle
from datetime import datetime


class Cachebase:

    cache = {}
    cache_file_path = "cache_file.pkl"

    def __init__(self):
        self.load_cache()
        atexit.register(self.save_cache)  # Register save_cache to be called at program exit

    def get_cache_key(self, *args, **kwargs):
        return str(args) + str(kwargs)

    def is_cache_valid(self, key):
        if key in self.cache:
            timestamp, _ = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < 7200:  # 2 hours
                return True
        return False

    def get_from_cache(self, key):
        return self.cache[key][1] if key in self.cache else None

    def set_cache(self, key, value):
        self.cache[key] = (datetime.now(), value)

    def save_cache(self):
        with open(self.cache_file_path, 'wb') as cache_file:
            pickle.dump(self.cache, cache_file)


    def load_cache(self):
        if os.path.exists(self.cache_file_path):
            try:
                    with open(self.cache_file_path, 'rb') as cache_file:
                        self.cache = pickle.load(cache_file)
                    print("Cache loaded successfully.")
                    return True
            except (pickle.PickleError, EOFError, FileNotFoundError, Exception) as e:
                    print(f"Failed to load cache: {e}")
                    self.cache = {}
                    return False
        else:
            print("Cache file does not exist. Initiate the mem cache ")
            self.cache = {}
            return False
