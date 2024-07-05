import configparser


class SingletonINIUtility:
    _instance = None

    def __new__(cls, file_path):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.file_path = file_path
            cls._instance.config = configparser.ConfigParser()
            cls._instance.read_ini()
        return cls._instance

    def read_ini(self):
        self.config.read(self.file_path)

    def get_value(self, section, key):
        return self.config[section][key]

    def set_value(self, section, key, value):
        self.config[section][key] = value
        self.save_changes()

    def save_changes(self):
        with open(self.file_path, 'w') as configfile:
            self.config.write(configfile)

