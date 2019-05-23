from abc import abstractmethod
import json
import csv


class FileWriter(object):
    extension = ''

    def get_filename(self, path):
        return '{}.{}'.format(path, self.extension)

    def save(self, path, data):
        filename = self.get_filename(path)

        with open(filename, 'w') as f:
            self.write_data(f, data)

    @abstractmethod
    def write_data(self, fileobj, data):
        pass

    @classmethod
    def get_writer(cls, path, extension):
        if extension == 'json':
            return JSONWriter()

        if extension == 'csv':
            return CSVWriter()

        msg = 'Writer not implemented for format %s' % extension
        raise NotImplementedError(msg)


class JSONWriter(FileWriter):
    extension = 'json'

    def write_data(self, fileobj, data):
        json.dump(data, fileobj)


class CSVWriter(FileWriter):
    extension = 'csv'

    @staticmethod
    def get_fieldnames(data):
        return list(data[0].keys())

    def write_data(self, fileobj, data):
        writer = csv.DictWriter(
            fileobj,
            fieldnames=self.get_fieldnames(data))

        writer.writeheader()
        for row in data:
            writer.writerow(row)
