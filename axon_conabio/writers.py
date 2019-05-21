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

    @abstactmethod
    def write_data(self, fileobj, data):
        pass

    @classmethod
    def get_writer(cls, path, extension):
        if extension == 'json':
            return JSONWriter(path)

        if extension == 'csv':
            return CSVWriter(path)

        msg = 'Writer not implemented for format %s' % extension
        raise NotImplementedError(msg)


class JSONWriter(FileWriter):
    def save_data(self, fileobj, data):
        json.dump(data, fileobj)


class CSVWriter(FileWriter):
    def get_fieldnames(self, data):
        return list(data[0].keys())

    def save_data(self, fileobj, data):
        writer = csv.DictWriter(
            fileobj,
            fieldnames=self.get_fieldnames(data))

        writer.writeheader()
        for row in data:
            writer.writerow(row)
