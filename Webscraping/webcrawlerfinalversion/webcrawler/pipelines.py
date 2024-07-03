import json
import logging

class JsonWriterPipeline:
    def open_spider(self, spider):
        self.file = open('output.json', 'w', encoding='utf-8')
        self.file.write('[')
        logging.info("Opened output.json for writing")

    def close_spider(self, spider):
        if self.file.tell() > 1:  # Check if the file has content
            self.file.seek(self.file.tell() - 2, 0)  # Remove the last comma
        self.file.write(']')
        self.file.close()
        logging.info("Closed output.json")

    def process_item(self, item, spider):
        line = json.dumps(item, ensure_ascii=False) + ",\n"
        self.file.write(line)
        logging.info("Processed item: %s", item)
        return item
