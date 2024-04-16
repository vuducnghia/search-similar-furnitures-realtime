import lancedb
from lancedb.pydantic import LanceModel, Vector


class Furnitures(LanceModel):
    vector: Vector(256)
    style: str
    image_uri: str


class DB:
    def __init__(self, uri):
        self.connection = lancedb.connect(uri)

    def get(self):
        return self.connection.table_names()

    def create_embeddings_table(self, table_name, data):
        table = self.connection.create_table(table_name, schema=Furnitures, exist_ok=True)
        table.add([{"style": data["style"], "image_uri": data["image_uri"], "vector": data["vector"]}])

    def search(self, table_name, data):
        return self.connection.open_table(table_name).search(data).limit(5).to_pandas()

    def count(self, table_name):
        return self.connection.open_table(table_name).count_rows()
