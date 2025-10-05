from mongoengine import (
    Document,
    StringField,
    IntField,
    DateTimeField,
    EmbeddedDocument,
    EmbeddedDocumentListField
)

class MediaItem(EmbeddedDocument):
    index = IntField(required=True)
    type = StringField(required=True)
    url = StringField(required=True)

class Post(Document):
    post_id = StringField(required=True, unique=True)
    profile_id = StringField(required=True)
    text = StringField(required=True)
    likes_count = IntField(min_value=0, required=True)
    comments_number = IntField(min_value=0, default=0)
    created_at_utc = DateTimeField(required=True)
    scraped_at = DateTimeField(required=True)
    updated_at = DateTimeField(required=True)
    media_items = EmbeddedDocumentListField(MediaItem)

    meta = {
        "collection": "posts",
    }
