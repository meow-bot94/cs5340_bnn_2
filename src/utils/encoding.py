import base64
import uuid


def generate_32_byte_url_safe_key():
    byte_key = uuid.uuid4().bytes + uuid.uuid4().bytes
    encoded_key = base64.urlsafe_b64encode(byte_key)
    return encoded_key


def encode_b64(data: str) -> str:
    return base64.urlsafe_b64encode(data.encode('utf-8')).decode('utf-8')


def decode_b64(data: str) -> str:
    return base64.urlsafe_b64decode(data.encode('utf-8')).decode('utf-8')
