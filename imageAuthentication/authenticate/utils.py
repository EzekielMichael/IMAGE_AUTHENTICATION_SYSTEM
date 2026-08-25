from hashids import Hashids

hashids = Hashids(salt="forensix-secret-salt-2026.", min_length=8)

def encode_id(id):
    return hashids.encode(id)

def decode_id(hashed_id):
    decoded = hashids.decode(hashed_id)
    return decoded[0] if decoded else None