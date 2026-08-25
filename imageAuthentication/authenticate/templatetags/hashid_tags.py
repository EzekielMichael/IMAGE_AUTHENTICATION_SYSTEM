from django import template
from ..utils import encode_id

register = template.Library()

@register.filter(name='hashid')
def hashid(value):
    """Convert ID to hashed string for URLs"""
    if value is None:
        return ""
    try:
        return encode_id(int(value))
    except:
        return str(value)