from django import template

register = template.Library()

@register.filter(name='get_item')
def get_item(dictionary, key):
    """
    Template filter to access dictionary items by key.
    Usage: {{ my_dict|get_item:key_variable }}
    """
    if dictionary is None:
        return None

    # Convert the key to string if it's not already
    str_key = str(key)

    # Try to get the value from the dictionary
    return dictionary.get(str_key)

# The library is automatically registered when Django loads the templatetags module
