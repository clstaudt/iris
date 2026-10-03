# Coding Conventions

## Imports

All imports must be at module level. Do not import inside function or method bodies.

```python
# Wrong
def fetch_data(url):
    import requests
    return requests.get(url)

# Right
import requests

def fetch_data(url):
    return requests.get(url)
```

Optional dependencies that may not be installed should use a module-level try/except guard:

```python
try:
    from duckduckgo_search import DDGS
    HAS_SEARCH = True
except ImportError:
    HAS_SEARCH = False
```
