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

# Pull Requests

- One pull request per change request.
- A pull request either targets `main` or is part of a stack made with `gh stack`, which models its dependencies on other pull requests. Never chain pull requests by setting another pull request's branch as the base without `gh stack`.
- Before a pull request is opened, and again before it is merged, its commits on top of `main` (`git log origin/main..<branch>`) are only its own. A pull request that would bring in another open pull request's commits is not merged.
- Publish a pull request as soon as it is ready, so CI runs on it while other work goes on.
- Merge pull requests one at a time, from the bottom of a stack, and wait for CI on `main` to pass after each merge before merging the next.
