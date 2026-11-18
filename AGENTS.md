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

# User Data

Nothing from a development session or from the developer's own setup ever goes into the repository: not into code, docstrings, comments, tests, fixtures, test data, docs, commit messages or pull request text. That covers what was said in a conversation, and every user customization, for example:

- names of personas, voices, wake words and sleep words, and persona descriptions;
- cloned voices, voice samples and recordings;
- memory, transcripts and conversation content;
- model servers, hostnames, addresses, ports and the models in use;
- local paths, usernames, email addresses, keys and tokens.

Use neutral placeholders instead: invented names (Ada, Robin, Felix), `https://<host>:<port>/v1`, "a small model". Tests make the personas, voices and data they need themselves, and never use the developer's own personas, voices, memory or settings. Downloaded model files may come from the shared cache, and the end-to-end test talks only to the model server it is given, never to one it finds on the machine.

# Pull Requests

- One pull request per change request.
- A pull request either targets `main` or is part of a stack made with `gh stack`, which models its dependencies on other pull requests. Never chain pull requests by setting another pull request's branch as the base without `gh stack`.
- Before a pull request is opened, and again before it is merged, its commits on top of `main` (`git log origin/main..<branch>`) are only its own. A pull request that would bring in another open pull request's commits is not merged.
- Publish a pull request as soon as it is ready, so CI runs on it while other work goes on.
- Merge pull requests one at a time, from the bottom of a stack, and wait for CI on `main` to pass after each merge before merging the next.
- When mentioning an issue or pull request, always add its title in parentheses: "#95 (Show the features in a short tour, at the end of the walkthrough or from Settings)", never a bare "#95". This holds everywhere: commit messages, pull request and issue text, code comments and docs.
