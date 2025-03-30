darca-embeddings
================

Modular, backend-agnostic interface for generating text embeddings. 
Provides a consistent Pythonic API for embedding text using various underlying providers.

|Build Status| |Deploy Status| |CodeCov| |Formatting| |License| |PyPi Version| |Docs|

.. |Build Status| image:: https://github.com/roelkist/darca-embeddings/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/roelkist/darca-embeddings/actions
.. |Deploy Status| image:: https://github.com/roelkist/darca-embeddings/actions/workflows/cd.yml/badge.svg
   :target: https://github.com/roelkist/darca-embeddings/actions
.. |Codecov| image:: https://codecov.io/gh/roelkist/darca-embeddings/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/roelkist/darca-embeddings
   :alt: Codecov
.. |Formatting| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black code style
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
.. |PyPi Version| image:: https://img.shields.io/pypi/v/darca-embeddings
   :target: https://pypi.org/project/darca-embeddings/
   :alt: PyPi
.. |Docs| image:: https://img.shields.io/github/deployments/roelkist/darca-embeddings/github-pages
   :target: https://roelkist.github.io/darca-embeddings/
   :alt: GitHub Pages

Key Features
------------
- **Unified Embedding Interface**: Use one client class for multiple backends.
- **Default OpenAI Support**: Integrates seamlessly with OpenAI's Embedding API.
- **Extensible Design**: Easy to add new backends (e.g., Hugging Face).

Quick Start
-----------
1. **Install** (using Make + Poetry):

   .. code-block:: console

      make install

2. **Set** your environment variables (for example, ``OPENAI_API_KEY`` for OpenAI).
3. **Use** the embedding client:

   .. code-block:: python

      from darca_embeddings import EmbeddingClient

      client = EmbeddingClient()
      embedding = client.get_embedding("Hello World!")
      print(embedding)

Documentation
-------------
We use **Sphinx** for building documentation. See `docs/source/` for the reST files. 
You can build the docs locally by:

.. code-block:: console

   make docs

Contributing
------------
Contributions are welcome! See the `CONTRIBUTING.rst`_ for more details.

License
-------
This project is licensed under the MIT License. See the `LICENSE` file for more details.

Author
------
**Roel Kist**

.. _CONTRIBUTING.rst: ./CONTRIBUTING.rst
