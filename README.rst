darca-embeddings
================

Modular, backend-agnostic interface for generating text embeddings. 
Provides a consistent Pythonic API for embedding text using various underlying providers.

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
