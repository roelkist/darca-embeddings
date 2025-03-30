Usage Guide
===========

This page demonstrates how to use **darca-embeddings** in your projects.

Installation
------------

#. Ensure you have Python 3.8+ installed (or whichever version is required).
#. Install using Poetry (or from source if you prefer), for example:

   .. code-block:: console

      make install

   This will create a virtual environment (unless in CI) and install all dependencies.

Environment Variables
---------------------

For the **OpenAI** backend, you must set an environment variable named ``OPENAI_API_KEY``:

.. code-block:: console

   export OPENAI_API_KEY="your-openai-api-key"

Basic Usage
-----------

#. **Import** and **initialize** an embedding client:

   .. code-block:: python

      from darca_embeddings import EmbeddingClient

      # By default, uses the OpenAI backend with model "text-embedding-ada-002"
      client = EmbeddingClient()

#. **Generate a single embedding**:

   .. code-block:: python

      embedding = client.get_embedding("Hello World!")
      print(embedding)  # a list of floating-point numbers

#. **Generate multiple embeddings**:

   .. code-block:: python

      texts = ["Hello World!", "Another text to embed"]
      embeddings = client.get_embeddings(texts)
      for i, emb in enumerate(embeddings):
          print(f"Embedding for text {i}:", emb)

Backend Support
---------------

Currently, **OpenAI** is the only production-ready backend. You can switch backends by:

.. code-block:: python

   # Attempting the "huggingface" backend will raise an EmbeddingException (unimplemented).
   client = EmbeddingClient(backend="huggingface")

Error Handling
--------------

The package defines custom exceptions to help you handle various error scenarios:

- **EmbeddingAPIKeyMissing**: No valid API key found for the chosen provider.
- **EmbeddingResponseError**: An embedding request failed or returned unexpected data.
- **EmbeddingException**: Base class for all darca-embeddings custom exceptions.

Refer to the :ref:`api-reference` for a full API breakdown.
