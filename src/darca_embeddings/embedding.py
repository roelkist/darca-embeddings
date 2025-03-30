import os
from abc import ABC, abstractmethod

import openai
from darca_exception.exception import DarcaException
from darca_log_facility.logger import DarcaLogger
from openai import OpenAIError

# === Custom Exceptions ===


class EmbeddingException(DarcaException):
    """
    Base class for all darca-embeddings exceptions.

    This exception can be extended or raised for any errors
    that occur within the darca-embeddings package.
    """


class EmbeddingAPIKeyMissing(EmbeddingException):
    """
    Raised when the API key is missing for the selected embeddings provider.

    Typically occurs when the environment variable (e.g. ``OPENAI_API_KEY``)
    is not set or is invalid.
    """


class EmbeddingResponseError(EmbeddingException):
    """
    Raised when an embeddings API request fails or returns malformed data.

    For example, it may be raised if the network request times out, the API
    returns a non-200 status code, or the response is not in the expected
    format.
    """


# === Abstract Base Client ===


class BaseEmbeddingClient(ABC):
    """
    Abstract base class for Embedding providers.

    Implementations of this class must provide methods to generate embeddings
    from text input, typically by interacting with an external model or
    service.
    """

    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """
        Generate a vector embedding for a single piece of text.

        :param text: The input string to be embedded.
        :type text: str
        :returns: A floating-point vector (list) representation of the input
                  text.
        :rtype: list[float]
        :raises EmbeddingException: If something goes wrong with the embedding
                request.
        """

        raise NotImplementedError("Must be overridden in subclass.")

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate vector embeddings for a list of input texts.

        By default, this method calls :meth:`get_embedding` for each string
        in ``texts``.
        Subclasses may override this method to provide more efficient batch
        processing.

        :param texts: A list of input strings to embed.
        :type texts: list[str]
        :returns: A list of embedding vectors, where each vector is a list of
                 floats.
        :rtype: list[list[float]]
        :raises EmbeddingException: If something goes wrong while generating
                                    embeddings.
        """

        return [self.get_embedding(t) for t in texts]


# === OpenAI Implementation ===


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """
    Embedding provider that uses OpenAI's Embedding API.

    This client relies on the ``OPENAI_API_KEY`` environment variable
    for authentication. It uses the default model
    ``text-embedding-ada-002`` unless otherwise specified.

    :param model: The OpenAI Embedding model to use.
    :type model: str
    :raises EmbeddingAPIKeyMissing: If the required API key environment
                                    variable is not found.
    """

    def __init__(self, model="text-embedding-ada-002"):
        self.logger = DarcaLogger("darca-embeddings").get_logger()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingAPIKeyMissing(
                message="OPENAI_API_KEY environment variable is not set.",
                error_code="EMBEDDING_API_KEY_MISSING",
                metadata={"provider": "openai"},
            )
        openai.api_key = api_key
        self.model = model

    def get_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for a single piece of text using OpenAI.

        :param text: The input text to be embedded.
        :type text: str
        :returns: A list of floats representing the embedding of the text.
        :rtype: list[float]
        :raises EmbeddingResponseError:
                - If the OpenAI API call fails or returns
                  an unexpected format.
                - If an unknown error occurs during processing.
        """

        try:
            self.logger.debug(
                "Requesting embedding from OpenAI", extra={"model": self.model}
            )
            response = openai.embeddings.create(input=text, model=self.model)
            # Each 'data' item has an 'embedding' list
            embedding = response.data[0].embedding
            self.logger.debug("Received embedding from OpenAI")
            return embedding

        except OpenAIError as oe:
            raise EmbeddingResponseError(
                message="OpenAI embedding API returned an error.",
                error_code="EMBEDDING_API_REQUEST_FAILED",
                metadata={"model": self.model, "text_preview": text[:100]},
                cause=oe,
            )
        except Exception as e:
            raise EmbeddingResponseError(
                message=(
                    "Unexpected failure during OpenAI "
                    "embedding response parsing."
                ),
                error_code="EMBEDDING_RESPONSE_PARSE_ERROR",
                metadata={"model": self.model},
                cause=e,
            )


# === EmbeddingClient Wrapper ===


class EmbeddingClient:
    """
    A unified client for generating embeddings using the darca pluggable
    backend system. Defaults to OpenAI.

    This class wraps the functionality of various embedding providers
    behind a single interface. By specifying the ``backend`` argument,
    you can switch between multiple embeddings backends (e.g., OpenAI,
    HuggingFace, etc.).

    :param backend: The backend provider to use, e.g. ``openai``
                    or ``huggingface``.
    :type backend: str
    :param kwargs: Additional keyword arguments passed to the backend
                   client's constructor.
    :raises EmbeddingException: If the chosen backend is not implemented
                                or unsupported.
    """

    def __init__(self, backend: str = "openai", **kwargs):
        """
        Initialize the EmbeddingClient with the given backend and
        model settings.

        :param backend: The embedding backend to use (default: ``openai``).
        :type backend: str
        :param kwargs: Additional parameters for the backend's constructor.
        """
        if backend == "openai":
            self._client = OpenAIEmbeddingClient(**kwargs)
        elif backend == "huggingface":
            # Placeholder for a future Hugging Face implementation
            raise EmbeddingException(
                message=(
                    f"Embedding backend '{backend}' is"
                    f" not implemented yet."
                ),
                error_code="EMBEDDING_UNIMPLEMENTED_BACKEND",
                metadata={"requested_backend": backend},
            )
        else:
            raise EmbeddingException(
                message=f"Embedding backend '{backend}' is not supported.",
                error_code="EMBEDDING_UNSUPPORTED_BACKEND",
                metadata={"requested_backend": backend},
            )

    def __getattr__(self, name):
        """
        Delegate calls to the selected backend client.

        For instance, if you call :meth:`get_embedding` on an
        :class:`EmbeddingClient` instance, it will route to the current
        backend's implementation (e.g. :class:`OpenAIEmbeddingClient`).

        :param name: Name of the method or attribute to be retrieved.
        :type name: str
        :returns: The underlying backend client's method or attribute.
        """
        return getattr(self._client, name)
