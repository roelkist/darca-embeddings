===============================
Contributing to Darca Space Git
===============================

We welcome community contributions to **darca-space-git**!

How to Contribute
=================

1. Fork the repo
2. Create a new branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

3. Install the project:

   .. code-block:: bash

      make install

4. Make changes and validate them:

   .. code-block:: bash

      make check

5. Open a **pull request** against `main`.

Branch Naming
=============

Use semantic naming:

- `feature/...` for new functionality
- `fix/...` for bug fixes
- `docs/...` for doc-only updates

Pull Request Requirements
=========================

- ✅ Pass `make check` (format, test, lint, docs)
- 🔍 Be focused and concise
- 🧪 Include tests for your feature or fix

Tips
====

- Use `make format` to reformat code with Black and isort
- Dry-run logic is supported in most Git operations — test it!
- Ask questions or request early feedback anytime 🚀
