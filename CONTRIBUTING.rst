Contributing to darca-embeddings
================================

Thank you for your interest in improving **darca-embeddings**!

Code of Conduct
---------------
Please note that we follow a standard open-source code of conduct. 
Be kind and constructive in all your interactions.

Getting Started
---------------
1. **Create a feature branch** from the main branch:
   
   .. code-block:: console

      git checkout main
      git pull
      git checkout -b feature/my-new-feature

2. **Make changes** in your feature branch.
3. **Test** your changes locally by running partial or full checks:
   - **`make format`**: Auto-formats code via `isort` and `black`.
   - **`make precommit`**: Runs pre-commit hooks (lint checks, etc.).
   - **`make test`**: Runs the test suite (includes coverage).
   - **`make docs`**: Builds the Sphinx documentation.
   - **`make check`**: Runs install, format, pre-commit, test, and docs in one go.

   You can use individual commands to speed up development when fixing smaller things.

4. **Commit** your changes. 
   - Ensure all checks pass with `make check` before pushing.

5. **Open a Pull Request** for review.
   - Use the provided feature or issue templates if applicable.
   - |
   Provide a clear description of the changes, referencing any issues or
   discussions that they address.

Tips & Tricks
-------------
- If you need new dependencies, add them with:

  .. code-block:: console

     make add-deps group=dev deps="some-package"

  This will keep your Poetry project organized.
- If you want a production dependency, use:

  .. code-block:: console

     make add-prod-deps deps="some-package"

Contact
-------
If you have any questions or want to discuss a feature, open an issue in the repository 
or reach out directly to **Roel Kist**. We appreciate your support and contributions!
