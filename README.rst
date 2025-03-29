===========================
Darca Space Git
===========================

**darca-space-git** integrates Git operations into space-managed environments from `darca-space-manager`.

It acts as a high-level bridge between the `darca-git` Git abstraction and space/file management.

|Build Status| |Deploy Status| |CodeCov| |Formatting| |License| |PyPi Version| |Docs|

.. |Build Status| image:: https://github.com/roelkist/darca-space-git/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/roelkist/darca-space-git/actions
.. |Deploy Status| image:: https://github.com/roelkist/darca-space-git/actions/workflows/cd.yml/badge.svg
   :target: https://github.com/roelkist/darca-space-git/actions
.. |Codecov| image:: https://codecov.io/gh/roelkist/darca-space-git/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/roelkist/darca-space-git
   :alt: Codecov
.. |Formatting| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black code style
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
.. |PyPi Version| image:: https://img.shields.io/pypi/v/darca-space-git
   :target: https://pypi.org/project/darca-space-git/
   :alt: PyPi
.. |Docs| image:: https://img.shields.io/github/deployments/roelkist/darca-space-git/github-pages
   :target: https://roelkist.github.io/darca-space-git/
   :alt: GitHub Pages

Features
========

- ⛺ Git operations are scoped to logical spaces
- 🔒 No direct file system access required
- 🧪 Full test coverage & modular design
- 💡 Supports dry-run & file-level restore
- 🔁 Clean separation of Git vs Space responsibilities

Install
=======

Install everything needed:

.. code-block:: bash

    make install

Usage
=====

.. code-block:: python

    from darca_space_git.space_git import SpaceGitManager

    git_mgr = SpaceGitManager()
    git_mgr.init_repo("myspace")
    git_mgr.get_status("myspace")

Run Tests
=========

.. code-block:: bash

    make test

Run Pre-Commit Hooks
====================

.. code-block:: bash

    make precommit

Run Everything
==============

.. code-block:: bash

    make check
