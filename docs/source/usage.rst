===============
Usage Guide
===============

**darca-space-git** provides Git version control within space-managed environments.

Install
=======

To set up the project and all its dependencies:

.. code-block:: bash

    make install

Basic Workflow
==============

.. code-block:: python

    from darca_space_git.space_git import SpaceGitManager

    git_mgr = SpaceGitManager()
    git_mgr.init_repo("myspace")
    git_mgr.commit_all("myspace", "Initial commit")

Dry-Run Support
===============

Some operations (e.g. checkout) support a `dry_run=True` mode:

.. code-block:: python

    git_mgr.checkout_path("myspace", ["README.md"], dry_run=True)

This lets you preview changes before applying them.

Testing
=======

.. code-block:: bash

    make test

This will run all tests, generate HTML coverage, and a coverage badge.
