pydap_extras
============

Consolidates all PCIC Pydap handlers and responses. Uses the mainline Pydap for Python 3.

Installation
============
.. code-block:: bash
  
   pipenv install
   # or for development
   pipenv install --dev  
Tests
=====
.. code-block:: bash
  
   pipenv run pytest
App
===
The app will run on port 8001.

.. code-block:: bash
  
   pip install -r test_requirements.txt
   python pydap_extras/app.py [filepath]
