pydap_extras
============

Consolidates all PCIC Pydap handlers and responses. Uses the mainline Pydap for Python 3.

Installation
============
.. code-block:: bash
  
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .   
Tests
=====
.. code-block:: bash
  
   pip install -r test_requirements.txt
   pytest
App
===
The app will run on port 8001.

.. code-block:: bash
  
   pip install -r test_requirements.txt
   python pydap_extras/app.py [filepath]
