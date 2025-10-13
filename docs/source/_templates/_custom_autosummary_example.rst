..
   Custom template for our custom_autosummary.
   The purpose of using custom_autosummary is list the entries on
   the right side bar instead of the left side bar.
   This is achieved with the combination of custom directive, template and CSS.

   The `custom_autosummary` directive functions mostly like `autosummary`. We
   pass `:toctree:` option to generate a doc page for each entry, but we use
   our custom `:hide_from_toctree:` option so as not to attach them in ToC tree.

   This template inserts a section header for each entry, so that they show up
   in the right-side bar. The headers are hidden by CSS.

   Because each entry will have its own table and their borders no longer match,
   table borders are hidden by CSS and instead we use horizontal line.

.. raw:: html

   <div class="custom_autosummary">

.. _example-{{ name.replace('_', '-')}}:

{{ name.capitalize().replace('_', ' ') | escape | underline }}

.. automodule:: {{ fullname }}

..
   ############################################################################
   Source
   ############################################################################

Source
------

.. rubric:: {{ _('Source') }}

.. raw:: html

   <details>
   <summary><u>Click here to see the source.</u></summary>

.. literalinclude:: ../../../examples/{{ fullname | replace(".", "/")}}.py
   :linenos:

.. raw:: html

   </details>

..
   ############################################################################
   Functions
   ############################################################################

{% block functions %}
{% if functions %}

Functions
---------
.. rubric:: {{ _('Functions') }}

{% for item in functions %}

.. autofunction:: {{ fullname }}.{{ item }}

{% endfor %}
{% endif %}
{% endblock %}

..
   ############################################################################
   Attributes
   ############################################################################

{% block attributes %}
{% if attributes %}
Attributes
----------
.. rubric:: {{ _('Module Attributes') }}

{% for item in attributes %}
{{ item | escape | underline(line='^')}}

.. autoattribute:: {{ fullname }}.{{ item }}

{% endfor %}
{% endif %}
{% endblock %}

..
   ############################################################################
   Classes
   ############################################################################

{% block classes %}
{% if classes %}

Classes
-------

.. rubric:: {{ _('Classes') }}

{% for item in classes %}

.. autoclass:: {{ fullname }}.{{ item }}
   :members:

{% endfor %}
{% endif %}
{% endblock %}

.. raw:: html

   </div>
