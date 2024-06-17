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

{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

..
   ############################################################################
   Functions
   ############################################################################

{% block functions %}
{% if functions %}
.. rubric:: {{ _('Functions') }}

{% for item in functions %}
{{ item | escape | underline(line='-')}}
.. autosummary::
   :toctree:
   :nosignatures:
   :hide_from_toctree:

   ~{{ fullname }}.{{ item }}

{% endfor %}
{% endif %}
{% endblock %}

..
   ############################################################################
   Attributes
   ############################################################################

{% block attributes %}
{% if attributes %}
.. rubric:: {{ _('Module Attributes') }}

{% for item in attributes %}
{{ item | escape | underline(line='-')}}
.. autosummary::
   :toctree:
   :nosignatures:
   :hide_from_toctree:

   ~{{ fullname }}.{{ item }}

{% endfor %}
{% endif %}
{% endblock %}

..
   ############################################################################
   Classes
   ############################################################################

{% block classes %}
{% if classes %}
.. rubric:: {{ _('Classes') }}

{% for item in classes %}
{{ item | escape | underline(line='-')}}
.. autosummary::
   :toctree:
   :nosignatures:
   :hide_from_toctree:
   :template: _custom_autosummary_class.rst

   ~{{ fullname }}.{{ item }}

{% endfor %}
{% endif %}
{% endblock %}

..
   ############################################################################
   Exceptions
   ############################################################################

{% block exceptions %}
{% if exceptions %}
.. rubric:: {{ _('Exceptions') }}

{% for item in exceptions %}
{{ item | escape | underline(line='-')}}
.. autosummary::
   :toctree:
   :nosignatures:
   :hide_from_toctree:

   ~{{ fullname }}.{{ item }}

{% endfor %}
{% endif %}
{% endblock %}

..
   ############################################################################
   Sub modules
   ############################################################################

{% block modules %}
{% if modules %}
.. rubric:: Modules

{% for item in modules %}
{{ item | escape | underline(line='-')}}
.. autosummary::
   :toctree:
   :hide_from_toctree:
   :nosignatures:
   :template: _custom_autosummary_module.rst
   :recursive:

   {{ item }}
{% endfor %}
{% endif %}
{% endblock %}


.. raw:: html

   </div>
