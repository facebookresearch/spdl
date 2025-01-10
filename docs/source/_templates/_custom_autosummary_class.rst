{{ fullname | escape | underline}}

{%- set meths = [] %}
{%- for item in methods %}
{%- if item != "__init__" %}
{%- set meths = meths.append(item) %}
{%- endif %}
{%- endfor %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   {%- if module.startswith("spdl.source") %}
   :show-inheritance:
   {%- endif %}
   :members:

   {%- block meths %}
   {%- if meths %}

   .. rubric:: {{ _('Methods') }}

   .. autosummary::

      {%- for item in meths %}
      ~{{ name }}.{{ item }}
      {%- endfor %}
   {%- endif %}
   {%- endblock %}
   {%- block attributes %}
   {%- if attributes %}

   .. rubric:: {{ _('Attributes') }}

   .. autosummary::

      {%- for item in attributes %}
      ~{{ name }}.{{ item }}
      {%- endfor %}

   {%- endif %}
   {%- endblock %}
