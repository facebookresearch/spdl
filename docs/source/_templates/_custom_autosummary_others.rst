{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{%- if objtype == "data" %}

.. autodata:: {{ objname }}
   :annotation:
   :no-value:

{%- else %}

.. autoattribute:: {{ objname }}

{% endif %}
