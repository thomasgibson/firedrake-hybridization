from __future__ import absolute_import, print_function, division
from six import iteritems


class Parameters(object):
    """Class for storing parameter sets."""

    def __init__(self, name, parameter_dict):
        """The parameters passed to firedrake are given
        as a dictionary object.

        :arg name: A ``str`` denoting the name of the
                   parameter set.
        :arg parameter_dict: A ``dict`` of the parameters.
        """

        super(Parameters, self).__init__()
        self.name = name
        self._data = parameter_dict

    def __getitem__(self, key):
        """Gets a value of a particular key.

        :arg key: A key to look for.
        """

        if key not in self._data:
            raise KeyError("Parameter key %s not found." % key)

        return self._data[key]

    def __str__(self):
        """Converts the parameters into a string representation."""

        result = str(self.name) + ':\n'
        for key, value in iteritems(self._data):
            result += "    " + str(key) + " = " + str(value) + "\n"

        return result

    # TODO: Add read from file functionality
