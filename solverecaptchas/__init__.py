#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import sys

version_info = (0, 1, 6)
__version__ = "{}.{}.{}".format(*version_info)


authors = (("Michael Mooney", "me@mikey.id"),)

authors_email = ", ".join("{}".format(email) for _, email in authors)

__license__ = "GPL-3.0"
__author__ = ", ".join(
    "{} <{}>".format(name, email) for name, email in authors
)

package_info = (
    "An asynchronized Python library to automate solving ReCAPTCHA v2"
)
__maintainer__ = __author__

__all__ = (
    "__author__",
    "__author__",
    "__license__",
    "__maintainer__",
    "__version__",
    "version_info",
    "package_dir",
    "package_info",
)

sys.path.append(os.getcwd())
package_dir = os.path.dirname(os.path.abspath(__file__))
