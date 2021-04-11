Conventions
===========

* Files and folders: all `.py` files are in the current directory, data are in the `data/` subdirectory.
* The `.py` files whose name does not start with `exp_` nor `test_` contain functions and classes, without lines of code to be executed outside: they contain in particular all the algorithms and functions to access the data
* Scripts: `.py' files whose name starts with `exp_` are scripts that can be that can be executed.
* Tests: the `.py` files whose name starts with `test_` are test files test files, the `test_xxx.py` file contains the tests for the code of the `xxx.py` file
* Comments are used to document the code. Lines of code should not be disabled/enabled by putting or deleting comments at the beginning of those lines.

Tests
=====

You can run the tests in several ways.

1/ install the `spyder-unittest` package with `conda` or `pip` and use the menu added to `spyder`.

2/ without `spyder-unittest`, from Spyder: open the test file in the editor, press the "Run file" button

3/ in a terminal, go to the main directory (the directory), then type

python -m unittest

4/ in a terminal, go to the main directory (the parent directory of parent directory of `tests` and `song_dating`), then type

coverage run -m unittest

this runs the tests and also performs a coverage of the code. You can then then see the coverage report via the command

coverage report

command, which displays the report in the terminal, or via the commands

coverage html
open htmlcov/index.html

which builds a more detailed report in the form of a web page (first
command) and opens it in a web browser (second command).
