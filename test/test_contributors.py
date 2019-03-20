import unittest
from io import StringIO, BytesIO
from textwrap import dedent
from unittest.mock import patch

import contrib.contributors as contrib


class TestContributors(unittest.TestCase):
    def setUp(self):
        self.contrib_file_lines = dedent("""\
            ============
            Contributors
            ============

            This file contains a list of every person who has contributed code to
            Overviewer.

            ---------------
            Original Author
            ---------------

             * Andrew Brown <brownan@gmail.com>

            -------------------------
            Long-term Contributions
            -------------------------

            These contributors have made many changes, over a fairly long time span, or
            for many different parts of the code.

             * Alejandro Aguilera <fenixin@lavabit.com>

            ------------------------
            Short-term Contributions
            ------------------------

            These contributors have made specific changes for a particular bug fix or
            feature.

             * 3decibels <3db@3decibels.net>""").split("\n")

    def test_format_contributor_single_name(self):
        contributor = {"name": "John", "email": "<john@gmail.com>"}
        self.assertEqual(
            contrib.format_contributor(contributor),
            " * John <john@gmail.com>"
        )

    def test_format_contributor_multiple_names(self):
        contributor = {"name": "John Smith", "email": "<john@gmail.com>"}
        self.assertEqual(
            contrib.format_contributor(contributor),
            " * John Smith <john@gmail.com>"
        )

    def test_get_old_contributors(self):
        expected = [{"name": "Andrew Brown", "email": "<brownan@gmail.com>"},
                    {"name": "Alejandro Aguilera", "email": "<fenixin@lavabit.com>"},
                    {"name": "3decibels", "email": "<3db@3decibels.net>"}]

        self.assertListEqual(contrib.get_old_contributors(self.contrib_file_lines), expected)

    @patch('subprocess.run')
    def test_get_contributors(self, mock_run):
        mock_run.return_value.stdout = dedent("""\
            1	3decibels <3db@3decibels.net>
            585	Aaron Griffith <aargri@gmail.com>
            1	Aaron1011 <aa1ronham@gmail.com>
            """).encode()
        expected = [{"count": 1, "name": "3decibels", "email": "<3db@3decibels.net>"},
                    {"count": 585, "name": "Aaron Griffith", "email": "<aargri@gmail.com>"},
                    {"count": 1, "name": "Aaron1011", "email": "<aa1ronham@gmail.com>"}]
        self.assertListEqual(contrib.get_contributors(), expected)

    def test_get_new_contributors_new_contributors_alphabetical_order(self):
        contributors = [{"count": 1, "name": "3decibels", "email": "<3db@3decibels.net>"},
                        {"count": 585, "name": "Aaron Griffith", "email": "<aargri@gmail.com>"},
                        {"count": 1, "name": "Aaron1011", "email": "<aa1ronham@gmail.com>"}]

        old_contributors = [{"name": "Andrew Brown", "email": "<brownan@gmail.com>"},
                            {"name": "Alejandro Aguilera", "email": "<fenixin@lavabit.com>"},
                            {"name": "3decibels", "email": "<3db@3decibels.net>"}]

        new_contributors, new_alias, new_email = contrib.get_new_contributors(
            contributors, old_contributors)

        self.assertListEqual(new_contributors, [{"count": 1, "name": "Aaron1011", "email": "<aa1ronham@gmail.com>"}, {
                             "count": 585, "name": "Aaron Griffith", "email": "<aargri@gmail.com>"}])

    def test_get_new_contributors_new_alias(self):
        contributors = [{"count": 1, "name": "new_name", "email": "<3db@3decibels.net>"},
                        {"count": 585, "name": "Aaron Griffith", "email": "<aargri@gmail.com>"},
                        {"count": 1, "name": "Aaron1011", "email": "<aa1ronham@gmail.com>"}]

        old_contributors = [{"name": "Andrew Brown", "email": "<brownan@gmail.com>"},
                            {"name": "Alejandro Aguilera", "email": "<fenixin@lavabit.com>"},
                            {"name": "3decibels", "email": "<3db@3decibels.net>"}]

        new_contributors, new_alias, new_email = contrib.get_new_contributors(
            contributors, old_contributors)
        self.assertListEqual(
            new_alias, [({"count": 1, "name": "new_name", "email": "<3db@3decibels.net>"}, "3decibels")])

    def test_get_new_contributors_new_email(self):
        contributors = [{"count": 1, "name": "3decibels", "email": "<3db@3decibels.com>"},
                        {"count": 585, "name": "Aaron Griffith", "email": "<aargri@gmail.com>"},
                        {"count": 1, "name": "Aaron1011", "email": "<aa1ronham@gmail.com>"}]

        old_contributors = [{"name": "Andrew Brown", "email": "<brownan@gmail.com>"},
                            {"name": "Alejandro Aguilera", "email": "<fenixin@lavabit.com>"},
                            {"name": "3decibels", "email": "<3db@3decibels.net>"}]

        new_contributors, new_alias, new_email = contrib.get_new_contributors(
            contributors, old_contributors)
        self.assertListEqual(
            new_email, [({"count": 1, "name": "3decibels", "email": "<3db@3decibels.com>"}, "<3db@3decibels.net>")])

    def test_merge_short_term_contributors(self):
        new_contributors = [{"count": 1, "name": "Aaron1011", "email": "<aa1ronham@gmail.com>"}, {
            "count": 585, "name": "Aaron Griffith", "email": "<aargri@gmail.com>"}]
        expected = ['============',
                    'Contributors',
                    '============',
                    '',
                    'This file contains a list of every person who has contributed code to',
                    'Overviewer.',
                    '',
                    '---------------',
                    'Original Author',
                    '---------------',
                    '',
                    ' * Andrew Brown <brownan@gmail.com>',
                    '',
                    '-------------------------',
                    'Long-term Contributions',
                    '-------------------------',
                    '',
                    'These contributors have made many changes, over a fairly long time span, or',
                    'for many different parts of the code.',
                    '',
                    ' * Alejandro Aguilera <fenixin@lavabit.com>',
                    '',
                    '------------------------',
                    'Short-term Contributions',
                    '------------------------',
                    '',
                    'These contributors have made specific changes for a particular bug fix or',
                    'feature.',
                    '',
                    ' * 3decibels <3db@3decibels.net>',
                    ' * Aaron1011 <aa1ronham@gmail.com>\n',
                    ' * Aaron Griffith <aargri@gmail.com>\n']

        self.assertListEqual(contrib.merge_short_term_contributors(
            self.contrib_file_lines, new_contributors), expected)


if __name__ == "__main__":
    unittest.main()
