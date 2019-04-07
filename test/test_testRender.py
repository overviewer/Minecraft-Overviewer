import tempfile
import unittest
from unittest.mock import patch
from subprocess import CalledProcessError, PIPE, STDOUT
import contrib.testRender as test_render
from io import StringIO
from shlex import split


class TestTestRender(unittest.TestCase):
    @patch("contrib.testRender.run")
    def test_check_call_raises_CalledProcessError_from_subprocess_run(self, m_run):
        m_run.side_effect = CalledProcessError(1, "python program.js")
        with self.assertRaises(CalledProcessError):
            test_render.check_call(["python", "program.js"])

    @patch("contrib.testRender.run")
    def test_check_call_captures_stdout_if_not_verbose(self, m_run):
        test_render.check_call(["python", "program.py"])
        args, kwargs = m_run.call_args
        self.assertEqual(kwargs['stdout'], PIPE)
        self.assertEqual(kwargs['stderr'], STDOUT)

    @patch("contrib.testRender.run")
    def test_check_call_does_not_capture_stdout_if_verbose(self, m_run):
        test_render.check_call(["python", "program.py"], verbose=True)
        args, kwargs = m_run.call_args
        self.assertEqual(kwargs['stdout'], None)
        self.assertEqual(kwargs['stderr'], None)

    @patch('sys.stdout', new_callable=StringIO)
    @patch("contrib.testRender.run")
    def test_check_call_prints_exception_output_if_verbose(self, m_run, m_out):
        m_run.side_effect = CalledProcessError(
            1, "python program.js", output="SyntaxError: invalid syntax")
        with self.assertRaises(CalledProcessError):
            test_render.check_call(["python", "program.js"], verbose=True)
        self.assertEqual(m_out.getvalue().strip(), "SyntaxError: invalid syntax")

    @patch("contrib.testRender.run")
    def test_check_output_captures_stdout(self, m_run):
        test_render.check_call(["python", "program.py"])
        args, kwargs = m_run.call_args
        self.assertEqual(kwargs['stdout'], PIPE)

    @patch('contrib.testRender.check_output')
    def test_get_commits(self, m_check_output):
        gitrange = '2eca1a5fb5fa7eeb5494abb350cd535f67acfb8b..08a86a52abfabd59ac68b37dc7e5270bd7fb328a'
        m_check_output.return_value = (
            "commit 2eca1a5fb5fa7eeb5494abb350cd535f67acfb8b\nAuthor: Andrew "
            "<andrew@fry.(none)>\nDate:   Sun Aug 22 10:16:10 2010 -0400\n\n "
            "   initial comit\n\n:000000 100644 0000000 c398ada A\tchunk.py\n:000000 "
            "100644 0000000 d5ee6ed A\tnbt.py\n:000000 100644 0000000 8fc65c9 A\ttextures.py\n:"
            "000000 100644 0000000 6934326 A\tworld.py\n\ncommit 08a86a52abfabd59ac68b37dc7e5270bd7fb328a"
            "\nAuthor: Andrew <andrew@fry.(none)>\nDate:   Tue Aug 24 21:11:57 2010 -0400\n\n    "
            "uses multiprocessing to speed up rendering. Caches chunks\n\n:1"
        )

        result = list(test_render.get_commits(gitrange))
        self.assertListEqual(result, ['2eca1a5fb5fa7eeb5494abb350cd535f67acfb8b',
                                      '08a86a52abfabd59ac68b37dc7e5270bd7fb328a'])

    @patch('contrib.testRender.check_output', return_value="my-feature-branch")
    def test_get_current_branch(self, m_check_output):
        self.assertEqual(test_render.get_current_branch(), "my-feature-branch")

    @patch('contrib.testRender.check_output', return_value="HEAD")
    def test_get_current_branch_returns_none_for_detached_head(self, m_check_output):
        self.assertIsNone(test_render.get_current_branch())

    @patch('contrib.testRender.check_output', return_value="3f1f3d748e1c79843279ba18ab65a34368b95b67")
    def test_get_current_commit(self, m_check_output):
        self.assertEqual(
            test_render.get_current_branch(),
            "3f1f3d748e1c79843279ba18ab65a34368b95b67"
        )

    @patch('contrib.testRender.get_current_branch', return_value="my-feature-branch")
    def test_get_current_ref_returns_branch_name_if_possible(self, m_branch):
        self.assertEqual(test_render.get_current_ref(), "my-feature-branch")

    @patch('contrib.testRender.get_current_commit', return_value="3f1f3d748e1c79843279ba18ab65a34368b95b67")
    @patch('contrib.testRender.get_current_branch', return_value=None)
    def test_get_current_ref_returns_current_commit_if_no_branch(self, m_branch, m_commit):
        self.assertEqual(
            test_render.get_current_ref(),
            "3f1f3d748e1c79843279ba18ab65a34368b95b67"
        )

    @patch('contrib.testRender.check_output')
    def test_get_commits(self, m_check_output):
        m_check_output.return_value = "\n".join(
            [
                "41ceaeab58473416bb79680ab21211764e6f1908",
                "a4d0daa91c25a51ca95182301e503c020900dafe",
                "05906c81f5778a543dfab14e77231db0a99bae24",
            ]
        )
        gitrange = "41ceaeab58473416bb79680ab21211764e6f1908..05906c81f5778a543dfab14e77231db0a99bae24"
        result = list(test_render.get_commits(gitrange))
        self.assertListEqual(
            result,
            [
                "41ceaeab58473416bb79680ab21211764e6f1908",
                "a4d0daa91c25a51ca95182301e503c020900dafe",
                "05906c81f5778a543dfab14e77231db0a99bae24"
            ]
        )
