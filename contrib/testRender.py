#!/usr/bin/env python3

"Test Render Script"

import argparse
import math
import os
import re
import shutil
import sys
import tempfile
import time
from shlex import split
from subprocess import PIPE, STDOUT, CalledProcessError, run

overviewer_scripts = ['./overviewer.py', './gmap.py']


def check_call(args, verbose=False):
    try:
        return run(
            args,
            check=True,
            stdout=None if verbose else PIPE,
            stderr=None if verbose else STDOUT,
            universal_newlines=True,
        )
    except CalledProcessError as e:
        if verbose:
            print(e.output)
        raise e


def check_output(args):
    p = run(
        args,
        check=True,
        stdout=PIPE,
        universal_newlines=True
    )
    return p.stdout


def clean_render(overviewerargs, verbose=False):
    tempdir = tempfile.mkdtemp('mc-overviewer-test')
    overviewer_script = None
    for script in overviewer_scripts:
        if os.path.exists(script):
            overviewer_script = script
            break
    if overviewer_script is None:
        sys.stderr.write("could not find main overviewer script\n")
        sys.exit(1)

    try:
        # check_call raises CalledProcessError when overviewer.py exits badly
        check_call([sys.executable] + split("setup.py clean build"), verbose=verbose)
        try:
            check_call([sys.executable, overviewer_script, '-d'] + overviewerargs, verbose=verbose)
        except CalledProcessError:
            pass
        starttime = time.time()
        check_call([sys.executable, overviewer_script] +
                   overviewerargs + [tempdir, ], verbose=verbose)
        endtime = time.time()

        return endtime - starttime
    finally:
        shutil.rmtree(tempdir, True)


def get_stats(timelist):
    average = sum(timelist) / float(len(timelist))
    meandiff = [(x - average) ** 2 for x in timelist]
    sd = math.sqrt(sum(meandiff) / len(meandiff))
    return {
        "count": len(timelist),
        "minimum": min(timelist),
        "maximum": max(timelist),
        "average": average,
        "standard deviation": sd
    }


def get_current_branch():
    gittext = check_output(split('git rev-parse --abbrev-ref HEAD'))
    return gittext.strip() if gittext != "HEAD" else None


def get_current_commit():
    gittext = check_output(split('git rev-parse HEAD'))
    return gittext.strip() if gittext else None


def get_current_ref():
    branch = get_current_branch()
    if branch:
        return branch

    commit = get_current_commit()
    if commit:
        return commit


def get_commits(gitrange):
    gittext = check_output(split('git rev-list --reverse') + [gitrange, ])
    return (c for c in gittext.split("\n"))


def set_commit(commit):
    check_call(split('git checkout') + [commit, ])


def main(args):
    commits = []
    for commit in args.commits:
        if '..' in commit:
            commits = get_commits(commit)
        else:
            commits.append(commit)
    if not commits:
        commits = [get_current_ref(), ]

    log = None
    if args.log:
        log = args.log

    reset_commit = get_current_ref()
    try:
        for commit in commits:
            print("testing commit", commit)
            set_commit(commit)
            timelist = []
            print(" -- "),
            try:
                for i in range(args.number):
                    sys.stdout.write(str(i + 1) + " ")
                    sys.stdout.flush()
                    timelist.append(clean_render(args.overviewer_args, verbose=args.verbose))
                print("... done")
                stats = get_stats(timelist)
                print(stats)
                if log:
                    log.write("%s %s\n" % (commit, repr(stats)))
            except CalledProcessError as e:
                if args.fatal_errors:
                    print(e)
                    print("Overviewer croaked, exiting...")
                    print("(to avoid this, use --keep-going)")
                    sys.exit(1)
    finally:
        set_commit(reset_commit)
        if log:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overviewer_args", metavar="[overviewer options/world]", nargs="+")
    parser.add_argument("-n", "--option", metavar="N", type=int, action="store",
                        dest="number", default=3, help="number of renders per commit [default: 3]")
    parser.add_argument("-c", "--commits", metavar="RANGE",
                        action="append", type=str, dest="commits", default=[],
                        help="the commit (or range of commits) to test [default: current]")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                        help="don't suppress overviewer output")
    parser.add_argument("-k", "--keep-going",
                        action="store_false", dest="fatal_errors", default=True,
                        help="don't stop testing when Overviewer croaks")
    parser.add_argument("-l", "--log", dest="log", type=argparse.FileType('w'), metavar="FILE",
                        help="log all test results to a file")

    args = parser.parse_args()
    main(args)
