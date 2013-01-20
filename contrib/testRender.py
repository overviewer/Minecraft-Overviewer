#!/usr/bin/python

"Test Render Script"

import os, shutil, tempfile, time, sys, math, re
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
from optparse import OptionParser

overviewer_scripts = ['./overviewer.py', './gmap.py']

def check_call(*args, **kwargs):
    quiet = False
    if "quiet" in kwargs.keys():
        quiet = kwargs["quiet"]
        del kwargs["quiet"]
    if quiet:
        kwargs['stdout'] = PIPE
        kwargs['stderr'] = STDOUT
    p = Popen(*args, **kwargs)
    output = ""
    if quiet:
        while p.poll() == None:
            output += p.communicate()[0]
    returncode = p.wait()
    if returncode:
        if quiet:
            print output
        raise CalledProcessError(returncode, args)
    return returncode

def check_output(*args, **kwargs):
    kwargs['stdout'] = PIPE
    # will hang for HUGE output... you were warned
    p = Popen(*args, **kwargs)
    returncode = p.wait()
    if returncode:
        raise CalledProcessError(returncode, args)
    return p.communicate()[0]

def clean_render(overviewerargs, quiet):
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
        check_call([sys.executable, 'setup.py', 'clean', 'build'], quiet=quiet)
        try:
            check_call([sys.executable, overviewer_script, '-d'] + overviewerargs, quiet=quiet)
        except CalledProcessError:
            pass
        starttime = time.time()
        check_call([sys.executable, overviewer_script,] + overviewerargs + [tempdir,], quiet=quiet)
        endtime = time.time()
        
        return endtime - starttime
    finally:
        shutil.rmtree(tempdir, True)

def get_stats(timelist):
    stats = {}
    
    stats['count'] = len(timelist)
    stats['minimum'] = min(timelist)
    stats['maximum'] = max(timelist)
    stats['average'] = sum(timelist) / float(len(timelist))
    
    meandiff = map(lambda x: (x - stats['average'])**2, timelist)
    stats['standard deviation'] = math.sqrt(sum(meandiff) / float(len(meandiff)))
    
    return stats

commitre = re.compile('^commit ([a-z0-9]{40})$', re.MULTILINE)
branchre = re.compile('^\\* (.+)$', re.MULTILINE)
def get_current_commit():
    gittext = check_output(['git', 'branch'])
    match = branchre.search(gittext)
    if match and not ("no branch" in match.group(1)):
        return match.group(1)
    gittext = check_output(['git', 'show', 'HEAD'])
    match = commitre.match(gittext)
    if match == None:
        return None
    return match.group(1)

def get_commits(gitrange):
    gittext = check_output(['git', 'log', '--raw', '--reverse', gitrange])
    for match in commitre.finditer(gittext):
        yield match.group(1)

def set_commit(commit):
    check_call(['git', 'checkout', commit], quiet=True)

parser = OptionParser(usage="usage: %prog [options] -- [overviewer options/world]")
parser.add_option("-n", "--number", metavar="N",
                  action="store", type="int", dest="number", default=3,
                  help="number of renders per commit [default: 3]")
parser.add_option("-c", "--commits", metavar="RANGE",
                  action="append", type="string", dest="commits", default=[],
                  help="the commit (or range of commits) to test [default: current]")
parser.add_option("-v", "--verbose",
                  action="store_false", dest="quiet", default=True,
                  help="don't suppress overviewer output")
parser.add_option("-k", "--keep-going",
                  action="store_false", dest="fatal_errors", default=True,
                  help="don't stop testing when Overviewer croaks")
parser.add_option("-l", "--log", dest="log", default="", metavar="FILE",
                  help="log all test results to a file")

(options, args) = parser.parse_args()

if len(args) == 0:
    parser.print_help()
    sys.exit(0)

commits = []
for commit in options.commits:
    if '..' in commit:
        commits = get_commits(commit)
    else:
        commits.append(commit)
if not commits:
    commits = [get_current_commit(),]

log = None
if options.log != "":
    log = open(options.log, "w")

reset_commit = get_current_commit()
try:
    for commit in commits:
        print "testing commit", commit
        set_commit(commit)
        timelist = []
        print " -- ",
        try:
            for i in range(options.number):
                sys.stdout.write(str(i+1)+" ")
                sys.stdout.flush()
                timelist.append(clean_render(args, options.quiet))
            print "... done"
            stats = get_stats(timelist)
            print stats
            if log:
                log.write("%s %s\n" % (commit, repr(stats)))
        except CalledProcessError, e:
            if options.fatal_errors:
                print
                print "Overviewer croaked, exiting..."
                print "(to avoid this, use --keep-going)"
                sys.exit(1)
finally:
    set_commit(reset_commit)
    if log:
        log.close()
