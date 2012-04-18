#!/usr/bin/python2
"""List contributors that are not yet in the contributor list

Alias handling is done by git with .mailmap
"""

import sys
from subprocess import Popen, PIPE

def main():
    if len(sys.argv) > 1:
        branch = sys.argv[1]
    else:
        branch = "master"

    contributors=[]
    p_git = Popen(["git", "shortlog", "-se", branch], stdout=PIPE)
    for line in p_git.stdout:
        contributors.append({
            'count': int(line.split("\t")[0].strip()),
            'name': line.split("\t")[1].split()[0:-1],
            'email': line.split("\t")[1].split()[-1]
            })

    old_contributors=[]
    with open("CONTRIBUTORS.rst", "r") as contrib_file:
        for line in contrib_file:
            if "@" in line:
                old_contributors.append({
                    'name': line.split()[1:-1],
                    'email': line.split()[-1]
                    })
    # We don't access the name of old/listed contributors at all
    # but that might change.
    # So we parse it anyways and strip it off again.
    old_emails = map(lambda x: x['email'], old_contributors)

    new_contributors=[]
    for contributor in contributors:
        if contributor["email"] not in old_emails:
            new_contributors.append(contributor)

    # sort on the last word of the name
    new_contributors = sorted(new_contributors,
            key=lambda x: x['name'][-1].lower())
    for contributor in new_contributors:
        print "{0:3d} {1:25s} {2}".format(contributor["count"],
                " ".join(contributor["name"]), contributor["email"])

if __name__ == "__main__":
    main()
