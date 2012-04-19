#!/usr/bin/python2
"""Update the contributor list

Alias handling is done by git with .mailmap
New contributors are merged in the short-term list.
Moving them to a "higher" list should be a manual process.
"""

import fileinput
from subprocess import Popen, PIPE

def format_contributor(contributor):
    return " * {0} {1}".format(
            " ".join(contributor["name"]),
            contributor["email"])


def main():
    # generate list of contributors
    contributors=[]
    p_git = Popen(["git", "shortlog", "-se"], stdout=PIPE)
    for line in p_git.stdout:
        contributors.append({
            'count': int(line.split("\t")[0].strip()),
            'name': line.split("\t")[1].split()[0:-1],
            'email': line.split("\t")[1].split()[-1]
            })

    # cache listed contributors
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

    # check which contributors are new
    new_contributors=[]
    for contributor in contributors:
        if contributor["email"] not in old_emails:
            new_contributors.append(contributor)

    # sort on the last word of the name
    new_contributors = sorted(new_contributors,
            key=lambda x: x['name'][-1].lower())

    # show new contributors to be merged to the list
    for contributor in new_contributors:
        print format_contributor(contributor)

    # merge with contributor list
    i = 0
    short_term_found = False
    for line in fileinput.input("CONTRIBUTORS.rst", inplace=1):
        if not short_term_found:
            print line,
            if "Short-term" in line:
                short_term_found = True
        else:
            if i >= len(new_contributors) or "@" not in line:
                print line,
            else:
                contributor = new_contributors[i]
                if line.split()[-2] > contributor["name"][-1]:
                    print format_contributor(contributor)
                    i += 1
                print line,
    # append remaining contributors
    with open("CONTRIBUTORS.rst", "a") as contrib_file:
        while i < len(new_contributors):
            contrib_file.write(format_contributor(new_contributors[i]) + "\n")
            i += 1


if __name__ == "__main__":
    main()
